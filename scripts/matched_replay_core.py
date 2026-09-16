"""Fixed mask interventions and past-only paired forecast-risk decisions."""

import hashlib

import numpy as np

RULES = ("generic", "shuffled", "matched")
TIE_TOLERANCE = 1e-10


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def run_signature(mask):
    result = []
    for column in np.asarray(mask, bool).T:
        changes = np.diff(np.r_[False, column, False].astype(int))
        result.append(
            tuple(sorted((np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)).tolist()))
        )
    return tuple(result)


def mask_rules(mask, key):
    mask = np.asarray(mask, bool)
    if mask.shape[0] != 96 or mask.ndim != 2:
        raise ValueError("the registered query mask has length 96")
    signature = run_signature(mask)
    shifts = [
        amount
        for amount in range(1, 96)
        if run_signature(np.roll(mask, amount, axis=0)) == signature
        and not np.array_equal(np.roll(mask, amount, axis=0), mask)
    ]
    if not shifts:
        return None, {"reason": "no_distinct_run_preserving_position_shift"}
    rng = np.random.default_rng(stable_seed("r19-mask|" + key))
    amount = int(shifts[int(rng.integers(len(shifts)))])
    generic = np.zeros_like(mask)
    for column, count in enumerate(mask.sum(0)):
        generic[rng.choice(96, int(count), replace=False), column] = True
    masks = np.stack([generic, np.roll(mask, amount, axis=0), mask])
    return masks, {
        "shift": amount,
        "valid_shifts": len(shifts),
        "seed": stable_seed("r19-mask|" + key),
        "generic_matches_current": bool(np.array_equal(generic, mask)),
    }


def observed_risks(points, truth, scale):
    """Return [anchor, action, target] MAE/MSE with equal target weighting downstream."""
    observed = np.isfinite(truth)
    counts = observed.sum(axis=1)
    if (counts < 48).any() or not np.isfinite(points).all():
        raise ValueError("historical forecasts or common target observations are incomplete")
    error = np.where(
        observed[:, None], (points - truth[:, None]) / scale[None, None, None, :2], 0.0
    )
    return abs(error).sum(2) / counts[:, None], (error**2).sum(2) / counts[:, None]


def history_decisions(losses, joint):
    """Eight individual actions plus median fallback, scored from eight past blocks."""
    if losses.shape != (8, 9, 2) or not np.isfinite(losses).all():
        raise ValueError("registered past loss coverage is eight blocks, nine actions, two targets")
    risk = losses.mean(2, keepdims=True) if joint else losses
    delta = risk[:, :8] - risk[:, 8:9]
    mean = delta.mean(0)
    uncertainty = delta.std(0, ddof=1) / np.sqrt(8)
    choices = {}
    for name, penalty in (("forced", None), ("erm", 0.0), ("conservative", 1.0)):
        score = mean if penalty is None else mean + penalty * uncertainty
        minimum = score.min(0)
        best = (score <= minimum[None] + TIE_TOLERANCE).argmax(0)
        if penalty is not None:
            best = np.where(minimum < -TIE_TOLERANCE, best, 8)
        choices[name] = np.repeat(best, 2) if joint else best
    return choices, mean, uncertainty


def choose_current(bank, choices):
    if bank.shape != (9, 96, 2) or np.asarray(choices).shape != (2,):
        raise ValueError("current forecast bank and target choices disagree")
    return np.stack([bank[int(choices[slot]), :, slot] for slot in (0, 1)], axis=1)
