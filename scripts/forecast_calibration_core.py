"""Temporal source calibration, observable gate features and fixed MAE controls."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from peer_outage_core import ROOT, PrefixRegression, augmented, sources
from scipy import sparse
from scipy.optimize import linprog

from tsfm_fais.utility_experiment import file_sha256

BASE = ROOT / "artifacts/iclr27-r31"
PARENT = ROOT / "artifacts/iclr27-r30"
SEED = 5101
LEARNED = ("forecast_gate", "imputation_gate", "fixed_input_mix")
NEW_METHODS = (
    *LEARNED,
    "half_input_mix",
    "half_output_mix",
    "calibrated_output_global",
    "calibrated_output_station",
)
FEATURE_NAMES = (
    "gap_fraction",
    "target_observed",
    "peer_gap_observed",
    "local_observed_mae",
    "peer_observed_mae",
    "repair_disagreement",
    "last_observed_z",
    "last_local_z",
    "last_peer_z",
    "aux_gap_observed",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path, sha=None):
    if sha is not None and file_sha256(Path(path)) != sha:
        raise ValueError("a frozen array file changed")
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def calibration_sources():
    records, _ = sources()
    records = {
        s: {
            **r,
            "values": r["values"][: r["prefix_end"]],
            "original_prefix_end": r["prefix_end"],
            "prefix_end": r["prefix_end"] // 2,
        }
        for s, r in records.items()
    }
    peers, selection = {}, []
    for station, own in sorted(records.items()):
        for target in (0, 1):
            ranked = []
            x = own["values"][: own["prefix_end"], target]
            for other, record in sorted(records.items()):
                if other == station:
                    continue
                y = record["values"][: own["prefix_end"], target]
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 128 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
                    continue
                ranked.append(
                    {
                        "station": other,
                        "support": int(valid.sum()),
                        "abs_correlation": float(abs(np.corrcoef(x[valid], y[valid])[0, 1])),
                    }
                )
            ranked.sort(key=lambda r: (-r["abs_correlation"], r["station"]))
            if len(ranked) < 3:
                raise ValueError("insufficient half-prefix peer support")
            peers[station, target] = [r["station"] for r in ranked[:3]]
            selection.append(
                {
                    "station": station,
                    "target": target,
                    "prefix_end": own["prefix_end"],
                    "ranked": ranked,
                }
            )
    return records, peers, selection


def calibration_population(records):
    rows, eligibility = [], []
    for station, record in sorted(records.items()):
        obs = np.isfinite(record["values"][:, :2])
        eligible = [
            t
            for t in range(record["prefix_end"] + 192, len(obs) - 23, 24)
            if obs[t - 54 : t].all() and (obs[t : t + 24].sum(0) >= 12).all()
        ]
        if len(eligible) < 8:
            raise ValueError("insufficient source calibration histories")
        selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1, 8, dtype=int)]
        eligibility.append({"station": station, "eligible": eligible, "selected": selected})
        for t in selected:
            for gap in (6, 24, 54):
                key = f"r31_source|{station}|{t}|{gap}"
                rows.append(
                    {
                        "case_id": hashlib.sha256(key.encode()).hexdigest()[:20],
                        "panel": "calibration_h24",
                        "station": station,
                        "origin": t,
                        "outage_age": gap,
                        "horizon": 24,
                        "prefix_end": record["prefix_end"],
                        "calibration_end": record["original_prefix_end"],
                    }
                )
    return rows, eligibility


def calibration_context(full, row):
    x = np.array(full[row["origin"] - 192 : row["origin"]], float, copy=True)
    x[-row["outage_age"] :, :2] = np.nan
    return x


def expert_targets(data):
    names = data["stat_names"].tolist()
    return tuple(data["stat_targets"][names.index(n)] for n in ("local_ridge", "peer_ridge"))


def observable_features(data, regression):
    """Accept only the arrived context, prefix models and their two repairs."""
    x = data["context"]
    mean, scale = data["mean"], data["scale"]
    z = (x - mean) / scale
    observed = np.isfinite(x)
    local, peer = [(a - mean[:2]) / scale[:2] for a in expert_targets(data)]
    rows = []
    for target in (0, 1):
        indices = np.flatnonzero(observed[:, target])
        age = len(x) - 1 - indices[-1] if len(indices) else len(x)
        if age < 1:
            raise ValueError("registered gate contexts must end in a target gap")
        quality = []
        for limit in (11, 17):
            errors = []
            for t in indices:
                available = (np.flatnonzero(observed[t, 2:limit]) + 2).tolist()
                fit = regression.fit(target, available)
                features = fit["features"]
                predicted = (
                    fit["beta"][0] + z[t, features] @ np.asarray(fit["beta"])[1:]
                    if features
                    else 0.0
                )
                errors.append(abs(z[t, target] - predicted))
            quality.append(float(np.mean(errors)) if errors else 1.0)
        peer_start = 11 + target * 3
        rows.append(
            [
                age / len(x),
                observed[:, target].mean(),
                observed[-age:, peer_start : peer_start + 3].mean(),
                *quality,
                abs(local[-age:, target] - peer[-age:, target]).mean(),
                z[indices[-1], target] if len(indices) else 0.0,
                local[-1, target],
                peer[-1, target],
                observed[-age:, 2:11].mean(),
            ]
        )
    result = np.asarray(rows, float)
    if result.shape != (2, 10) or not np.isfinite(result).all():
        raise ValueError("gate features must be finite and observable")
    return result


def tensor_inputs(data, device="cuda"):
    mean, scale = data["mean"], data["scale"]
    local, peer = expert_targets(data)
    return {
        "original": torch.tensor(
            (data["context"] - mean) / scale, dtype=torch.float32, device=device
        ),
        "local": torch.tensor((local - mean[:2]) / scale[:2], dtype=torch.float32, device=device),
        "peer": torch.tensor((peer - mean[:2]) / scale[:2], dtype=torch.float32, device=device),
        "keep": torch.tensor(data["keep"], dtype=torch.bool, device=device),
    }


def mixed_context(data, alpha):
    repair = (1 - alpha) * data["local"] + alpha * data["peer"]
    original = data["original"]
    targets = torch.where(torch.isfinite(original[:, :2]), original[:, :2], repair)
    return torch.cat([targets, original[:, 2:]], dim=1)[:, data["keep"]]


class CalibrationGate(torch.nn.Module):
    def __init__(self, fixed=False):
        super().__init__()
        if fixed:
            self.logits = torch.nn.Parameter(torch.zeros(2))
        else:
            self.network = torch.nn.Sequential(
                torch.nn.Linear(10, 8), torch.nn.GELU(), torch.nn.Linear(8, 1)
            )
            torch.nn.init.zeros_(self.network[-1].weight)
            torch.nn.init.zeros_(self.network[-1].bias)
        self.fixed = fixed

    def forward(self, features):
        return torch.sigmoid(self.logits if self.fixed else self.network(features).squeeze(-1))


def new_gate(name, device="cuda"):
    torch.manual_seed(SEED)
    return CalibrationGate(fixed=name == "fixed_input_mix").to(device)


def normalized_features(values, mean, scale, device="cuda"):
    return torch.tensor(np.clip((values - mean) / scale, -5, 5), dtype=torch.float32, device=device)


def smooth_mae(prediction, truth):
    valid = torch.isfinite(truth)
    if not bool((valid.sum(0) > 0).all()):
        raise ValueError("each target must have an observed training label")
    safe = torch.where(valid, truth, torch.zeros_like(truth))
    losses = torch.sqrt((prediction - safe).square() + 1e-6)
    return ((losses * valid).sum(0) / valid.sum(0)).mean()


def fixed_mae_fit(predictions, truth):
    """Fit one target: [case,method,time] with equal mass per calibration case."""
    if predictions.ndim != 3 or truth.shape != predictions.shape[::2]:
        raise ValueError("unaligned fixed-control calibration inputs")
    valid = np.isfinite(truth)
    if not (valid.sum(1) > 0).all() or not np.isfinite(predictions).all():
        raise ValueError("fixed-control source support is incomplete")
    matrix = predictions.transpose(0, 2, 1)[valid].astype(float)
    labels = truth[valid].astype(float)
    weights = np.broadcast_to((1 / valid.sum(1) / len(truth))[:, None], truth.shape)[valid]
    n, k = matrix.shape
    identity = sparse.eye(n, format="csr")
    constraints = sparse.vstack(
        [sparse.hstack([matrix, -identity]), sparse.hstack([-matrix, -identity])], format="csr"
    )
    bound = np.r_[labels, -labels]
    equality = sparse.csr_matrix(np.r_[np.ones(k), np.zeros(n)][None])
    fit = linprog(
        np.r_[np.zeros(k), weights],
        A_ub=constraints,
        b_ub=bound,
        A_eq=equality,
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
    )
    if not fit.success:
        raise ValueError(f"fixed MAE optimization failed: {fit.message}")
    w = fit.x[:k]
    value = float(weights @ abs(matrix @ w - labels))
    dual = float(fit.eqlin.marginals[0] + fit.ineqlin.marginals @ bound)
    violation = max(
        float(np.max(constraints @ fit.x - bound)), abs(w.sum() - 1), float(-fit.x.min()), 0.0
    )
    if max(violation, abs(value - fit.fun), abs(fit.fun - dual)) > 1e-7:
        raise ValueError("fixed MAE primal/dual checks failed")
    return {
        "weights": w.tolist(),
        "objective": value,
        "dual_objective": dual,
        "feasibility_error": violation,
        "duality_gap": abs(fit.fun - dual),
        "cases": len(truth),
        "observations": n,
    }


def evaluation_rows():
    manifest = read_json(PARENT / "peer-inputs-v001/manifest.json")
    return [r for r in manifest["cases"] if r["horizon"] == 24]


def evaluation_sources():
    records, peers = sources()
    return {
        s: PrefixRegression(augmented(s, records, peers)[0][: r["prefix_end"]])
        for s, r in records.items()
    }
