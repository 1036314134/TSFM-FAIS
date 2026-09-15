"""Origin-separated calibration and forecast-space regularization for source gates."""

import hashlib

import numpy as np
import torch
from train_latent_source_gates import condition_features
from train_shared_forecast_gate import SETTINGS

from tsfm_fais.routing.forecast_gate import SharedForecastGate, gate_objective
from tsfm_fais.routing.forecast_projection import simplex_quadratic_weights
from tsfm_fais.routing.utility import _family_weights

EPOCHS = (1, 5, 10, 25)
LAMBDAS = (0.0, 1.0)
FEATURES = ("point", "latent")


def split_origins(frame, allowed, origin_positions):
    selected = frame.iloc[allowed]
    origins = selected.drop_duplicates("origin_id").copy()
    origins["origin"] = origins.origin_id.map(origin_positions)
    if origins.origin.isna().any():
        raise ValueError("an origin has no verified source position")
    fit_ids, calibration_ids, purged_ids = set(), set(), set()
    for _, group in origins.groupby(["family_id", "dataset_id", "item_id"]):
        group = group.sort_values(["origin", "origin_id"])
        if len(group) < 4:
            fit_ids.update(group.origin_id)
            continue
        count = int(np.ceil(len(group) / 4))
        calibration = group.iloc[-count:]
        earlier = group.iloc[:-count]
        permitted = earlier.origin + 96 <= calibration.origin.min()
        fit_ids.update(earlier.loc[permitted, "origin_id"])
        purged_ids.update(earlier.loc[~permitted, "origin_id"])
        calibration_ids.update(calibration.origin_id)
    if not fit_ids or not calibration_ids or fit_ids & calibration_ids:
        raise ValueError("the inner origin split is empty or overlapping")
    indices = [
        np.asarray([i for i in allowed if frame.iloc[i].origin_id in group], dtype=np.int64)
        for group in (fit_ids, calibration_ids, purged_ids)
    ]
    if sum(map(len, indices)) != len(allowed):
        raise ValueError("the inner split omitted source decisions")
    return indices


def regularized_objective(probability, gram, alignment, anchor, strength):
    delta = probability - anchor
    deviation = torch.einsum("na,nab,nb->n", delta, gram, delta)
    return gate_objective(probability, gram, alignment, "ensemble") + strength * deviation


def fixed_weights(frame, gram, alignment, indices):
    weights = _family_weights(frame.iloc[indices])
    weights = weights / weights.sum()
    g = np.einsum("n,nab->ab", weights, gram[indices])
    b = np.einsum("n,na->a", weights, alignment[indices])
    probability, gap, _ = simplex_quadratic_weights(g[None], b[None])
    return probability[0], float(gap[0])


def calibration_metrics(frame, point, truth):
    values = frame.assign(mae=abs(point - truth).mean(1), mse=((point - truth) ** 2).mean(1))
    episodes = values.groupby(["family_id", "source_episode_id"])[["mae", "mse"]].mean()
    return episodes.groupby("family_id").mean().mean().to_dict()


def choose_configuration(records, reference, *, early_only=False):
    candidates = []
    for row in records:
        if early_only and row["strength"] != 0:
            continue
        if row["mae"] < reference["mae"] and row["mse"] < reference["mse"]:
            merit = 0.5 * (row["mae"] / reference["mae"] + row["mse"] / reference["mse"])
            candidates.append((merit, -row["strength"], row["epoch"], row))
    if not candidates:
        return {"kind": "fixed", "strength": None, "epoch": None}
    row = min(candidates, key=lambda value: value[:3])[-1]
    return {"kind": "gate", "strength": row["strength"], "epoch": row["epoch"]}


def fit_snapshots(features, gram, alignment, frame, indices, anchor, strength, seed, epochs):
    torch.manual_seed(seed)
    model = SharedForecastGate(features=97, hidden=16)
    weights = _family_weights(frame.iloc[indices])
    model.fit_normalization(features[indices], weights)
    initial = hashlib.sha256(
        b"".join(value.detach().numpy().tobytes() for value in model.parameters())
    ).hexdigest()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"]
    )
    x, g, b, w = [
        torch.as_tensor(np.asarray(value), dtype=torch.float32)
        for value in (features[indices], gram[indices], alignment[indices], weights)
    ]
    anchor_tensor = torch.as_tensor(anchor, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 100000)
    states, history = {}, []
    for epoch in range(1, max(epochs) + 1):
        total = 0.0
        for batch in torch.randperm(len(indices), generator=generator).split(
            SETTINGS["batch_size"]
        ):
            loss = (
                regularized_objective(model(x[batch]), g[batch], b[batch], anchor_tensor, strength)
                * w[batch]
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite calibrated-gate objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["gradient_norm"])
            if not bool(torch.isfinite(norm)):
                raise ValueError("nonfinite calibrated-gate gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append({"epoch": epoch, "relative_training_loss": total / len(indices)})
        if epoch in epochs:
            states[str(epoch)] = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
    return {
        "states": states,
        "history": history,
        "initial_parameter_sha256": initial,
        "train_indices_sha256": hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest(),
        "training_origins": sorted(frame.iloc[indices].origin_id.unique()),
        "training_families": sorted(frame.iloc[indices].family_id.unique()),
        "anchor": np.asarray(anchor).tolist(),
        "strength": strength,
        "seed": seed,
    }


def inputs_for(arrays, feature):
    return condition_features(arrays["features"], feature + "_future")
