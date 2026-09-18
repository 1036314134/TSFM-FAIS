"""Shared immutable sources and prefix-only regression for the peer outage screen."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset  # noqa: F401
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import file_sha256

BASE = ROOT / "artifacts/iclr27-r30"
COHORT = ROOT / "artifacts/iclr27-r6/cohort-v001/manifest.json"
PEERS = ROOT / "artifacts/iclr27-r27/peer-information-preflight-v001/manifest.json"
STAT_METHODS = ("local_ridge", "peer_ridge", "peer_residual_linear", "peer_ar_bridge")


def sources():
    records = [r for r in read_json(COHORT)["sources"] if r["dataset_id"] == "beijing_multisite"]
    if (
        len(records) != 12
        or len({(r["start"], r["frequency"], tuple(r["columns"])) for r in records}) != 1
    ):
        raise ValueError("peer station alignment or original variables changed")
    result = {}
    for record in records:
        if file_sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("an immutable station source changed")
        result[record["item_id"]] = {**record, "values": np.load(record["path"], mmap_mode="r")}
    peer_map = {}
    for row in read_json(PEERS)["prefix_peer_map"]:
        peer_map[(row["target_item"], row["target_slot"])] = [
            x["item_id"] for x in row["selected_using_prefix_only"]
        ]
    return result, peer_map


def augmented(station, records, peer_map):
    own = records[station]
    columns = [own["values"]]
    names = []
    for slot in (0, 1):
        for peer in peer_map[(station, slot)]:
            if peer == station or records[peer]["prefix_end"] != own["prefix_end"]:
                raise ValueError("invalid peer or unequal registered prefixes")
            columns.append(records[peer]["values"][:, slot : slot + 1])
            names.append({"station": peer, "slot": slot})
    return np.concatenate(columns, axis=1), names


def population(records):
    cases, rejected = [], []
    for station, source in sorted(records.items()):
        obs = np.isfinite(source["values"][:, :2])
        n, prefix = len(obs), source["prefix_end"]
        start = max(int(0.6 * n) + 192, prefix + 192)
        age, last = 0, -100000
        for i, absent in enumerate((~obs).all(1)):
            age = age + 1 if absent else 0
            t = i + 1
            if t < start or t + 24 > n or age < 6 or (age - 6) % 24:
                continue
            if (obs[t : t + 24].sum(0) < 12).any():
                rejected.append(
                    {
                        "station": station,
                        "origin": t,
                        "reason": "future_observed_support",
                        "age": age,
                    }
                )
                continue
            if t - last < 24:
                rejected.append(
                    {"station": station, "origin": t, "reason": "overlapping_forecast", "age": age}
                )
                continue
            cases.append(
                {
                    "panel": "natural_outage_h24",
                    "station": station,
                    "origin": t,
                    "horizon": 24,
                    "outage_age": age,
                }
            )
            last = t
        complete = [t for t in range(start, n - 24 + 1, 24) if obs[t - 192 : t + 24].all()]
        if len(complete) < 4:
            raise ValueError("insufficient complete synthetic anchors")
        for index in np.linspace(0, len(complete) - 1, 4, dtype=int):
            cases.append(
                {
                    "panel": "synthetic_outage_h24",
                    "station": station,
                    "origin": complete[index],
                    "horizon": 24,
                    "outage_age": 24,
                }
            )
    legacy = read_json(ROOT / "artifacts/iclr27-r25/long-plan-v001/manifest.json")
    for row in legacy["cases"]:
        if row["group_id"] == "beijing_multisite":
            cases.append(
                {
                    "panel": "legacy_native_h96",
                    "station": row["item_id"],
                    "origin": row["origin"],
                    "horizon": 96,
                    "legacy_case_id": row["case_id"],
                }
            )
    counts = {p: sum(r["panel"] == p for r in cases) for p in {r["panel"] for r in cases}}
    if counts != {"natural_outage_h24": 45, "synthetic_outage_h24": 48, "legacy_native_h96": 138}:
        raise ValueError(f"registered input-only population changed: {counts}")
    for row in cases:
        key = f"{row['panel']}|{row['station']}|{row['origin']}|{row['horizon']}"
        row["case_id"] = hashlib.sha256(key.encode()).hexdigest()[:20]
        row["prefix_end"] = records[row["station"]]["prefix_end"]
    return cases, rejected


def case_context(full, row):
    t = row["origin"]
    x = np.array(full[t - 192 : t], dtype=float, copy=True)
    if row["panel"] == "synthetic_outage_h24":
        x[-24:, :2] = np.nan
    return x


def bridge_residual(phi, position, indices, residuals, linear=False):
    if len(indices) == 0:
        return 0.0
    insert = np.searchsorted(indices, position)
    left = insert - 1 if insert else None
    right = insert if insert < len(indices) else None
    if left is None:
        return float(
            residuals[right] if linear else phi ** (indices[right] - position) * residuals[right]
        )
    if right is None:
        return float(
            residuals[left] if linear else phi ** (position - indices[left]) * residuals[left]
        )
    dl, dr = position - indices[left], indices[right] - position
    if linear:
        return float((dr * residuals[left] + dl * residuals[right]) / (dl + dr))
    a, b = phi**dl, phi**dr
    return float(
        (a * (1 - b * b) * residuals[left] + b * (1 - a * a) * residuals[right])
        / (1 - a * a * b * b)
    )


class PrefixRegression:
    def __init__(self, prefix):
        self.prefix = np.array(prefix, float, copy=True)
        self.mean = np.nanmean(self.prefix, axis=0)
        self.scale = np.nanstd(self.prefix, axis=0, ddof=0)
        self.scale = np.where(self.scale <= 1e-12, 1.0, self.scale)
        self.z = (self.prefix - self.mean) / self.scale
        self.cache = {}
        self.corr = {}
        for target in (0, 1):
            for feature in range(self.prefix.shape[1]):
                valid = np.isfinite(self.z[:, target]) & np.isfinite(self.z[:, feature])
                correlation = (
                    np.corrcoef(self.z[valid, target], self.z[valid, feature])[0, 1]
                    if valid.sum() >= 128 and np.std(self.z[valid, feature]) > 1e-12
                    else 0.0
                )
                self.corr[target, feature] = (
                    float(abs(correlation)) if np.isfinite(correlation) else 0.0
                )

    def fit(self, target, features):
        features = sorted(int(i) for i in features if i != target)
        request = f"{target}:" + ",".join(map(str, features))
        if request in self.cache:
            return self.cache[request]
        while features:
            valid = np.isfinite(self.z[:, target]) & np.isfinite(self.z[:, features]).all(1)
            if valid.sum() >= 128:
                break
            features.remove(min(features, key=lambda i: (self.corr[target, i], -i)))
        if not features:
            result = {"target": target, "features": [], "support": 0, "beta": [], "phi": 0.0}
        else:
            design = np.column_stack([np.ones(valid.sum()), self.z[valid][:, features]])
            y = self.z[valid, target]
            penalty = np.eye(len(features) + 1) * 0.001
            penalty[0, 0] = 0
            beta = np.linalg.solve(design.T @ design / len(y) + penalty, design.T @ y / len(y))
            residual = np.full(len(valid), np.nan)
            residual[valid] = y - design @ beta
            consecutive = np.isfinite(residual[1:]) & np.isfinite(residual[:-1])
            previous, following = residual[:-1][consecutive], residual[1:][consecutive]
            phi = float(np.clip(previous @ following / max(previous @ previous, 1e-12), 0, 0.99))
            result = {
                "target": target,
                "features": features,
                "support": int(valid.sum()),
                "beta": beta.tolist(),
                "phi": phi,
            }
        self.cache[request] = result
        return result

    def targets(self, context, fallback):
        observed = np.isfinite(context)
        z = (context - self.mean) / self.scale
        result = {name: context[:, :2].copy() for name in STAT_METHODS}
        for target in (0, 1):
            for t in np.flatnonzero(~observed[:, target]):
                for local in (True, False):
                    available = np.flatnonzero(observed[t, :11] if local else observed[t])
                    model = self.fit(target, available)
                    features = model["features"]
                    names = ("local_ridge",) if local else STAT_METHODS[1:]
                    if not features:
                        for name in names:
                            result[name][t, target] = fallback[t, target]
                        continue
                    beta = np.asarray(model["beta"])
                    g = beta[0] + z[t, features] @ beta[1:]
                    plain = g * self.scale[target] + self.mean[target]
                    result[names[0]][t, target] = plain
                    if local:
                        continue
                    valid = observed[:, target] & observed[:, features].all(1)
                    indices = np.flatnonzero(valid)
                    residuals = z[indices, target] - (beta[0] + z[indices][:, features] @ beta[1:])
                    for name, linear in (("peer_residual_linear", True), ("peer_ar_bridge", False)):
                        change = bridge_residual(model["phi"], t, indices, residuals, linear)
                        result[name][t, target] = (g + change) * self.scale[target] + self.mean[
                            target
                        ]
        for values in result.values():
            np.testing.assert_array_equal(values[observed[:, :2]], context[:, :2][observed[:, :2]])
            if not np.isfinite(values).all():
                raise ValueError("statistical target repair did not finish")
        return result

    def future_models(self, keep):
        return [self.fit(target, np.flatnonzero(keep)) for target in (0, 1)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "models": self.cache},
                indent=2,
            ),
            encoding="utf-8",
        )
