"""Shared input loading for the aligned portfolio study and its replay audit."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.aligned_portfolio import option_catalog, option_vectors  # noqa: E402
from tsfm_fais.utility_experiment import file_sha256  # noqa: E402


def load_prepared_model(root, preparation, model):
    entry = next(row for row in preparation["models"] if row["model_id"] == model)
    path = root / entry["path"]
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("prepared model metadata changed")
    info = json.loads(path.read_text(encoding="utf-8"))
    if info["status"] != "completed" or info["identity_sha256"] != preparation["identity_sha256"]:
        raise ValueError("prepared model identity changed")
    path = root / info["decisions_path"]
    if file_sha256(path) != info["decisions_sha256"]:
        raise ValueError("prepared decisions changed")
    decisions = pd.read_parquet(path)
    count = info["decision_count"]
    if len(decisions) != count or decisions.episode_id.duplicated().any():
        raise ValueError("decision identities are incomplete or duplicated")
    arrays = {"features": np.empty((count, 43, 40), np.float32)}
    arrays.update(
        {name: np.empty((count, 43)) for name in ("norm", "unit_projection", "direct_risk")}
    )
    covered = np.zeros(count, bool)
    for chunk in info["chunks"]:
        path = root / chunk["path"]
        start, stop = chunk["start"], chunk["stop"]
        if (
            not 0 <= start < stop <= count
            or covered[start:stop].any()
            or file_sha256(path) != chunk["sha256"]
        ):
            raise ValueError("a prepared chunk changed or overlaps another chunk")
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["identity_sha256"]) != preparation["identity_sha256"]:
                raise ValueError("a prepared chunk has another identity")
            for name in arrays:
                arrays[name][start:stop] = saved[name]
        covered[start:stop] = True
    if not covered.all() or not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("prepared inputs or labels are incomplete")
    return info, decisions, arrays


def option_rows(decisions, features, info, indices):
    indices = np.asarray(indices, int)
    values = pd.DataFrame(features[indices].reshape(-1, 40), columns=info["feature_names"])
    metadata = decisions.iloc[np.repeat(indices, 43)].reset_index(drop=True)
    metadata["candidate_id"] = np.tile(info["option_names"], len(indices))
    return pd.concat([metadata, values], axis=1)


def decision_vectors(decisions, bank):
    return np.stack(
        [
            bank[row.episode_index].reshape(bank.shape[1], -1)
            if row.target_slot == -1
            else bank[row.episode_index, :, :, row.target_slot]
            for row in decisions.itertuples(index=False)
        ]
    )


def current_options(decisions, bank, actions):
    _, members = option_catalog(actions)
    return option_vectors(decision_vectors(decisions, bank), members)


def decision_truth(decisions, truth):
    return np.stack(
        [
            truth[row.episode_index].reshape(-1)
            if row.target_slot == -1
            else truth[row.episode_index, :, row.target_slot]
            for row in decisions.itertuples(index=False)
        ]
    )
