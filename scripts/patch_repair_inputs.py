"""Read source-only supervised inputs and freeze a dimensionality-limited training population."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from latent_source_inputs import ROOT, read_json

from tsfm_fais.routing.utility import _family_weights
from tsfm_fais.utility_experiment import file_sha256

SOURCE = ROOT / "artifacts/iclr27-r3/development-expanded-v001"
ACCURACY = ROOT / "artifacts/iclr27-r4/accuracy-development-v002"


def source_population():
    source = read_json(SOURCE / "episodes_manifest.json")
    scalers = {
        (r["dataset_id"], r["item_id"]): r for r in read_json(ACCURACY / "standardizers.json")
    }
    kept, excluded = [], []
    for index, record in enumerate(source["episodes"]):
        if record["mask_seed"] != 6101:
            continue
        scaler = scalers[(record["dataset_id"], record["item_id"])]
        row = {
            **record,
            "episode_index": index,
            "dimensions": len(scaler["mean"]),
            "scaler": scaler,
        }
        (kept if row["dimensions"] <= 64 else excluded).append(row)
    return kept, excluded


def load_case(row):
    path = SOURCE / row["path"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError("a source training case changed")
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["candidate_ids"].tolist()
        raw, context, truth = (
            saved["candidate_values"][ids.index("seasonal_lag")],
            saved["context"],
            saved["future"][:, :2],
        )
    observed = np.isfinite(context)
    np.testing.assert_array_equal(raw[observed], context[observed])
    if not np.isfinite(raw).all() or not np.isfinite(truth).all():
        raise ValueError(
            "source learning requires finite inputs and originally observed future labels"
        )
    return raw, observed, truth


def training_weights(rows):
    weights = _family_weights(pd.DataFrame(rows))
    return weights * len(rows) / weights.sum()


def source_identity():
    return {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in (
            Path(__file__),
            SOURCE / "episodes_manifest.json",
            ACCURACY / "standardizers.json",
            ACCURACY / "truth_z.npy",
        )
    }
