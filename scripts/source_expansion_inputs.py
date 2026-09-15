"""Load the fixed source validation bank and its training-only supplement."""

import json

import numpy as np
import pandas as pd
from aligned_portfolio_io import decision_truth
from latent_source_inputs import load_source_inputs, read_json

from tsfm_fais.utility_experiment import file_sha256


def load_expanded_inputs(base_root, supplement_root, accuracy_root, model_id):
    _, frame, arrays = load_source_inputs(base_root, model_id)
    arrays["features"] = arrays["features"].copy()
    arrays["features"][:, :, 33:] = 0
    train = np.flatnonzero(frame.split.to_numpy() == "train")
    arrays["truth"] = np.full_like(arrays["teacher"], np.nan)
    arrays["truth"][train] = decision_truth(
        frame.iloc[train], np.load(accuracy_root / "truth_z.npy", mmap_mode="r")
    )
    frame = frame.assign(source_population="original")
    manifest = read_json(supplement_root / "manifest.json")
    if manifest["status"] != "completed" or manifest["identity"]["base_sha256"] != file_sha256(
        base_root / "manifest.json"
    ):
        raise ValueError("the completed supplement and original source do not match")
    entry = next(row for row in manifest["models"] if row["model_id"] == model_id)
    if file_sha256(supplement_root / entry["path"]) != entry["sha256"]:
        raise ValueError("a supplemental predictor manifest changed")
    model = read_json(supplement_root / entry["path"])
    if model["identity_sha256"] != manifest["identity_sha256"] or len(model["episodes"]) != 2826:
        raise ValueError("supplemental predictor coverage changed")
    extra_frames = []
    extra_arrays = {name: [] for name in arrays}
    for row in model["episodes"]:
        path = supplement_root / model_id / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("a supplemental decision changed")
        with np.load(path, allow_pickle=False) as saved:
            extra_frames.append(pd.DataFrame(json.loads(str(saved["decisions"]))))
            for name in extra_arrays:
                values = saved[name]
                if name == "features":
                    values = np.pad(values, ((0, 0), (0, 0), (0, 64)))
                extra_arrays[name].append(values)
    extra_frame = pd.concat(extra_frames, ignore_index=True).assign(source_population="supplement")
    if set(extra_frame.origin_id) & set(frame.origin_id) or set(extra_frame.split) != {"train"}:
        raise ValueError("the supplement overlaps an original history or contains validation")
    combined = pd.concat([frame, extra_frame], ignore_index=True)
    values = {
        name: np.ascontiguousarray(np.concatenate([arrays[name], *extra_arrays[name]]))
        for name in arrays
    }
    if (
        combined.episode_id.duplicated().any()
        or combined.origin_id.nunique() != 374
        or combined[combined.split == "train"].origin_id.nunique() != 322
        or combined[combined.split == "validation"].origin_id.nunique() != 52
    ):
        raise ValueError("the 322-training/52-validation population changed")
    return combined, values
