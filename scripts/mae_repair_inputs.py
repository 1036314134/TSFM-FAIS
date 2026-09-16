"""Use immutable R24/R25 inputs and prediction controls for the MAE-only check."""

import numpy as np
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import file_sha256


def evaluation_cases(smoke=False):
    source_root = ROOT / "artifacts/iclr27-r24/repair-evaluation-inputs-v001"
    source_predictions = ROOT / "artifacts/iclr27-r24/repair-evaluation-v001"
    source_map = {
        r["case_id"]: r
        for r in read_json(source_predictions / "predictions_frozen.json")["predictions"]
        if r["model_id"] == "chronos2"
    }
    native_root = ROOT / "artifacts/iclr27-r25/long-inputs-v001"
    native_predictions = ROOT / "artifacts/iclr27-r25/long-forecasts-v001"
    native_map = {r["case_id"]: r for r in read_json(native_predictions / "manifest.json")["cases"]}
    cases = []
    for row in read_json(source_root / "manifest.json")["cases"]:
        if row["panel"] != "source_validation":
            continue
        previous = source_map[row["case_id"]]
        cases.append(
            {
                **row,
                "input_path": str(source_root / row["path"]),
                "input_sha256": row["sha256"],
                "input_kind": "repair",
                "old_prediction_path": str(source_predictions / previous["path"]),
                "old_prediction_sha256": previous["sha256"],
            }
        )
    for row in read_json(native_root / "manifest.json")["cases"]:
        previous = native_map[row["case_id"]]
        cases.append(
            {
                **row,
                "case_id": "native_" + row["case_id"] + "_l192",
                "panel": "native_development",
                "context_length": 192,
                "origin_id": row["episode_id"],
                "input_path": str(native_root / row["path"]),
                "input_sha256": row["sha256"],
                "input_kind": "pool",
                "old_prediction_path": str(native_predictions / previous["path"]),
                "old_prediction_sha256": previous["sha256"],
            }
        )
    if len(cases) != 1165:
        raise ValueError("the fixed MAE-repair evaluation panel changed")
    if smoke:
        return [cases[0], next(r for r in cases if r["panel"] == "native_development")]
    return cases


def load_evaluation_case(row):
    from pathlib import Path

    if file_sha256(Path(row["input_path"])) != row["input_sha256"]:
        raise ValueError("a frozen MAE-repair input changed")
    with np.load(row["input_path"], allow_pickle=False) as saved:
        base = (
            saved["base"]
            if row["input_kind"] == "repair"
            else saved["candidate_values"][saved["candidate_ids"].tolist().index("seasonal_lag")]
        )
        return {
            "context": saved["context"],
            "base": base,
            "mean": saved["mean"],
            "scale": saved["scale"],
        }
