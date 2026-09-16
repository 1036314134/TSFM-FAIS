"""Freeze all fixed provenance-pilot methods before current observed-target scoring."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from evaluate_matched_replay import aggregate_groups, score_points
from latent_source_inputs import ROOT, read_json

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def methods(bank, actions, control):
    if actions != control["actions"]:
        raise ValueError("source action identities changed")
    result = {}
    for prefix, points in zip(("", "provenance_", "shifted_"), bank, strict=True):
        result.update({prefix + action: points[i] for i, action in enumerate(actions)})
        result[prefix + "median8"] = np.median(points, axis=0)
        result[prefix + "mean8"] = points.mean(0)
        result[prefix + "source_single_mae"] = points[control["single_index"]]
        result[prefix + "source_fixed_mae"] = (
            points * np.asarray(control["fixed_mae"]["weights"])[:, None, None]
        ).sum(0)
        result[prefix + "source_fixed_joint"] = (
            points * np.asarray(control["fixed_joint_weights"])[:, None, None]
        ).sum(0)
    if len(result) != 39:
        raise ValueError("the registered method population changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    base = ROOT / "artifacts/iclr27-r21"
    input_root, forecast_root = base / "provenance-inputs-v001", base / "provenance-forecasts-v001"
    inputs, forecasts = [read_json(p / "manifest.json") for p in (input_root, forecast_root)]
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed provenance results")
    if (
        inputs["status"] != "completed"
        or forecasts["status"] != "completed"
        or len(forecasts["cases"]) != 46
    ):
        raise ValueError("finish all 46 input-model forecasts before scoring")
    defaults_path = ROOT / "artifacts/iclr27-r19/replay-results-v001/source_defaults.json"
    identity = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in (
            Path(__file__),
            input_root / "manifest.json",
            forecast_root / "manifest.json",
            defaults_path,
            ROOT / "docs/iclr2027/R21_PROVENANCE_PILOT_PROTOCOL.md",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial provenance readout definitions changed")
    _write_json(output / "identity.json", identity)
    defaults = read_json(defaults_path)
    records = []
    for entry in forecasts["cases"]:
        path = forecast_root / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("a frozen forecast bank changed")
        with np.load(path, allow_pickle=False) as saved:
            points = methods(
                saved["bank"], saved["actions"].tolist(), defaults["models"][entry["model_id"]]
            )
        path = output / "predictions" / entry["model_id"] / f"{entry['case_id']}.npz"
        _save_npz(path, methods=np.asarray(list(points)), points=np.stack(list(points.values())))
        records.append(
            {
                "model_id": entry["model_id"],
                "case_id": entry["case_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
    _write_json(
        output / "predictions_frozen.json",
        {
            "identity": identity,
            "predictions": records,
            "current_future_read": False,
            "source_defaults_sha256": file_sha256(defaults_path),
        },
    )
    metadata = {r["case_id"]: r for r in inputs["cases"]}
    scores, targets = [], []
    for record in records:
        row = metadata[record["case_id"]]
        if (
            file_sha256(Path(row["original_path"])) != row["original_sha256"]
            or file_sha256(input_root / row["path"]) != row["sha256"]
        ):
            raise ValueError("an original or prepared current case changed")
        with (
            np.load(row["original_path"], allow_pickle=False) as original,
            np.load(input_root / row["path"], allow_pickle=False) as data,
        ):
            truth = original["future"][:96]
            np.testing.assert_array_equal(np.isfinite(truth), original["future_observed"][:96])
            truth_z = (truth - data["mean"][:2]) / data["scale"][:2]
        with np.load(output / record["path"], allow_pickle=False) as saved:
            mae, mse = score_points(saved["points"], truth_z)
            for index, method in enumerate(saved["methods"].tolist()):
                entry = {
                    name: row[name]
                    for name in (
                        "case_id",
                        "group_id",
                        "family_id",
                        "dataset_id",
                        "item_id",
                        "origin",
                        "episode_id",
                    )
                }
                entry.update(
                    model_id=record["model_id"],
                    method=method,
                    mae=float(mae[index].mean()),
                    mse=float(mse[index].mean()),
                )
                scores.append(entry)
                for slot in (0, 1):
                    targets.append(
                        {
                            **entry,
                            "target_slot": slot,
                            "observed_count": int(np.isfinite(truth[:, slot]).sum()),
                            "mae": float(mae[index, slot]),
                            "mse": float(mse[index, slot]),
                        }
                    )
    frame = pd.DataFrame(scores)
    if len(frame) != 1794 or len(targets) != 3588:
        raise ValueError("the frozen scoring population changed")
    frame.to_parquet(output / "case_scores.parquet", index=False)
    pd.DataFrame(targets).to_parquet(output / "target_scores.parquet", index=False)
    for name, table in zip(
        ("series", "datasets", "groups", "summary"), aggregate_groups(frame), strict=True
    ):
        table.to_csv(output / f"{name}.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "predictions": records,
            "score_rows": len(frame),
            "target_score_rows": len(targets),
            "primary": "provenance_median8",
            "primary_metric": "mae",
            "independent_confirmation": False,
        },
    )


if __name__ == "__main__":
    main()
