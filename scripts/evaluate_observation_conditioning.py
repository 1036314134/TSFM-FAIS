"""Use current observed history for fixed updates, freeze predictions, then score future."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
from evaluate_matched_replay import aggregate_groups, score_points
from latent_source_inputs import ROOT, read_json
from observation_conditioning import conditional_forecasts

from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    base = ROOT / "artifacts/iclr27-r23"
    inputs_root, forecasts_root = (
        base / "conditioning-inputs-v001",
        base / "conditioning-forecasts-v001",
    )
    inputs, forecasts = [read_json(p / "manifest.json") for p in (inputs_root, forecasts_root)]
    if (
        inputs["status"] != "completed"
        or forecasts["status"] != "completed"
        or len(forecasts["cases"]) != 46
    ):
        raise ValueError("all registered priors and direct controls must complete first")
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed conditioning results")
    old_root = ROOT / "artifacts/iclr27-r21/provenance-results-v001"
    old = read_json(old_root / "manifest.json")
    old_audit = read_json(ROOT / "artifacts/iclr27-r21/provenance-audit-v001/manifest.json")
    if old_audit["status"] != "completed" or old_audit["study_sha256"] != file_sha256(
        old_root / "manifest.json"
    ):
        raise ValueError("the R21 control audit binding changed")
    identity = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in (
            Path(__file__),
            ROOT / "scripts/observation_conditioning.py",
            inputs_root / "manifest.json",
            forecasts_root / "manifest.json",
            old_root / "manifest.json",
            ROOT / "docs/iclr2027/R23_OBSERVATION_CONDITIONING_PROTOCOL.md",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial conditioning readout definitions changed")
    _write_json(output / "identity.json", identity)
    metadata = {r["case_id"]: r for r in inputs["cases"]}
    controls = {(r["model_id"], r["case_id"]): r for r in old["predictions"]}
    records = []
    for entry in forecasts["cases"]:
        row, old_record = (
            metadata[entry["case_id"]],
            controls[(entry["model_id"], entry["case_id"])],
        )
        for path, digest in (
            (inputs_root / row["path"], row["sha256"]),
            (forecasts_root / entry["path"], entry["sha256"]),
            (old_root / old_record["path"], old_record["sha256"]),
        ):
            if file_sha256(path) != digest:
                raise ValueError("a frozen input, prior or control changed")
        with (
            np.load(inputs_root / row["path"], allow_pickle=False) as data,
            np.load(forecasts_root / entry["path"], allow_pickle=False) as prior,
            np.load(old_root / old_record["path"], allow_pickle=False) as old_points,
        ):
            context = (data["current_context"][:, :2] - data["mean"][:2]) / data["scale"][:2]
            methods, diagnostics = conditional_forecasts(
                prior["prior"], prior["prior_quantiles"], context
            )
            methods.update(native192=prior["native192"], budget_native192=prior["budget_native192"])
            for name, point in zip(
                old_points["methods"].tolist(), old_points["points"], strict=True
            ):
                if not name.startswith(("provenance_", "shifted_")):
                    methods[name] = point
            if len(methods) != 20:
                raise ValueError("the prespecified method population changed")
            path = output / "predictions" / entry["model_id"] / f"{entry['case_id']}.npz"
            _save_npz(
                path, methods=np.asarray(list(methods)), points=np.stack(list(methods.values()))
            )
            records.append(
                {
                    "model_id": entry["model_id"],
                    "case_id": entry["case_id"],
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "conditioning": diagnostics,
                }
            )
    _write_json(
        output / "predictions_frozen.json",
        {
            "identity": identity,
            "predictions": records,
            "current_history_observations_used": True,
            "current_future_read": False,
        },
    )
    scores, targets = [], []
    for entry in records:
        row = metadata[entry["case_id"]]
        if file_sha256(Path(row["original_path"])) != row["original_sha256"]:
            raise ValueError("the current future source changed")
        with (
            np.load(row["original_path"], allow_pickle=False) as original,
            np.load(inputs_root / row["path"], allow_pickle=False) as data,
            np.load(output / entry["path"], allow_pickle=False) as points,
        ):
            truth = original["future"][:96]
            np.testing.assert_array_equal(np.isfinite(truth), original["future_observed"][:96])
            normalized = (truth - data["mean"][:2]) / data["scale"][:2]
            mae, mse = score_points(points["points"], normalized)
            for index, method in enumerate(points["methods"].tolist()):
                row_data = {
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
                row_data.update(
                    model_id=entry["model_id"],
                    method=method,
                    mae=float(mae[index].mean()),
                    mse=float(mse[index].mean()),
                )
                scores.append(row_data)
                for slot in (0, 1):
                    targets.append(
                        {
                            **row_data,
                            "target_slot": slot,
                            "observed_count": int(np.isfinite(truth[:, slot]).sum()),
                            "mae": float(mae[index, slot]),
                            "mse": float(mse[index, slot]),
                        }
                    )
    frame = pd.DataFrame(scores)
    if len(frame) != 920 or len(targets) != 1840:
        raise ValueError("the registered scoring population changed")
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
            "primary": "unit_conditioner",
            "primary_metric": "mae",
            "independent_confirmation": False,
        },
    )


if __name__ == "__main__":
    main()
