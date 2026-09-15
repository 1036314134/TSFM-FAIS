"""Assess complete-input teacher targets without treating them as deployment inputs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.utility import family_macro  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accuracy-root", "chronos-repair-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, output = args.accuracy_root.resolve(), args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed clean-teacher diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    accuracy = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source_root = Path(accuracy["source_root"])
    source_path = source_root / "episodes_manifest.json"
    if file_sha256(source_path) != accuracy["source_episode_manifest_sha256"]:
        raise ValueError("source episodes changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    repair_path = args.chronos_repair_root / "manifest.json"
    if file_sha256(repair_path) != accuracy["chronos_repair_manifest_sha256"]:
        raise ValueError("the corrected Chronos reference changed")
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    repairs = {row["episode_id"]: row for row in repair["episodes"]}
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads((root / "standardizers.json").read_text(encoding="utf-8"))
    }
    targets = source["identity"]["config"]["target_indices"]
    n, horizon, dims = (
        len(source["episodes"]),
        source["identity"]["config"]["horizon"],
        len(targets),
    )
    identity = {
        "accuracy_manifest_sha256": file_sha256(root / "manifest.json"),
        "source_manifest_sha256": file_sha256(source_path),
        "repair_manifest_sha256": file_sha256(repair_path),
        "script_sha256": file_sha256(Path(__file__)),
        "teacher": "fixed forecaster applied to complete source history; not actual future labels",
        "evaluation_role": "unavailable complete-history information reference; never a deployable selection result",
    }
    _write_json(output / "identity.json", identity)
    truth_path = root / "truth_z.npy"
    if file_sha256(truth_path) != accuracy["prediction_arrays"][truth_path.name]:
        raise ValueError("standardized evaluation outcomes changed")
    rows, model_records = [], []
    for model in ("chronos2", "timesfm2p5"):
        forecast_manifest_path = source_root / model / "forecast_manifest.json"
        original = json.loads(forecast_manifest_path.read_text(encoding="utf-8"))
        hashes = {row["episode_id"]: row["sha256"] for row in original["episodes"]}
        clean = np.empty((n, horizon, dims), dtype=float)
        sources = []
        for index, record in enumerate(source["episodes"]):
            if model == "chronos2" and record["episode_id"] in repairs:
                item = repairs[record["episode_id"]]
                path, digest = args.chronos_repair_root / item["path"], item["sha256"]
            else:
                path = source_root / model / "predictions" / Path(record["path"]).name
                digest = hashes[record["episode_id"]]
                if model == "chronos2" and len(
                    scalers[(record["dataset_id"], record["item_id"])]["mean"]
                ) in (3, horizon):
                    raise ValueError(
                        "an affected complete-history forecast lacks a corrected source"
                    )
            if file_sha256(path) != digest:
                raise ValueError("a complete-history forecast changed")
            with np.load(path, allow_pickle=False) as saved:
                point = saved["clean_point"]
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            mean, scale = np.asarray(scaler["mean"])[targets], np.asarray(scaler["scale"])[targets]
            if point.shape != (horizon, dims) or not np.isfinite(point).all():
                raise ValueError("invalid complete-history prediction shape")
            clean[index] = (point - mean) / scale
            sources.append(
                {"episode_id": record["episode_id"], "path": str(path), "sha256": digest}
            )
        clean_path = output / f"{model}_clean_point_z.npy"
        np.save(clean_path, clean, allow_pickle=False)
        actions = [
            name
            for name in accuracy["action_orders"][model]
            if name not in {"native_missing", "vendor_missing"}
        ]
        positions = [accuracy["action_orders"][model].index(name) for name in actions]
        point_path = root / f"{model}_point_z.npy"
        if file_sha256(point_path) != accuracy["prediction_arrays"][point_path.name]:
            raise ValueError("candidate forecasts changed")
        points = np.load(point_path, mmap_mode="r")[:, positions]
        costs = ((points - clean[:, None]) ** 2).mean(axis=2)
        cost_path = output / f"{model}_clean_projection_costs.npy"
        np.save(cost_path, costs, allow_pickle=False)
        if model == "chronos2":
            choice = costs.mean(axis=2).argmin(axis=1)
            projected = points[np.arange(n), choice]
        else:
            choice = costs.argmin(axis=1)
            projected = np.stack(
                [points[np.arange(n), choice[:, slot], :, slot] for slot in range(dims)], axis=2
            )
        choice_path = output / f"{model}_unavailable_clean_projection_choices.npy"
        np.save(choice_path, choice, allow_pickle=False)
        # Both the complete teacher and its closest candidate use hidden history at evaluation.
        # Actual future outcomes enter here only to quantify their diagnostic accuracy.
        truth = np.load(truth_path, mmap_mode="r")
        forecasts = {
            "unavailable_complete_history": clean,
            "unavailable_clean_projection": projected,
            "forecast_median_guarded": np.median(points, axis=1),
        }
        for method, value in forecasts.items():
            for index, record in enumerate(source["episodes"]):
                if record["split"] != "validation":
                    continue
                error = value[index] - truth[index]
                rows.append(
                    {
                        "model_id": model,
                        "method": method,
                        "episode_id": record["episode_id"],
                        "family_id": record["family_id"],
                        "dataset_id": record["dataset_id"],
                        "mae": float(np.abs(error).mean()),
                        "mse": float((error**2).mean()),
                    }
                )
        _write_json(output / f"{model}_teacher_sources.json", sources)
        model_records.append(
            {
                "model_id": model,
                "teacher_file": clean_path.name,
                "teacher_sha256": file_sha256(clean_path),
                "cost_file": cost_path.name,
                "cost_sha256": file_sha256(cost_path),
                "choice_file": choice_path.name,
                "choice_sha256": file_sha256(choice_path),
                "actions": actions,
                "original_forecast_manifest_sha256": file_sha256(forecast_manifest_path),
                "source_list_sha256": file_sha256(output / f"{model}_teacher_sources.json"),
            }
        )
        print(
            json.dumps({"model": model, "status": "clean_teacher_exported", "source_records": n}),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_metrics.parquet", index=False)
    summary = pd.DataFrame(
        [
            {
                "model_id": model,
                "method": method,
                **{metric: family_macro(group, metric) for metric in ("mae", "mse")},
            }
            for (model, method), group in frame.groupby(["model_id", "method"])
        ]
    )
    prior = pd.read_csv(root / "analysis-joint-mae-mse-v001/summary.csv")
    for row in summary.itertuples():
        old_method = "clean" if row.method == "unavailable_complete_history" else row.method
        if old_method not in {"clean", "forecast_median_guarded"}:
            continue
        expected = prior[(prior.model_id == row.model_id) & (prior.method == old_method)]
        if len(expected) != 1:
            raise ValueError("the original accuracy reference is not unique")
        np.testing.assert_allclose(
            [row.mae, row.mse], expected[["mae", "mse"]].to_numpy()[0], rtol=0, atol=1e-9
        )
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": model_records,
            "summary": summary.to_dict("records"),
            "forecast_calls_added": 0,
            "limits": "complete teacher inputs and clean-projection selection are unavailable at deployment; all are information references, not achieved student results; no new student has been trained",
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
