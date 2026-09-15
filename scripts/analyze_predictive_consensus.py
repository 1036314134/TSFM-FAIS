"""Compare forecast medoids and convex consensus fits without outcome-based fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.recent_feedback import simplex_mse_weights  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def consensus_controls(predictions):
    """Fit only to the current candidate forecasts; no future observations are accepted."""
    point = np.asarray(predictions, float)
    if point.ndim != 3 or not np.isfinite(point).all():
        raise ValueError("finite [candidate,horizon,target] forecasts are required")
    teacher = np.median(point, axis=0)
    residual = point - teacher
    selected = int(np.argmin((residual**2).mean(axis=(1, 2))))
    per_target = np.argmin((residual**2).mean(axis=1), axis=0)
    weights, _ = simplex_mse_weights(residual.reshape(len(point), -1).T)
    target_weights = np.stack(
        [simplex_mse_weights(residual[:, :, slot].T)[0] for slot in range(point.shape[2])]
    )
    return {
        "median": (teacher, None),
        "medoid_sequence": (point[selected], np.eye(len(point))[selected]),
        "medoid_target": (
            np.column_stack([point[action, :, slot] for slot, action in enumerate(per_target)]),
            np.eye(len(point))[per_target],
        ),
        "simplex_sequence": (np.einsum("a,ahk->hk", weights, point), weights),
        "simplex_target": (np.einsum("ka,ahk->hk", target_weights, point), target_weights),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "accuracy-root", "controls-root", "motm-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed consensus controls")
    source_path = args.source_root / "episodes_manifest.json"
    accuracy_path = args.accuracy_root / "manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    motm_path = args.motm_root / "manifest.json"
    motm = json.loads(motm_path.read_text(encoding="utf-8"))
    accuracy_sha = file_sha256(accuracy_path)
    if (
        accuracy["source_episode_manifest_sha256"] != file_sha256(source_path)
        or motm["status"] != "completed"
        or motm["identity"]["accuracy_manifest_sha256"] != accuracy_sha
    ):
        raise ValueError("forecast sources do not share the same completed protocol")
    records = {
        record["episode_id"]: (index, record) for index, record in enumerate(source["episodes"])
    }
    truth = np.load(args.accuracy_root / "truth_z.npy", mmap_mode="r")
    rows, traces, oracles, provenance = [], [], [], {}
    for model in ("chronos2", "timesfm2p5"):
        controls = args.controls_root / model
        control_path = controls / "manifest.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        new_path = args.motm_root / model / "manifest.json"
        new = json.loads(new_path.read_text(encoding="utf-8"))
        if (
            control["status"] != "completed"
            or new["status"] != "completed"
            or control["identity"]["accuracy_manifest_sha256"] != accuracy_sha
            or not new["parameters_unchanged"]
        ):
            raise ValueError("a matched forecast stage is incomplete or inconsistent")
        additions = {record["episode_id"]: record for record in new["predictions"]}
        provenance[model] = {"control": file_sha256(control_path), "motm": file_sha256(new_path)}
        for record in control["episodes"]:
            index, metadata = records[record["episode_id"]]
            original_path = controls / record["path"]
            extra = additions[record["episode_id"]]
            extra_path = args.motm_root / extra["path"]
            if (
                file_sha256(original_path) != record["sha256"]
                or file_sha256(extra_path) != extra["sha256"]
            ):
                raise ValueError("candidate prediction arrays changed")
            with (
                np.load(original_path, allow_pickle=False) as saved,
                np.load(extra_path, allow_pickle=False) as added,
            ):
                names = saved["methods"].tolist()
                actions = source["identity"]["config"]["candidate_ids"] + ["guarded_direct"]
                positions = [names.index("prefix_input_z_" + action) for action in actions]
                base = saved["point_z"][positions]
                motm_forecast = added["point_z"][motm["identity"]["views"].index("motm_prefix_z")]
            for pool_name, pool in (
                ("base", base),
                ("with_motm", np.concatenate([base, motm_forecast[None]])),
            ):
                pool_actions = actions + (["motm_prefix_z"] if pool_name == "with_motm" else [])
                controls_fitted = consensus_controls(pool)
                # Outcomes enter only after all candidate choices and weights are fixed.
                outcome = truth[index]
                common = {
                    key: metadata[key]
                    for key in (
                        "episode_id",
                        "family_id",
                        "dataset_id",
                        "mechanism",
                        "missing_rate",
                    )
                }
                common.update(model_id=model, pool=pool_name)
                for method, (forecast, weights) in controls_fitted.items():
                    residual = forecast - outcome
                    rows.append(
                        common
                        | {
                            "method": method,
                            "mae": float(np.abs(residual).mean()),
                            "mse": float((residual**2).mean()),
                        }
                    )
                    traces.append(
                        common
                        | {
                            "method": method,
                            "actions": json.dumps(pool_actions),
                            "weights": json.dumps(
                                weights.tolist() if weights is not None else None
                            ),
                            "teacher_mse": float(
                                ((forecast - controls_fitted["median"][0]) ** 2).mean()
                            ),
                        }
                    )
                errors = pool - outcome
                oracle = int(np.argmin((errors**2).mean(axis=(1, 2))))
                oracles.append(
                    common
                    | {
                        "method": "future_mse_oracle",
                        "mae": float(np.abs(errors[oracle]).mean()),
                        "mse": float((errors[oracle] ** 2).mean()),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    keys = ["model_id", "pool", "method"]
    expected = motm["decision_count"]
    for _, group in data.groupby(keys):
        if len(group) != expected or group.episode_id.duplicated().any():
            raise ValueError("consensus controls have inconsistent task coverage")
    family = (
        data.groupby(keys + ["family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse"]].mean().reset_index()
    data.to_parquet(output / "episode_results.parquet", index=False)
    pd.DataFrame(traces).to_parquet(output / "decision_traces.parquet", index=False)
    pd.DataFrame(oracles).to_parquet(output / "future_oracle_diagnostic.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development_controls",
            "decision_count": expected,
            "script_sha256": file_sha256(Path(__file__)),
            "accuracy_manifest_sha256": accuracy_sha,
            "motm_manifest_sha256": file_sha256(motm_path),
            "forecast_manifests": provenance,
            "information": "median, medoid and simplex rules use current candidate forecasts only; no forecast outcomes or historical feedback are used in fitting",
            "interpretation": "forecast-space aggregation controls, not actual imputed-input combinations; future-label oracle is a separate diagnostic",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
