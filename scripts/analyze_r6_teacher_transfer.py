"""Audit complete-history teacher oracles and empirical real/teacher loss differences."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_fais.routing.forecast_gate import compose_forecasts  # noqa: E402
from tsfm_fais.routing.forecast_projection import (  # noqa: E402
    forecast_geometry,
    projection_targets,
    simplex_quadratic_weights,
)
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def loss_differences(point, reference, teacher, truth):
    real = ((point - truth) ** 2).mean((1, 2)) - ((reference - truth) ** 2).mean((1, 2))
    proxy = ((point - teacher) ** 2).mean((1, 2)) - ((reference - teacher) ** 2).mean((1, 2))
    cross = 2 * ((point - reference) * (teacher - truth)).mean((1, 2))
    return real, proxy, cross


def direct_optimality_gap(vectors, target, weights):
    prediction = compose_forecasts(vectors, weights)
    gradient = 2 * np.mean(vectors * (prediction - target)[:, None], axis=2)
    gap = (gradient * weights).sum(1) - gradient.min(1)
    relative = gap / np.maximum(1.0, abs(gradient).max(1))
    if relative.max() > 1e-7:
        raise ValueError("direct residual optimality failed")
    return float(relative.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "teacher-root",
        "cohort-root",
        "prepared-root",
        "forecast-root",
        "policy-root",
        "audit-root",
        "readout-root",
        "method-freeze",
        "fixed-oracle-root",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed teacher-transfer diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests/unit/test_r6_teacher_transfer.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        check=True,
        timeout=120,
    )
    prep = read_json(args.prepared_root / "manifest.json")
    cohort = read_json(args.cohort_root / "manifest.json")
    audit = read_json(args.audit_root / "manifest.json")
    binding = read_json(args.method_freeze)
    fixed_oracles = read_json(args.fixed_oracle_root / "manifest.json")
    if any(row["status"] != "completed" for row in (prep, cohort, audit, fixed_oracles)):
        raise ValueError("complete the preceding confirmation and oracle audits")
    if file_sha256(Path(binding["controls_path"])) != binding["controls_sha256"]:
        raise ValueError("the source controls changed")
    controls = read_json(Path(binding["controls_path"]))
    sources = {(row["dataset_id"], row["item_id"]): row for row in cohort["sources"]}
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in read_json(args.prepared_root / "standardizers.json")
    }
    if file_sha256(args.prepared_root / "standardizers.json") != prep["standardizers_sha256"]:
        raise ValueError("the original prefix scalers changed")
    records = [row for row in prep["episodes"] if row["panel"] == "new_synthetic"]
    if len(records) != 1152 or len({row["origin_id"] for row in records}) != 32:
        raise ValueError("synthetic coverage changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
        "confirmation_audit_sha256": file_sha256(args.audit_root / "manifest.json"),
        "fixed_oracle_sha256": file_sha256(args.fixed_oracle_root / "manifest.json"),
    }
    _write_json(output / "identity.json", identity)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    rows, oracle_rows, bank_records, teacher_manifests = [], [], [], []
    maximum_gap, maximum_identity_delta = 0.0, 0.0
    verified_sources = set()
    for model_id in ("chronos2", "timesfm2p5"):
        teacher_root, policy_root, forecast_root = (
            args.teacher_root / model_id,
            args.policy_root / model_id,
            args.forecast_root / model_id,
        )
        collected = read_json(teacher_root / "manifest.json")
        original = next(row for row in audit["models"] if row["model_id"] == model_id)
        if (
            collected["status"] != "completed"
            or collected["historical_prediction_requests"] != 64
            or collected["identity"]["protocol_sha256"] != identity["protocol_sha256"]
            or collected["identity"]["prepared_sha256"] != identity["prepared_sha256"]
            or file_sha256(policy_root / "manifest.json") != original["manifest_sha256"]
        ):
            raise ValueError("teacher or audited policy provenance changed")
        teacher_manifests.append(
            {"model_id": model_id, "sha256": file_sha256(teacher_root / "manifest.json")}
        )
        policy_manifest = read_json(policy_root / "manifest.json")
        if (
            collected["parameter_sha256"]
            != read_json(forecast_root / "manifest.json")["parameter_sha256"]
        ):
            raise ValueError("the teacher uses different forecaster parameters")
        actions = controls[model_id]["actions"]
        source_weights = np.asarray(controls[model_id]["convex_weights"])
        for horizon in (96, 192):
            teachers = {}
            for entry in [row for row in collected["teachers"] if row["horizon"] == horizon]:
                source = sources[(entry["dataset_id"], entry["item_id"])]
                source_path = Path(source["path"])
                if source_path not in verified_sources:
                    if file_sha256(source_path) != source["sha256"]:
                        raise ValueError("a teacher's original trajectory changed")
                    verified_sources.add(source_path)
                path = teacher_root / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("a collected teacher changed")
                with np.load(path, allow_pickle=False) as saved:
                    context = saved["clean_context"]
                    if (
                        not np.isfinite(context).all()
                        or str(saved["parameter_sha256"]) != collected["parameter_sha256"]
                    ):
                        raise ValueError("a complete teacher input or parameter record is invalid")
                    np.testing.assert_array_equal(
                        context,
                        np.load(source_path, mmap_mode="r")[entry["origin"] - 96 : entry["origin"]],
                    )
                    teachers[entry["origin_id"]] = saved["point_z"]
            if set(teachers) != {row["origin_id"] for row in records}:
                raise ValueError("complete teacher origin coverage changed")
            teacher = np.stack([teachers[row["origin_id"]] for row in records])
            policy_dir = policy_root / f"h{horizon}"
            policy_path = policy_dir / "policy_predictions.npz"
            marker = read_json(policy_dir / "predictions_frozen.json")
            horizon_record = next(
                row for row in policy_manifest["horizons"] if row["horizon"] == horizon
            )
            if (
                file_sha256(policy_dir / "predictions_frozen.json")
                != horizon_record["marker_sha256"]
            ):
                raise ValueError("the audited policy prediction marker changed")
            if file_sha256(policy_path) != marker["prediction_sha256"]:
                raise ValueError("frozen policy predictions changed")
            with np.load(policy_path, allow_pickle=False) as saved:
                ids = saved["episode_ids"].tolist()
                if ids != [row["episode_id"] for row in prep["episodes"]]:
                    raise ValueError("the audited policy order changed")
                positions = {identifier: index for index, identifier in enumerate(ids)}
                selected = [positions[row["episode_id"]] for row in records]
                names = saved["methods"].tolist()
                bank = saved["point_z"][selected]
            methods = {name: bank[:, index] for index, name in enumerate(names)}
            vectors_by_episode = []
            forecast_path = forecast_root / f"h{horizon}" / "manifest.json"
            forecast = read_json(forecast_path)
            candidates = {row["episode_id"]: row for row in forecast["predictions"]}
            for record in records:
                entry = candidates[record["episode_id"]]
                path = forecast_path.parent / entry["path"]
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError("an original candidate forecast changed")
                with np.load(path, allow_pickle=False) as saved:
                    order = saved["candidate_ids"].tolist()
                    vectors_by_episode.append(
                        saved["point_z"][[order.index(name) for name in actions]]
                    )
            points = np.stack(vectors_by_episode)
            for action_index, action in enumerate(actions):
                np.testing.assert_array_equal(points[:, action_index], methods[action])
            window_point, item_point = np.empty_like(teacher), np.empty_like(teacher)
            fixed_future_point = np.empty_like(teacher)
            for slot in [-1] if model_id == "chronos2" else [0, 1]:
                vectors = (
                    points.reshape(len(points), 7, -1) if slot == -1 else points[:, :, :, slot]
                )
                target = teacher.reshape(len(teacher), -1) if slot == -1 else teacher[:, :, slot]
                _, _, _, gram = forecast_geometry(vectors)
                alignment = projection_targets(vectors, target)["raw_projection"]
                weights, _, _ = simplex_quadratic_weights(gram, alignment)
                maximum_gap = max(maximum_gap, direct_optimality_gap(vectors, target, weights))
                predicted = compose_forecasts(vectors, weights)
                if slot == -1:
                    window_point[:] = predicted.reshape(len(points), horizon, 2)
                else:
                    window_point[:, :, slot] = predicted
                for dataset, item in sorted(
                    {(row["dataset_id"], row["item_id"]) for row in records}
                ):
                    indices = np.asarray(
                        [
                            i
                            for i, row in enumerate(records)
                            if (row["dataset_id"], row["item_id"]) == (dataset, item)
                        ]
                    )
                    optimum, certificate, _ = simplex_quadratic_weights(
                        gram[indices].mean(0)[None], alignment[indices].mean(0)[None]
                    )
                    local_weights = np.repeat(optimum, len(indices), axis=0)
                    local_prediction = compose_forecasts(vectors[indices], local_weights)
                    # Item-wise optimum can trade off windows; check its mean loss against the feasible source mixture.
                    source_prediction = compose_forecasts(
                        vectors[indices], np.repeat(source_weights[None], len(indices), axis=0)
                    )
                    if ((local_prediction - target[indices]) ** 2).mean() > (
                        (source_prediction - target[indices]) ** 2
                    ).mean() + 1e-9:
                        raise ValueError(
                            "item teacher optimum is worse than a feasible source mixture"
                        )
                    future_entry = next(
                        row
                        for row in fixed_oracles["oracles"]
                        if (
                            row["model_id"],
                            row["horizon"],
                            row["dataset_id"],
                            row["item_id"],
                            row["target_slot"],
                        )
                        == (model_id, horizon, dataset, item, slot)
                    )
                    future_prediction = compose_forecasts(
                        vectors[indices],
                        np.repeat(np.asarray(future_entry["weights"])[None], len(indices), axis=0),
                    )
                    if slot == -1:
                        item_point[indices] = local_prediction.reshape(len(indices), horizon, 2)
                        fixed_future_point[indices] = future_prediction.reshape(
                            len(indices), horizon, 2
                        )
                    else:
                        item_point[indices, :, slot] = local_prediction
                        fixed_future_point[indices, :, slot] = future_prediction
                    oracle_rows.append(
                        {
                            "model_id": model_id,
                            "horizon": horizon,
                            "dataset_id": dataset,
                            "item_id": item,
                            "target_slot": slot,
                            "teacher_item_weights": optimum[0].tolist(),
                            "optimality_gap": float(certificate[0]),
                            "complete_history_unavailable": True,
                        }
                    )
                _save_npz(
                    output / model_id / f"h{horizon}_slot{slot}_teacher_weights.npz",
                    weights=weights,
                    episode_ids=np.asarray([row["episode_id"] for row in records]),
                )
            methods.update(
                complete_history_teacher=teacher,
                unavailable_teacher_window_convex=window_point,
                unavailable_teacher_item_convex=item_point,
                unavailable_future_item_convex=fixed_future_point,
            )
            window_teacher_mse = ((window_point - teacher) ** 2).mean((1, 2))
            for name in {
                *actions,
                "ensemble_gate",
                "member_gate",
                "source_future_gate",
                "gate_source_fixed_convex",
                "gate_source_fixed_single",
            }:
                if (
                    window_teacher_mse - ((methods[name] - teacher) ** 2).mean((1, 2))
                ).max() > 1e-8:
                    raise ValueError("teacher optimum exceeds a known feasible convex policy")
            saved_path = output / model_id / f"h{horizon}_diagnostic_predictions.npz"
            _save_npz(
                saved_path,
                point_z=np.stack(list(methods.values()), axis=1),
                methods=np.asarray(list(methods)),
                episode_ids=np.asarray([row["episode_id"] for row in records]),
            )
            bank_records.append(
                {
                    "model_id": model_id,
                    "horizon": horizon,
                    "path": str(saved_path.relative_to(output)),
                    "sha256": file_sha256(saved_path),
                }
            )
            # No new weight is fitted below this point; evaluation labels only score fixed outputs.
            truth = []
            for record in records:
                path = args.prepared_root / record["path"]
                if file_sha256(path) != record["sha256"]:
                    raise ValueError("an original evaluation window changed")
                with np.load(path, allow_pickle=False) as saved:
                    future = saved["future"][:horizon]
                source = sources[(record["dataset_id"], record["item_id"])]
                origin = record["window"]["origin"]
                np.testing.assert_array_equal(
                    future, np.load(source["path"], mmap_mode="r")[origin : origin + horizon, :2]
                )
                if not np.isfinite(future).all():
                    raise ValueError("the synthetic diagnostic requires complete actual futures")
                scaler = scalers[(record["dataset_id"], record["item_id"])]
                truth.append(
                    (future - np.asarray(scaler["mean"])[:2]) / np.asarray(scaler["scale"])[:2]
                )
            truth = np.stack(truth)
            reference = methods["gate_source_fixed_convex"]
            for method, prediction in methods.items():
                real_delta, teacher_delta, cross = loss_differences(
                    prediction, reference, teacher, truth
                )
                difference = float(abs(real_delta - teacher_delta - cross).max())
                maximum_identity_delta = max(maximum_identity_delta, difference)
                np.testing.assert_allclose(
                    real_delta, teacher_delta + cross, rtol=1e-11, atol=1e-11
                )
                scores = {
                    "mae": abs(prediction - truth).mean((1, 2)),
                    "mse": ((prediction - truth) ** 2).mean((1, 2)),
                    "teacher_mae": abs(prediction - teacher).mean((1, 2)),
                    "teacher_mse": ((prediction - teacher) ** 2).mean((1, 2)),
                    "real_mse_delta": real_delta,
                    "teacher_mse_delta": teacher_delta,
                    "empirical_cross_term": cross,
                    "teacher_regret_vs_window_convex": ((prediction - teacher) ** 2).mean((1, 2))
                    - window_teacher_mse,
                }
                rows.extend(
                    {
                        **{
                            key: record[key]
                            for key in (
                                "episode_id",
                                "origin_id",
                                "family_id",
                                "dataset_id",
                                "item_id",
                            )
                        },
                        "model_id": model_id,
                        "horizon": horizon,
                        "method": method,
                        "convex_regret_is_bounded": method
                        in {
                            *actions,
                            "ensemble_gate",
                            "member_gate",
                            "source_future_gate",
                            "gate_source_fixed_convex",
                            "gate_source_fixed_single",
                            "unavailable_teacher_window_convex",
                            "unavailable_teacher_item_convex",
                            "unavailable_future_item_convex",
                        },
                        **{metric: float(values[index]) for metric, values in scores.items()},
                    }
                    for index, record in enumerate(records)
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "episode_metrics.parquet", index=False)
    keys = ["model_id", "horizon", "method", "family_id", "dataset_id", "item_id", "origin_id"]
    metrics = [
        "mae",
        "mse",
        "teacher_mae",
        "teacher_mse",
        "real_mse_delta",
        "teacher_mse_delta",
        "empirical_cross_term",
        "teacher_regret_vs_window_convex",
    ]
    origins = frame.groupby(keys)[metrics].mean()
    items = origins.groupby(keys[:-1])[metrics].mean()
    datasets = items.groupby(keys[:-2])[metrics].mean()
    families = datasets.groupby(keys[:-3])[metrics].mean()
    summary = families.groupby(keys[:3])[metrics].mean().reset_index()
    origins.to_csv(output / "origin_metrics.csv")
    families.to_csv(output / "family_metrics.csv")
    summary.to_csv(output / "summary.csv", index=False)
    original = pd.read_csv(args.readout_root / "summary.csv", float_precision="round_trip")
    original = original[original.panel == "new_synthetic_all"].set_index(keys[:3])
    actual = (
        summary[summary.method.isin(original.index.get_level_values("method"))]
        .set_index(keys[:3])
        .sort_index()
    )
    np.testing.assert_allclose(
        actual[["mae", "mse"]], original.loc[actual.index, ["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    previous_oracle = pd.read_csv(
        args.fixed_oracle_root / "summary.csv", float_precision="round_trip"
    )
    expected = previous_oracle[previous_oracle.method == "unavailable_item_fixed_oracle"].set_index(
        ["model_id", "horizon"]
    )
    current = summary[summary.method == "unavailable_future_item_convex"].set_index(
        ["model_id", "horizon"]
    )
    np.testing.assert_allclose(
        current[["mae", "mse"]], expected.loc[current.index, ["mae", "mse"]], rtol=1e-12, atol=1e-12
    )
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "teachers": teacher_manifests,
            "checked_complete_teacher_inputs": 128,
            "targeted_tests_passed": 3,
            "historical_origins": 32,
            "synthetic_inputs": 1152,
            "metric_rows": len(frame),
            "methods": frame.method.nunique(),
            "prediction_banks": bank_records,
            "item_teacher_oracles": oracle_rows,
            "maximum_direct_optimality_gap": maximum_gap,
            "maximum_loss_identity_difference": maximum_identity_delta,
            "original_metric_replay_max_difference": float(
                abs(
                    actual[["mae", "mse"]].to_numpy()
                    - original.loc[actual.index, ["mae", "mse"]].to_numpy()
                ).max()
            ),
            "summary_sha256": file_sha256(output / "summary.csv"),
            "limits": "post-confirmation explanatory reuse; teacher and future oracles unavailable; cross-term includes realized future variation and is not identified causal bias; median outputs need not lie in the candidate convex hull",
        },
    )


if __name__ == "__main__":
    main()
