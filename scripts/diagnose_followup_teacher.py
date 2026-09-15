"""Separate source-selector transfer from its unavailable clean-history teacher target."""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_triple_objective_gap import best_option, teacher_top_three  # noqa: E402
from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from probe_differentiable_imputation import parameter_digest  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cohort-root",
        "prepared-root",
        "forecast-root",
        "policy-root",
        "policy-audit-root",
        "old-bundle",
        "protocol",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed post-evaluation diagnostics")
    prep = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    cohort = json.loads((args.cohort_root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((args.policy_audit_root / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads((args.old_bundle / "manifest.json").read_text(encoding="utf-8"))
    if audit["status"] != "completed" or prep["identity"]["cohort_sha256"] != file_sha256(
        args.cohort_root / "manifest.json"
    ):
        raise ValueError("complete the registered primary evaluation audit first")
    indices = [i for i, row in enumerate(prep["episodes"]) if row["panel"] == "new_synthetic"]
    records = [prep["episodes"][i] for i in indices]
    origins = list(dict.fromkeys(row["origin_id"] for row in records))
    if len(records) != 576 or len(origins) != 16:
        raise ValueError("the explanatory panel changed")
    source_map = {(row["dataset_id"], row["item_id"]): row for row in cohort["sources"]}
    scaler_path = args.prepared_root / "standardizers.json"
    if file_sha256(scaler_path) != prep["standardizers_sha256"]:
        raise ValueError("prefix scoring statistics changed")
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "protocol_sha256": file_sha256(args.protocol),
        "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
        "prepared_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "audit_sha256": file_sha256(args.policy_audit_root / "manifest.json"),
        "oracle_helpers_sha256": file_sha256(ROOT / "scripts/analyze_triple_objective_gap.py"),
        "timing_of_design": "post-evaluation diagnostic; not independent confirmation",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial diagnostic changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    all_rows, model_records = [], []
    for model in ("chronos2", "timesfm2p5"):
        forecast_dir, policy_dir = args.forecast_root / model, args.policy_root / model
        forecast = json.loads((forecast_dir / "manifest.json").read_text(encoding="utf-8"))
        entry = next(row for row in audit["models"] if row["model_id"] == model)
        if file_sha256(policy_dir / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError("primary results changed after audit")
        registry = default_forecast_registry()
        joint = model == "chronos2"
        model_path = old["identity"]["forecaster_artifacts"][model]
        adapter = (
            registry.build(model, model_name=model_path, device="cuda", batch_size=8)
            if joint
            else TimesFMVendorMissingAdapter(model_name=model_path, device="cuda", batch_size=8)
        )
        runner = ForecastRunner(registry, {model: adapter})
        backbone = adapter._ensure_backend().model.eval().requires_grad_(False)
        if parameter_digest(backbone) != forecast["parameter_sha256"]:
            raise ValueError("the diagnostic forecaster changed")
        teacher_path = output / f"{model}_clean_histories.npz"
        if not teacher_path.exists():
            teachers = []
            for origin in origins:
                record = next(row for row in records if row["origin_id"] == origin)
                key = (record["dataset_id"], record["item_id"])
                source = source_map[key]
                path = args.cohort_root / source["path"]
                if file_sha256(path) != source["sha256"]:
                    raise ValueError("the original clean trajectory changed")
                values = np.load(path, mmap_mode="r")
                start = record["window"]["origin"]
                context = values[start - 96 : start]
                if not np.isfinite(context).all():
                    raise ValueError(
                        "the unavailable clean-history diagnostic requires actual complete source history"
                    )
                spec = ForecastSpec(
                    model, registry.get(model).mode, 96, context_length=96, target_indices=[0, 1]
                )
                point = runner.predict(context[None], spec).point[0]
                scaler = scalers[key]
                teachers.append(
                    (point - np.asarray(scaler["mean"])[:2]) / np.asarray(scaler["scale"])[:2]
                )
            _save_npz(
                teacher_path,
                point_z=np.stack(teachers),
                origin_ids=np.asarray(origins),
                identity_sha256=np.asarray(identity_sha),
            )
        with np.load(teacher_path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or saved["origin_ids"].tolist() != origins
            ):
                raise ValueError("clean-history diagnostic cache changed identity")
            teacher_bank = saved["point_z"]
        if parameter_digest(backbone) != forecast["parameter_sha256"]:
            raise ValueError("the frozen diagnostic predictor was modified")
        calls = runner.resource_metrics()
        del runner, adapter, backbone
        torch.cuda.empty_cache()
        prediction_map = {row["episode_id"]: row for row in forecast["predictions"]}
        points, truth = [], []
        for record in records:
            entry = prediction_map[record["episode_id"]]
            path = forecast_dir / entry["path"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError("candidate forecast bank changed")
            with np.load(path, allow_pickle=False) as saved:
                order = saved["candidate_ids"].tolist()
                actions = sorted(order)
                points.append(saved["point_z"][[order.index(name) for name in actions]])
            source_path = args.prepared_root / record["path"]
            if file_sha256(source_path) != record["sha256"]:
                raise ValueError("an audited evaluation future changed")
            with np.load(source_path, allow_pickle=False) as saved:
                future = saved["future"]
            if not np.isfinite(future).all():
                raise ValueError("the synthetic future must retain complete ground truth")
            scaler = scalers[(record["dataset_id"], record["item_id"])]
            truth.append(
                (future - np.asarray(scaler["mean"])[:2]) / np.asarray(scaler["scale"])[:2]
            )
        points, truth = np.stack(points), np.stack(truth)
        teacher = teacher_bank[[origins.index(row["origin_id"]) for row in records]]
        subsets = list(combinations(range(7), 3))
        triples = np.stack([np.median(points[:, subset], axis=1) for subset in subsets], axis=1)
        outputs = {
            "forecast_median_seven": np.median(points, axis=1),
            "unavailable_clean_forecast": teacher,
            "unavailable_individual_teacher_top3": teacher_top_three(points, teacher, joint=joint),
        }
        outputs["unavailable_teacher_best3"], _ = best_option(
            triples, ((triples - teacher[:, None]) ** 2).mean(axis=2), joint=joint
        )
        primary_marker = json.loads(
            (policy_dir / "policy_predictions.json").read_text(encoding="utf-8")
        )
        if (
            file_sha256(policy_dir / "policy_predictions.npz")
            != primary_marker["prediction_sha256"]
        ):
            raise ValueError("primary policy predictions changed")
        with np.load(policy_dir / "policy_predictions.npz", allow_pickle=False) as saved:
            for name in (
                "median_risk",
                "member_risk",
                "old_teacher_rank3",
                "source_fixed_median_risk",
            ):
                outputs[name] = saved["point_z"][indices, saved["methods"].tolist().index(name)]
        for kind, bank in (("single", points), ("triple", triples)):
            for metric in ("mae", "mse"):
                errors = bank - truth[:, None]
                cost = (abs(errors) if metric == "mae" else errors**2).mean(axis=2)
                outputs[f"future_oracle_{kind}_{metric}"], _ = best_option(bank, cost, joint=joint)
        optimal = ((outputs["unavailable_teacher_best3"] - teacher) ** 2).mean(axis=(1, 2))
        for name in ("median_risk", "unavailable_individual_teacher_top3"):
            if np.any(optimal > ((outputs[name] - teacher) ** 2).mean(axis=(1, 2)) + 1e-10):
                raise ValueError("teacher triple optimization violated its finite-class bound")
        metadata = pd.DataFrame(records)[
            [
                "episode_id",
                "origin_id",
                "dataset_id",
                "family_id",
                "mechanism",
                "missing_rate",
                "mask_seed",
            ]
        ]
        original_scores = pd.read_parquet(policy_dir / "episode_results.parquet")
        for name, point in outputs.items():
            frame = metadata.assign(
                model_id=model,
                method=name,
                mae=abs(point - truth).mean(axis=(1, 2)),
                mse=((point - truth) ** 2).mean(axis=(1, 2)),
                teacher_mse=((point - teacher) ** 2).mean(axis=(1, 2)),
            )
            if name in {
                "median_risk",
                "member_risk",
                "old_teacher_rank3",
                "source_fixed_median_risk",
            }:
                expected = (
                    original_scores[original_scores.method == name]
                    .set_index("episode_id")
                    .loc[frame.episode_id]
                )
                np.testing.assert_allclose(
                    frame[["mae", "mse"]], expected[["mae", "mse"]], rtol=1e-10, atol=1e-10
                )
            all_rows.append(frame)
        path = output / f"{model}_diagnostic_predictions.npz"
        _save_npz(
            path,
            point_z=np.stack(list(outputs.values())),
            methods=np.asarray(list(outputs)),
            episode_ids=metadata.episode_id.to_numpy(str),
        )
        model_records.append(
            {
                "model_id": model,
                "teacher_sha256": file_sha256(teacher_path),
                "predictions_sha256": file_sha256(path),
                "runtime_current_process_only": calls,
            }
        )
    scores = pd.concat(all_rows, ignore_index=True)
    scores.to_parquet(output / "episode_metrics.parquet", index=False)
    family = scores.groupby(["model_id", "method", "family_id"])[
        ["mae", "mse", "teacher_mse"]
    ].mean()
    family.to_csv(output / "family_metrics.csv")
    family.groupby(["model_id", "method"]).mean().to_csv(output / "summary.csv")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "models": model_records,
            "synthetic_histories": 16,
            "synthetic_tasks": 576,
            "limits": "post-evaluation analysis using unavailable complete histories or future outcomes; these oracles are not deployable methods; natural-missing causes remain untested",
        },
    )


if __name__ == "__main__":
    main()
