"""Prepare a frozen two-horizon confirmation bank without reading forecast outcomes."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_followup_inputs import fit_deep  # noqa: E402
from prepare_native_confirmation import (  # noqa: E402
    ACTIONS,
    DEEP_PARAMS,
    MECHANISMS,
    complete_candidates,
)

from tsfm_fais.contracts import SeriesBatch, TimeSeriesItem  # noqa: E402
from tsfm_fais.data.masking import MaskingSpec, mask_time_series  # noqa: E402
from tsfm_fais.forecasting.accuracy import PrefixStandardizer  # noqa: E402
from tsfm_fais.imputers.motm import MOTMReference  # noqa: E402
from tsfm_fais.imputers.registry import DEFAULT_REGISTRY  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.stage_execution import _training_batch  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort-root", "method-freeze", "motm-reference", "motm-runtime", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed R6 candidate inputs")
    cohort = json.loads((args.cohort_root / "manifest.json").read_text(encoding="utf-8"))
    if cohort["status"] != "completed" or cohort["identity_sha256"]["method_freeze"] != file_sha256(
        args.method_freeze
    ):
        raise ValueError("the cohort and method freeze disagree")
    if cohort["horizons"] != [96, 192] or cohort["context_length"] != 96:
        raise ValueError("the registered context and horizons changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
        "method_freeze_sha256": file_sha256(args.method_freeze),
        "context_length": 96,
        "maximum_horizon": 192,
        "candidate_ids": list(ACTIONS),
        "deep_params": DEEP_PARAMS,
        "training_windows": 64,
        "training_stride": 24,
        "training_missing_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
        "training_mask_seeds": [1101, 1102, 1103],
        "motm_reference_sha256": file_sha256(args.motm_reference / "manifest.json"),
        "motm_runtime_sha256": file_sha256(args.motm_runtime / "manifest.json"),
        "runtime_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/prepare_followup_inputs.py",
                "scripts/prepare_native_confirmation.py",
                "src/tsfm_fais/stage_execution.py",
                "src/tsfm_fais/data/masking.py",
                "src/tsfm_fais/imputers/motm.py",
                "src/tsfm_fais/imputers/runner.py",
                "src/tsfm_fais/imputers/pypots.py",
                "src/tsfm_fais/imputers/classical.py",
                "src/tsfm_fais/imputers/structured.py",
                "src/tsfm_fais/forecasting/accuracy.py",
            )
        },
    }
    if (
        identity["runtime_source_sha256"]["src/tsfm_fais/data/masking.py"]
        != cohort["identity_sha256"]["masking"]
    ):
        raise ValueError("mask generation changed after registration")
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("partial R6 input preparation changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    runner = CandidateRunner()
    motm = MOTMReference(
        args.motm_reference, args.motm_runtime, device="cuda", ridge=0.5, batch_size=32
    )
    groups = defaultdict(list)
    for source in cohort["sources"]:
        groups[source["dataset_id"]].append(source)
    episodes, datasets, scalers = [], [], []
    for dataset, sources in groups.items():
        items = []
        for source in sources:
            path = Path(source["path"])
            if file_sha256(path) != source["sha256"]:
                raise ValueError("a registered original trajectory changed")
            items.append(
                TimeSeriesItem(
                    source["item_id"],
                    np.load(path, mmap_mode="r"),
                    tuple(source["columns"]),
                    pd.Timestamp(source["start"]),
                    source["frequency"],
                )
            )
        period = sources[0]["period"]
        prefixes = {source["item_id"]: source["prefix_end"] for source in sources}
        batch = _training_batch(
            items,
            96,
            192,
            64,
            dataset_id=dataset,
            masking_specs=[
                MaskingSpec(name, rate)
                for name in MECHANISMS
                for rate in identity["training_missing_rates"]
            ],
            configured_seeds=identity["training_mask_seeds"],
            fit_fraction=0.2,
            training_stride=24,
        )
        for identifier in batch.item_ids:
            item, start = identifier.rsplit("@", 1)
            if int(start.split("|", 1)[0]) + 96 > prefixes[item]:
                raise ValueError("a neural imputer training example extends beyond its prefix")
        dataset_dir = output / "imputers" / dataset
        artifacts, failures, fits = fit_deep(
            runner, batch, dataset_dir, identity_sha, period, dataset
        )
        for item in items:
            prefix = item.values[: prefixes[item.item_id]]
            scaler, defaults = PrefixStandardizer.fit(prefix), np.nanmedian(prefix, axis=0)
            scalers.append(
                {
                    "dataset_id": dataset,
                    "item_id": item.item_id,
                    "prefix_end": prefixes[item.item_id],
                    "mean": scaler.mean.tolist(),
                    "scale": scaler.scale.tolist(),
                    "constant": scaler.constant.tolist(),
                    "observed_count": scaler.observed_count.tolist(),
                    "fallback_medians": defaults.tolist(),
                }
            )
            local, params = dict(artifacts), {"seasonal_lag": {"period": max(2, period)}}
            prefix_batch = SeriesBatch(
                prefix[None], np.isfinite(prefix[None]), metadata={"period": period}
            )
            for name in ACTIONS:
                if (
                    name not in local
                    and name not in failures
                    and (
                        name == "seasonal_lag"
                        or DEFAULT_REGISTRY.get_spec(name).fit_scope == "dataset"
                    )
                ):
                    local[name] = runner.fit(
                        name, prefix_batch, {"period": period}, params=params.get(name)
                    )
            conditions = defaultdict(list)
            for task in cohort["tasks"]:
                if task["dataset_id"] == dataset and task["item_id"] == item.item_id:
                    conditions[
                        (task["mechanism"], task["missing_rate"], task["realization_seed"])
                    ].append(task)
            for (mechanism, rate, seed), tasks in conditions.items():
                masked = None
                for task in tasks:
                    key = hashlib.sha256(task["episode_id"].encode()).hexdigest()[:24]
                    path, marker = (
                        output / "episodes" / f"{key}.npz",
                        output / "episodes" / f"{key}.json",
                    )
                    if marker.exists():
                        result = json.loads(marker.read_text(encoding="utf-8"))
                        if (
                            result["identity_sha256"] != identity_sha
                            or file_sha256(path) != result["sha256"]
                        ):
                            raise ValueError("a completed R6 input changed")
                    else:
                        if mechanism != "native" and masked is None:
                            masked = mask_time_series(
                                item.values,
                                MaskingSpec(mechanism, rate),
                                seed,
                                calibration_values=prefix,
                            )
                        origin = task["window"]["origin"]
                        values = item.values if masked is None else masked.values
                        context = values[origin - 96 : origin].copy()
                        future = item.values[origin : origin + 192, :2].copy()
                        if future.shape != (192, 2):
                            raise ValueError("the maximum-horizon future is incomplete")
                        for horizon in (96, 192):
                            if (
                                np.isfinite(future[:horizon]).sum(0).tolist()
                                != task["window"]["future_observed_by_horizon"][str(horizon)]
                            ):
                                raise ValueError("registered future observation counts changed")
                        case_artifacts = dict(local)
                        if "seasonal_lag" in case_artifacts:
                            seasonal = dict(case_artifacts["seasonal_lag"])
                            seasonal["profiles"] = np.roll(
                                seasonal["profiles"], -(origin - 96) % seasonal["period"], axis=1
                            )
                            case_artifacts["seasonal_lag"] = seasonal
                        context_batch = SeriesBatch(
                            context[None], np.isfinite(context[None]), metadata={"period": period}
                        )
                        results = runner.run_many(
                            ACTIONS,
                            context_batch,
                            case_artifacts,
                            seed=int(key[:8], 16),
                            params=params,
                            artifact_failures=failures,
                        )
                        completed, coverage, statuses = complete_candidates(
                            context, results, defaults
                        )
                        motm_values, motm_diagnostics = motm.impute(
                            context, completed[list(ACTIONS).index("locf")]
                        )
                        np.testing.assert_array_equal(
                            motm_values[np.isfinite(context)], context[np.isfinite(context)]
                        )
                        if not np.isfinite(motm_values).all():
                            raise ValueError("the MoTM comparator is not finite")
                        _save_npz(
                            path,
                            context=context,
                            future=future,
                            future_observed=np.isfinite(future),
                            candidate_values=completed,
                            candidate_ids=np.asarray(ACTIONS),
                            native_coverage=coverage,
                            motm_values=motm_values,
                            motm_diagnostics=np.asarray(json.dumps(motm_diagnostics)),
                            identity_sha256=np.asarray(identity_sha),
                        )
                        result = {
                            **task,
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            "identity_sha256": identity_sha,
                            "candidate_statuses": statuses,
                            "motm_diagnostics": motm_diagnostics,
                            "realized_context_missing_fraction": float(
                                (~np.isfinite(context)).mean()
                            ),
                            "realization_id": None if masked is None else masked.realization_id,
                        }
                        _write_json(marker, result)
                    episodes.append(result)
                    _write_json(
                        output / "progress.json",
                        {
                            "status": "preparing",
                            "completed_episodes": len(episodes),
                            "total_episodes": cohort["task_count"],
                            "dataset_id": dataset,
                        },
                    )
        datasets.append(
            {
                "dataset_id": dataset,
                "fit_items": [item.item_id for item in items],
                "imputers": fits,
                "training_batch_sha256": file_sha256(dataset_dir / "training_batch.npz"),
            }
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "completed_inputs": len(episodes),
                    "total_inputs": cohort["task_count"],
                }
            ),
            flush=True,
        )
        del artifacts, local, batch, items, prefix_batch, masked
        torch.cuda.empty_cache()
    if len(episodes) != cohort["task_count"] or {row["episode_id"] for row in episodes} != {
        row["episode_id"] for row in cohort["tasks"]
    }:
        raise ValueError("R6 input coverage changed")
    motm.verify_frozen()
    _write_json(output / "standardizers.json", scalers)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "episodes": episodes,
            "datasets": datasets,
            "standardizers_sha256": file_sha256(output / "standardizers.json"),
            "forecaster_calls": 0,
            "forecast_scores_computed": False,
            "motm_networks_unchanged": True,
            "limits": "input preparation only; dual-horizon runtime and prediction/metric audits remain required",
        },
    )


if __name__ == "__main__":
    main()
