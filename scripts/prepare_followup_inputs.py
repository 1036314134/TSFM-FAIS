"""Prepare frozen follow-up inputs using each dataset's observed historical prefixes."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_native_confirmation import (  # noqa: E402
    ACTIONS,
    DEEP_PARAMS,
    MECHANISMS,
    complete_candidates,
    saved_files,
)

from tsfm_fais.contracts import SeriesBatch, TimeSeriesItem  # noqa: E402
from tsfm_fais.data.masking import MaskingSpec, mask_time_series  # noqa: E402
from tsfm_fais.forecasting.accuracy import PrefixStandardizer  # noqa: E402
from tsfm_fais.imputers.registry import DEFAULT_REGISTRY  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.stage_execution import _training_batch  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def fit_deep(runner, batch, directory, identity_sha, period, dataset):
    directory.mkdir(parents=True, exist_ok=True)
    training_path = directory / "training_batch.npz"
    if training_path.exists():
        with np.load(training_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["values"], batch.values)
            np.testing.assert_array_equal(saved["observed"], batch.observed_mask)
            assert saved["window_ids"].tolist() == list(batch.item_ids)
    else:
        _save_npz(
            training_path,
            values=batch.values,
            observed=batch.observed_mask,
            window_ids=np.asarray(batch.item_ids),
        )
    artifacts, failures, records = {}, {}, []
    for name in ("saits", "timemixerpp"):
        marker, artifact_dir = directory / f"{name}.json", directory / name
        adapter = DEFAULT_REGISTRY.create(name, **DEEP_PARAMS)
        if marker.exists():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record["identity_sha256"] != identity_sha:
                raise ValueError("the prefix fit belongs to another preparation")
            if record["status"] == "fitted":
                for row in record["files"]:
                    if file_sha256(artifact_dir / row["path"]) != row["sha256"]:
                        raise ValueError("a saved prefix imputer changed")
                artifacts[name] = adapter.load_artifact(artifact_dir)
            else:
                failures[name] = record["reason"]
        else:
            started = monotonic()
            try:
                artifact = runner.fit(
                    name, batch, {"period": period, "dataset_id": dataset}, params=DEEP_PARAMS
                )
                adapter.save_artifact(artifact, artifact_dir)
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
                record = {"status": "failed", "reason": failures[name]}
            else:
                artifacts[name] = artifact
                record = {"status": "fitted", "files": saved_files(artifact_dir)}
            record.update(
                identity_sha256=identity_sha,
                seconds=monotonic() - started,
                training_windows=batch.shape[0],
            )
            _write_json(marker, record)
        records.append(
            {
                "candidate_id": name,
                "status": record["status"],
                "path": str(marker),
                "sha256": file_sha256(marker),
            }
        )
    return artifacts, failures, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort-root", "source-bundle", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed follow-up preparations")
    cohort = json.loads((args.cohort_root / "manifest.json").read_text(encoding="utf-8"))
    if cohort["status"] != "completed" or cohort["task_count"] != len(cohort["tasks"]):
        raise ValueError("a complete frozen cohort is required")
    if (
        file_sha256(args.source_bundle / "manifest.json")
        != cohort["identity_sha256"]["source_bundle"]
    ):
        raise ValueError("the source selector bundle changed after cohort freeze")
    identity = {
        "cohort_sha256": file_sha256(args.cohort_root / "manifest.json"),
        "source_bundle_sha256": file_sha256(args.source_bundle / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "candidate_ids": list(ACTIONS),
        "deep_params": DEEP_PARAMS,
        "prefix_fraction": 0.2,
        "maximum_training_windows": 64,
        "training_stride": 24,
        "training_mask_mechanisms": list(MECHANISMS),
        "training_missing_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
        "training_mask_seeds": [1101, 1102, 1103],
        "runtime_source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/prepare_native_confirmation.py",
                "src/tsfm_fais/data/masking.py",
                "src/tsfm_fais/stage_execution.py",
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
        raise ValueError("partial follow-up inputs changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    groups = defaultdict(list)
    for source in cohort["sources"]:
        groups[source["dataset_id"]].append(source)
    runner, episodes, standardizers, datasets = CandidateRunner(), [], [], []
    for dataset, sources in groups.items():
        items = []
        for source in sources:
            path = args.cohort_root / source["path"]
            if file_sha256(path) != source["sha256"]:
                raise ValueError("a frozen raw trajectory changed")
            items.append(
                TimeSeriesItem(
                    source["item_id"],
                    np.load(path, mmap_mode="r"),
                    tuple(source["columns"]),
                    pd.NaT if source["start"] is None else pd.Timestamp(source["start"]),
                    source["frequency"],
                )
            )
        period = sources[0]["period"]
        prefix_by_item = {row["item_id"]: row["prefix_end"] for row in sources}
        batch = _training_batch(
            items,
            96,
            96,
            64,
            dataset_id=dataset,
            masking_specs=[
                MaskingSpec(name, rate, (6, 12, 24, 48))
                for name in MECHANISMS
                for rate in identity["training_missing_rates"]
            ],
            configured_seeds=identity["training_mask_seeds"],
            fit_fraction=0.2,
            training_stride=24,
        )
        for identifier in batch.item_ids:
            item_id, rest = identifier.rsplit("@", 1)
            if int(rest.split("|", 1)[0]) + 96 > prefix_by_item[item_id]:
                raise ValueError("an imputer fitting window extends beyond its prefix")
        dataset_dir = output / "imputers" / dataset
        artifacts, failures, fits = fit_deep(
            runner, batch, dataset_dir, identity_sha, period, dataset
        )
        for item in items:
            prefix_end = prefix_by_item[item.item_id]
            prefix = item.values[:prefix_end]
            scaler, defaults = PrefixStandardizer.fit(prefix), np.nanmedian(prefix, axis=0)
            standardizers.append(
                {
                    "dataset_id": dataset,
                    "item_id": item.item_id,
                    "prefix_end": prefix_end,
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
            for (mechanism, rate, realization_seed), tasks in conditions.items():
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
                            raise ValueError("a partially prepared follow-up episode changed")
                    else:
                        if mechanism != "native" and masked is None:
                            masked = mask_time_series(
                                item.values,
                                MaskingSpec(mechanism, rate),
                                realization_seed,
                                calibration_values=prefix,
                            )
                        origin = task["window"]["origin"]
                        raw = item.values if masked is None else masked.values
                        context = raw[origin - 96 : origin].copy()
                        future = item.values[origin : origin + 96, :2].copy()
                        if (
                            np.isfinite(future).sum(axis=0).tolist()
                            != task["window"]["future_observed_by_target"]
                        ):
                            raise ValueError("future observation counts changed")
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
                        _save_npz(
                            path,
                            context=context,
                            future=future,
                            future_observed=np.isfinite(future),
                            candidate_values=completed,
                            candidate_ids=np.asarray(ACTIONS),
                            native_coverage=coverage,
                            identity_sha256=np.asarray(identity_sha),
                        )
                        result = {
                            **task,
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            "identity_sha256": identity_sha,
                            "candidate_statuses": statuses,
                            "realization_id": None if masked is None else masked.realization_id,
                            "realized_context_missing_fraction": float(
                                (~np.isfinite(context)).mean()
                            ),
                            "realized_target_missing_counts": (~np.isfinite(context[:, :2]))
                            .sum(axis=0)
                            .tolist(),
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
                    "completed_episodes": len(episodes),
                    "total_episodes": cohort["task_count"],
                }
            ),
            flush=True,
        )
        del artifacts, local, batch, items, prefix_batch, masked
        torch.cuda.empty_cache()
    expected = {task["episode_id"] for task in cohort["tasks"]}
    if len(episodes) != len(expected) or {row["episode_id"] for row in episodes} != expected:
        raise ValueError("follow-up preparation coverage is incomplete or duplicated")
    _write_json(output / "standardizers.json", standardizers)
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
        },
    )


if __name__ == "__main__":
    main()
