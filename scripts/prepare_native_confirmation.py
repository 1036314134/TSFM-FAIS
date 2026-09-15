"""Prepare registered natural-missing histories using historical-prefix fitting only."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.contracts import SeriesBatch  # noqa: E402
from tsfm_fais.data.catalog import DatasetSpec  # noqa: E402
from tsfm_fais.data.episodes import fit_prefix_end  # noqa: E402
from tsfm_fais.data.loaders import load_dataset  # noqa: E402
from tsfm_fais.data.masking import MaskingSpec  # noqa: E402
from tsfm_fais.forecasting.accuracy import PrefixStandardizer  # noqa: E402
from tsfm_fais.imputers.registry import DEFAULT_REGISTRY  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.stage_execution import _training_batch  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402

ACTIONS = ("locf", "linear_interp", "seasonal_lag", "knn_multivariate", "saits", "timemixerpp")
MECHANISMS = (
    "random_point",
    "independent_block",
    "synchronous_block",
    "staggered_correlated",
    "value_dependent",
    "mixed_outage",
)
DEEP_PARAMS = {
    "epochs": 10,
    "batch_size": 16,
    "random_state": 0,
    "num_samples": 5,
    "device": "cuda",
}


def complete_candidates(context, results, defaults, actions=ACTIONS):
    observed = np.isfinite(context)
    anchor = results["locf"].values[0].copy()
    anchor[~results["locf"].native_valid_mask[0]] = np.broadcast_to(defaults, context.shape)[
        ~results["locf"].native_valid_mask[0]
    ]
    if not np.isfinite(anchor).all():
        raise ValueError("the shared historical fallback is not finite")
    outputs, coverage, statuses = [], [], []
    for name in actions:
        result = results[name]
        valid = result.native_valid_mask[0].copy()
        if not DEFAULT_REGISTRY.get_spec(name).supports_tail:
            for channel in range(context.shape[1]):
                indices = np.flatnonzero(observed[:, channel])
                if len(indices):
                    valid[indices[-1] + 1 :, channel] = False
        value = result.values[0].copy()
        value[~valid] = anchor[~valid]
        np.testing.assert_array_equal(value[observed], context[observed])
        if not np.isfinite(value).all():
            raise ValueError("a completed candidate is not finite")
        outputs.append(value)
        coverage.append(float(valid[~observed].mean()) if (~observed).any() else 1.0)
        statuses.append(
            {
                "candidate_id": name,
                "status": result.status.value,
                "failure_reason": result.failure_reason,
                "seconds": result.runtime_seconds,
            }
        )
    return np.stack(outputs), np.asarray(coverage), statuses


def saved_files(directory):
    return [
        {"path": str(path.relative_to(directory)), "sha256": file_sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed confirmation preparations")
    output.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    if cohort["task_count"] != len(cohort["tasks"]) or cohort["task_count"] != 373:
        raise ValueError("the registered confirmation cohort changed")
    first_source = Path(next(iter(cohort["sources"][0]["source_sha256"])))
    properties_path = first_source.parent.parent.parent / "dataset_properties.json"
    properties = json.loads(properties_path.read_text(encoding="utf-8"))
    identity = {
        "cohort_sha256": file_sha256(args.cohort),
        "properties_sha256": file_sha256(properties_path),
        "script_sha256": file_sha256(Path(__file__)),
        "candidate_ids": list(ACTIONS),
        "deep_params": DEEP_PARAMS,
        "prefix_fraction": 0.2,
        "maximum_training_windows": 64,
        "training_stride": 24,
        "training_mask_mechanisms": list(MECHANISMS),
        "training_missing_rates": [0.1, 0.2, 0.3, 0.4, 0.5],
        "training_mask_seeds": [1101, 1102, 1103],
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/stage_execution.py",
                "src/tsfm_fais/imputers/runner.py",
                "src/tsfm_fais/imputers/pypots.py",
                "src/tsfm_fais/imputers/classical.py",
                "src/tsfm_fais/imputers/structured.py",
                "src/tsfm_fais/data/loaders.py",
                "src/tsfm_fais/forecasting/accuracy.py",
            )
        },
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("confirmation preparation identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    runner, records, dataset_records, standardizers = CandidateRunner(), [], [], []
    for source in cohort["sources"]:
        dataset_id = source["dataset_id"]
        source_paths = [Path(name) for name in source["source_sha256"]]
        for name, expected_sha in source["source_sha256"].items():
            if file_sha256(Path(name)) != expected_sha:
                raise ValueError("a registered source file changed")
        if len({path.parent for path in source_paths}) != 1:
            raise ValueError("a dataset has inconsistent source directories")
        meta = properties[dataset_id]
        tasks = [task for task in cohort["tasks"] if task["dataset_id"] == dataset_id]
        spec = DatasetSpec(
            dataset_id=dataset_id,
            family_id=source["family_id"],
            format="arrow",
            path=source_paths[0].parent,
            frequency=dataset_id.rsplit("_", 1)[1],
            period=meta["period"],
            expected_num_variates=meta["num_variates"],
            missingness="native",
            provenance="source_native_missing",
            allow_implicit_regular_time=True,
        )
        item_map = {item.item_id: item for item in load_dataset(spec)}
        item_ids = list(dict.fromkeys(task["item_id"] for task in tasks))
        items = [item_map[name] for name in item_ids]
        expected_items = {
            name: next(task for task in tasks if task["item_id"] == name) for name in item_ids
        }
        for item in items:
            expected = expected_items[item.item_id]
            if (
                len(item.values) != expected["series_length"]
                or item.values.shape[1] != expected["dimensions"]
                or fit_prefix_end(len(item.values), 96, 96, 0.2) != expected["prefix_end"]
            ):
                raise ValueError("a registered series or prefix changed")
        batch = _training_batch(
            items,
            96,
            96,
            64,
            dataset_id=dataset_id,
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
            item, rest = identifier.rsplit("@", 1)
            start = int(rest.split("|", 1)[0])
            if start + 96 > expected_items[item]["prefix_end"]:
                raise ValueError("a training window extends beyond its historical prefix")
        dataset_dir = output / "imputers" / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _save_npz(
            dataset_dir / "training_batch.npz",
            values=batch.values,
            observed=batch.observed_mask,
            window_ids=np.asarray(batch.item_ids),
        )
        artifacts, failures, fits = {}, {}, []
        for name in ("saits", "timemixerpp"):
            marker = dataset_dir / f"{name}.json"
            artifact_dir = dataset_dir / name
            adapter = DEFAULT_REGISTRY.create(name, **DEEP_PARAMS)
            if marker.exists():
                fitted = json.loads(marker.read_text(encoding="utf-8"))
                if fitted["identity_sha256"] != identity_sha:
                    raise ValueError("a fitted imputer belongs to another preparation")
                if fitted["status"] == "fitted":
                    for record in fitted["files"]:
                        if file_sha256(artifact_dir / record["path"]) != record["sha256"]:
                            raise ValueError("a frozen target-prefix imputer changed")
                    artifacts[name] = adapter.load_artifact(artifact_dir)
                else:
                    failures[name] = fitted["reason"]
            else:
                started = monotonic()
                try:
                    artifact = runner.fit(
                        name,
                        batch,
                        {"period": spec.period, "dataset_id": dataset_id},
                        params=DEEP_PARAMS,
                    )
                    adapter.save_artifact(artifact, artifact_dir)
                except Exception as error:
                    failures[name] = f"{type(error).__name__}: {error}"
                    fitted = {"status": "failed", "reason": failures[name]}
                else:
                    artifacts[name] = artifact
                    fitted = {"status": "fitted", "files": saved_files(artifact_dir)}
                fitted.update(
                    identity_sha256=identity_sha,
                    seconds=monotonic() - started,
                    training_windows=batch.shape[0],
                )
                _write_json(marker, fitted)
            fits.append(
                {
                    "candidate_id": name,
                    "path": str(marker.relative_to(output)),
                    "sha256": file_sha256(marker),
                }
            )
        for item in items:
            prefix_end = expected_items[item.item_id]["prefix_end"]
            prefix = item.values[:prefix_end]
            scaler = PrefixStandardizer.fit(prefix)
            defaults = np.nanmedian(prefix, axis=0)
            standardizers.append(
                {
                    "dataset_id": dataset_id,
                    "item_id": item.item_id,
                    "prefix_end": prefix_end,
                    "mean": scaler.mean.tolist(),
                    "scale": scaler.scale.tolist(),
                    "constant": scaler.constant.tolist(),
                    "observed_count": scaler.observed_count.tolist(),
                    "fallback_medians": defaults.tolist(),
                }
            )
            params = {"seasonal_lag": {"period": max(2, spec.period)}}
            local = dict(artifacts)
            prefix_batch = SeriesBatch(
                prefix[None], np.isfinite(prefix[None]), metadata={"period": spec.period}
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
                        name, prefix_batch, {"period": spec.period}, params=params.get(name)
                    )
            for task in [task for task in tasks if task["item_id"] == item.item_id]:
                origin = task["window"]["origin"]
                context = item.values[origin - 96 : origin].copy()
                future = item.values[origin : origin + 96, :2].copy()
                if (
                    np.isfinite(future).sum(axis=0).tolist()
                    != task["window"]["future_observed_by_target"]
                    or (~np.isfinite(context[:, :2])).sum(axis=0).tolist()
                    != task["window"]["context_missing_by_target"]
                ):
                    raise ValueError("registered observation masks changed")
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
                        raise ValueError("a prepared native episode changed")
                else:
                    case_artifacts = dict(local)
                    if "seasonal_lag" in case_artifacts:
                        seasonal = dict(case_artifacts["seasonal_lag"])
                        seasonal["profiles"] = np.roll(
                            seasonal["profiles"], -(origin - 96) % seasonal["period"], axis=1
                        )
                        case_artifacts["seasonal_lag"] = seasonal
                    batch_context = SeriesBatch(
                        context[None], np.isfinite(context[None]), metadata={"period": spec.period}
                    )
                    seed = int(key[:8], 16)
                    results = runner.run_many(
                        ACTIONS,
                        batch_context,
                        case_artifacts,
                        seed=seed,
                        params=params,
                        artifact_failures=failures,
                    )
                    completed, coverage, statuses = complete_candidates(context, results, defaults)
                    _save_npz(
                        path,
                        context=context,
                        long_context=item.values[origin - 1024 : origin].copy(),
                        future=future,
                        future_observed=np.isfinite(future),
                        candidate_values=completed,
                        candidate_ids=np.asarray(ACTIONS),
                        native_coverage=coverage,
                        identity_sha256=np.asarray(identity_sha),
                    )
                    result = {
                        **task,
                        "origin_id": f"{dataset_id}|{item.item_id}|{origin}",
                        "period": spec.period,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                        "identity_sha256": identity_sha,
                        "candidate_statuses": statuses,
                    }
                    _write_json(marker, result)
                records.append(result)
                _write_json(
                    output / "progress.json",
                    {
                        "status": "preparing",
                        "completed_episodes": len(records),
                        "total_episodes": 373,
                        "dataset_id": dataset_id,
                    },
                )
        dataset_records.append(
            {
                "dataset_id": dataset_id,
                "fit_items": item_ids,
                "training_batch_sha256": file_sha256(dataset_dir / "training_batch.npz"),
                "imputers": fits,
                "source_sha256": source["source_sha256"],
            }
        )
        print(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "completed_episodes": len(records),
                    "total_episodes": 373,
                }
            ),
            flush=True,
        )
        del artifacts, local, batch
        torch.cuda.empty_cache()
    if len(records) != 373 or {record["episode_id"] for record in records} != {
        task["episode_id"] for task in cohort["tasks"]
    }:
        raise ValueError("confirmation preparation lost or duplicated episodes")
    _write_json(output / "standardizers.json", standardizers)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "episodes": records,
            "datasets": dataset_records,
            "standardizers_sha256": file_sha256(output / "standardizers.json"),
            "forecaster_calls": 0,
            "forecast_scores_computed": False,
        },
    )
    _write_json(
        output / "progress.json",
        {"status": "completed", "completed_episodes": len(records), "total_episodes": 373},
    )


if __name__ == "__main__":
    main()
