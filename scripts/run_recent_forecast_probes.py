"""Prepare and evaluate horizon-matched historical forecast probes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402

from tsfm_fais.contracts import ForecastSpec, SeriesBatch  # noqa: E402
from tsfm_fais.data import (  # noqa: E402
    MaskingSpec,
    load_dataset,
    load_manifest,
    mask_time_series,
    stable_seed,
)
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.accuracy import PrefixStandardizer, guarded_direct_forecast  # noqa: E402
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner  # noqa: E402
from tsfm_fais.routing.recent_feedback import plan_recent_probes  # noqa: E402
from tsfm_fais.utility_experiment import (  # noqa: E402
    UtilityExperimentConfig,
    _load_frozen_imputers,
    _save_npz,
    _write_json,
    file_sha256,
)


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FrozenCandidatePool:
    def __init__(self, config, dataset_id, prefix, period):
        self.config, self.period = config, period
        self.runner = CandidateRunner()
        self.artifacts, self.failures, self.fit_record = _load_frozen_imputers(config, dataset_id)
        self.artifacts = dict(self.artifacts)
        self.params = {"seasonal_lag": {"period": max(2, period)}}
        self.defaults = np.nanmedian(prefix, axis=0)
        self.standardizer = PrefixStandardizer.fit(prefix)
        if not np.isfinite(self.defaults).all():
            raise ValueError("imputer prefix has an unobserved channel")
        batch = SeriesBatch(prefix[None], np.isfinite(prefix[None]), metadata={"period": period})
        for action in config.candidate_ids:
            if action in self.artifacts or action in self.failures:
                continue
            if action == "seasonal_lag" or DEFAULT_REGISTRY.get_spec(action).fit_scope == "dataset":
                self.artifacts[action] = self.runner.fit(
                    action, batch, {"period": period}, params=self.params.get(action)
                )

    def complete(self, context, origin, seed):
        batch = SeriesBatch(
            context[None], np.isfinite(context[None]), metadata={"period": self.period}
        )
        artifacts = dict(self.artifacts)
        if "seasonal_lag" in artifacts:
            seasonal = dict(artifacts["seasonal_lag"])
            seasonal["profiles"] = np.roll(
                seasonal["profiles"], -(origin - len(context)) % seasonal["period"], axis=1
            )
            artifacts["seasonal_lag"] = seasonal
        candidates = self.runner.run_many(
            self.config.candidate_ids,
            batch,
            artifacts,
            seed=stable_seed(seed, origin),
            params=self.params,
            artifact_failures=self.failures,
        )
        anchor = candidates["locf"].values[0].copy()
        invalid = ~candidates["locf"].native_valid_mask[0]
        anchor[invalid] = np.broadcast_to(self.defaults, anchor.shape)[invalid]
        outputs, coverage, statuses, times = [], [], [], []
        missing = ~np.isfinite(context)
        for action in self.config.candidate_ids:
            candidate = candidates[action]
            valid = candidate.native_valid_mask[0].copy()
            if not DEFAULT_REGISTRY.get_spec(action).supports_tail:
                for channel in range(context.shape[1]):
                    available = np.flatnonzero(~missing[:, channel])
                    if len(available):
                        valid[available[-1] + 1 :, channel] = False
            completed = candidate.values[0].copy()
            completed[~valid] = anchor[~valid]
            if not np.isfinite(completed).all() or not np.array_equal(
                completed[~missing], context[~missing]
            ):
                raise ValueError("candidate completion changed observations or is nonfinite")
            outputs.append(completed)
            coverage.append(float(valid[missing].mean()) if missing.any() else 1.0)
            statuses.append(
                {
                    "candidate_id": action,
                    "status": candidate.status.value,
                    "failure_reason": candidate.failure_reason,
                }
            )
            times.append(candidate.runtime_seconds)
        return np.stack(outputs), np.asarray(coverage), statuses, np.asarray(times)


def prepare(source_root, output, plan, source):
    config = UtilityExperimentConfig.model_validate(source["identity"]["config"])
    identity = {
        "source_manifest_sha256": file_sha256(source_root / "episodes_manifest.json"),
        "plan_sha256": file_sha256(output / "plan.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "imputer_config": config.model_dump(mode="json"),
        "data_manifest_sha256": file_sha256(config.data_manifest),
        "imputer_sources": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in sorted((ROOT / "src/tsfm_fais/imputers").rglob("*.py"))
        },
        "masking_source_sha256": file_sha256(ROOT / "src/tsfm_fais/data/masking.py"),
    }
    identity_path = output / "preparation_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("probe preparation identity changed; use a new output root")
    _write_json(identity_path, identity)
    binding = canonical_digest(identity)
    if (output / "prepared_manifest.json").exists():
        saved = json.loads((output / "prepared_manifest.json").read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("completed probe preparation has another identity")
        for record in saved["probes"]:
            if file_sha256(output / record["path"]) != record["sha256"]:
                raise ValueError("completed probe cache changed")
        return
    data_manifest = load_manifest(config.data_manifest)
    source_records = {record["episode_id"]: record for record in source["episodes"]}
    groups = defaultdict(list)
    for probe in plan["probes"]:
        groups[(probe["dataset_id"], probe["item_id"])].append(probe)
    records, fit_records = [], []
    started = monotonic()
    for (dataset_id, item_id), probes in groups.items():
        dataset = next(entry for entry in source["datasets"] if entry["dataset_id"] == dataset_id)
        for path, digest in dataset["sources"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError("raw source changed")
        pending = []
        for probe in probes:
            path = output / probe["path"]
            sidecar = path.with_suffix(".json")
            if path.exists() and sidecar.exists():
                saved = json.loads(sidecar.read_text(encoding="utf-8"))
                if (
                    saved["binding"] != binding
                    or saved["probe_id"] != probe["probe_id"]
                    or saved["sha256"] != file_sha256(path)
                ):
                    raise ValueError("prepared probe changed or belongs to another protocol")
                records.append(saved)
            else:
                pending.append(probe)
        if not pending:
            continue
        item = next(
            item for item in load_dataset(data_manifest.get(dataset_id)) if item.item_id == item_id
        )
        prefix_end = pending[0]["prefix_end"]
        prefix = item.values[:prefix_end]
        scaler = PrefixStandardizer.fit(prefix)
        pool = None
        masking_groups = defaultdict(list)
        for probe in pending:
            masking_groups[
                (probe["split"], probe["mechanism"], probe["missing_rate"], probe["mask_seed"])
            ].append(probe)
        for (split, mechanism, rate, seed), mask_probes in masking_groups.items():
            realized = mask_time_series(
                item.values,
                MaskingSpec(mechanism, rate, config.block_lengths),
                stable_seed(config.protocol_id, dataset_id, item_id, split, mechanism, rate, seed),
                calibration_values=prefix,
            )
            for probe in mask_probes:
                origin, horizon = probe["origin"], probe["horizon"]
                context = realized.values[origin - config.context_length : origin].copy()
                observed_future = realized.values[origin : origin + horizon][
                    :, list(config.target_indices)
                ].copy()
                if (
                    origin - config.context_length < prefix_end
                    or len(context) != config.context_length
                    or len(observed_future) != horizon
                ):
                    raise ValueError("probe window is outside its allowed historical interval")
                reused = probe["reusable_episode_id"]
                if reused is not None:
                    record = source_records[reused]
                    path = source_root / record["path"]
                    if file_sha256(path) != record["sha256"]:
                        raise ValueError("reused candidate episode changed")
                    with np.load(path, allow_pickle=False) as cached:
                        np.testing.assert_equal(context, cached["context"])
                        completed = cached["candidate_values"]
                        coverage = cached["native_coverage"]
                    statuses = record["candidate_status"]
                    times = np.zeros(len(config.candidate_ids))
                else:
                    if pool is None:
                        pool = FrozenCandidatePool(config, dataset_id, prefix, probe["period"])
                        expected = dataset["fit_artifacts"].get("manifest_sha256")
                        if (
                            expected is not None
                            and pool.fit_record.get("manifest_sha256") != expected
                        ):
                            raise ValueError("frozen imputer artifacts changed")
                        fit_records.append(
                            {"dataset_id": dataset_id, "item_id": item_id, **pool.fit_record}
                        )
                    completed, coverage, statuses, times = pool.complete(context, origin, seed)
                path = output / probe["path"]
                _save_npz(
                    path,
                    probe_id=np.asarray(probe["probe_id"]),
                    context=context,
                    observed_future_z=(observed_future - scaler.mean[list(config.target_indices)])
                    / scaler.scale[list(config.target_indices)],
                    candidate_values=completed,
                    candidate_ids=np.asarray(config.candidate_ids),
                    native_coverage=coverage,
                    candidate_runtime=times,
                    prefix_mean=scaler.mean,
                    prefix_scale=scaler.scale,
                    binding=np.asarray(binding),
                )
                record = {
                    "probe_id": probe["probe_id"],
                    "path": probe["path"],
                    "sha256": file_sha256(path),
                    "binding": binding,
                    "candidate_status": statuses,
                    "reused_candidate_episode": reused,
                }
                _write_json(path.with_suffix(".json"), record)
                records.append(record)
                if len(records) % 50 == 0:
                    state = {
                        "completed": len(records),
                        "total": len(plan["probes"]),
                        "elapsed_seconds": monotonic() - started,
                    }
                    _write_json(output / "prepare_progress.json", state)
                    print(json.dumps(state), flush=True)
        del pool, item
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_json(
        output / "prepared_manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "probes": records,
            "fit_records": fit_records,
            "elapsed_seconds": monotonic() - started,
        },
    )


def forecast(source_root, accuracy_root, output, plan, source, model):
    import torch

    torch.set_num_threads(1)
    config = UtilityExperimentConfig.model_validate(source["identity"]["config"])
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((output / "prepared_manifest.json").read_text(encoding="utf-8"))
    expected_source = file_sha256(source_root / "episodes_manifest.json")
    if (
        accuracy["source_episode_manifest_sha256"] != expected_source
        or prepared["identity"]["source_manifest_sha256"] != expected_source
    ):
        raise ValueError("probe and current forecasts do not share the same source")
    if not accuracy.get("chronos_repair_manifest_sha256"):
        raise ValueError(
            "probe experiments require source forecasts with the corrected GPU adapter"
        )
    destination = output / model
    destination.mkdir(parents=True, exist_ok=True)
    identity = {
        "prepared_manifest_sha256": file_sha256(output / "prepared_manifest.json"),
        "accuracy_manifest_sha256": file_sha256(accuracy_root / "manifest.json"),
        "model_id": model,
        "script_sha256": file_sha256(Path(__file__)),
        "checkpoint": str(config.forecaster_artifacts[model]),
        "adapter_sha256": file_sha256(
            ROOT
            / "src/tsfm_fais/forecasting/adapters"
            / ("timesfm.py" if model == "timesfm2p5" else "chronos.py")
        ),
        "accuracy_source_sha256": file_sha256(ROOT / "src/tsfm_fais/forecasting/accuracy.py"),
    }
    identity_path = destination / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("probe forecast identity changed")
    _write_json(identity_path, identity)
    if (destination / "manifest.json").exists():
        saved = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("completed probe predictions have another identity")
        for record in saved["probes"]:
            if file_sha256(output / record["path"]) != record["sha256"]:
                raise ValueError("completed probe prediction changed")
        return
    source_indices = {
        record["episode_id"]: index for index, record in enumerate(source["episodes"])
    }
    shared = np.load(accuracy_root / f"{model}_point_z.npy", mmap_mode="r")
    shared_ids = accuracy["action_orders"][model]
    actions = [*config.candidate_ids, "guarded_direct"]
    shared_positions = [shared_ids.index(action) for action in actions]
    registry = default_forecast_registry()
    adapter = (
        TimesFMVendorMissingAdapter(
            model_name=str(config.forecaster_artifacts[model]), device="cuda", batch_size=8
        )
        if model == "timesfm2p5"
        else registry.build(
            model, model_name=str(config.forecaster_artifacts[model]), device="cuda", batch_size=32
        )
    )
    runner = ForecastRunner(registry, {model: adapter})
    cached_records = {record["probe_id"]: record for record in prepared["probes"]}
    records = []
    reused_count = 0
    started = monotonic()
    for index, probe in enumerate(plan["probes"]):
        record = cached_records[probe["probe_id"]]
        path = output / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("prepared probe changed")
        target = destination / "predictions" / path.name
        if not target.exists():
            with np.load(path, allow_pickle=False) as e:
                reused = probe["reusable_episode_id"]
                if reused is not None and probe["horizon"] == config.horizon:
                    point_z = np.asarray(shared[source_indices[reused]][shared_positions])
                    reused_count += 1
                else:
                    spec = ForecastSpec(
                        model,
                        registry.get(model).mode,
                        probe["horizon"],
                        context_length=config.context_length,
                        target_indices=config.target_indices,
                    )
                    complete = runner.predict(e["candidate_values"], spec).point
                    direct = runner.predict_missing(e["context"][None], spec).point[0]
                    guarded, _ = guarded_direct_forecast(
                        e["context"],
                        config.target_indices,
                        direct,
                        complete[config.candidate_ids.index("locf")],
                        joint=model == "chronos2",
                    )
                    values = np.concatenate([complete, guarded[None]])
                    targets = list(config.target_indices)
                    point_z = (values - e["prefix_mean"][targets]) / e["prefix_scale"][targets]
                _save_npz(
                    target,
                    point_z=point_z,
                    probe_sha256=np.asarray(record["sha256"]),
                    action_ids=np.asarray(actions),
                )
        with np.load(target, allow_pickle=False) as prediction:
            if str(prediction["probe_sha256"]) != record["sha256"]:
                raise ValueError("probe prediction belongs to another context")
        records.append(
            {
                "probe_id": probe["probe_id"],
                "path": str(target.relative_to(output)),
                "sha256": file_sha256(target),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(plan["probes"]):
            state = {
                "completed": index + 1,
                "total": len(plan["probes"]),
                "elapsed_seconds": monotonic() - started,
            }
            _write_json(destination / "progress.json", state)
            print(json.dumps(state), flush=True)
    _write_json(
        destination / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "probes": records,
            "action_ids": actions,
            "reused_predictions_this_execution": reused_count,
            "resources_this_execution": runner.resource_metrics(),
            "elapsed_seconds": monotonic() - started,
            "cost_note": "Count all logical probe requests in cold-start deployment cost; existing offline caches do not make their information free.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "prepare", "forecast"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--accuracy-root", type=Path)
    parser.add_argument("--model", choices=("chronos2", "timesfm2p5"))
    parser.add_argument("--offsets", default="1,2")
    parser.add_argument("--horizons", default="")
    parser.add_argument("--screening", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve()
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    if args.stage == "plan":
        decision_ids = None
        if args.screening:
            validation = [
                record for record in source["episodes"] if record["split"] == "validation"
            ]
            first = {}
            for record in validation:
                key = (record["dataset_id"], record["item_id"])
                first[key] = min(record["origin"], first.get(key, record["origin"]))
            config = source["identity"]["config"]
            rates = (min(config["missing_rates"]), max(config["missing_rates"]))
            decision_ids = [
                record["episode_id"]
                for record in validation
                if record["origin"] == first[(record["dataset_id"], record["item_id"])]
                and record["mechanism"] in {"random_point", "independent_block", "value_dependent"}
                and record["missing_rate"] in rates
                and record["mask_seed"] == min(config["mask_seeds"])
            ]
        plan = plan_recent_probes(
            source,
            tuple(map(int, args.offsets.split(","))),
            tuple(map(int, args.horizons.split(","))) if args.horizons else None,
            decision_ids=decision_ids,
        )
        plan["screening_rule"] = (
            "first validation origin per source item; random, independent-block and value-dependent masks; smallest/largest configured rate; first seed"
            if args.screening
            else None
        )
        plan["source_manifest_sha256"] = file_sha256(source_root / "episodes_manifest.json")
        if (output / "plan.json").exists() and json.loads(
            (output / "plan.json").read_text(encoding="utf-8")
        ) != plan:
            raise ValueError("probe plan changed; use a new output root")
        _write_json(output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "probes": len(plan["probes"]),
                    "links": len(plan["links"]),
                    "reuse_candidates": sum(
                        probe["reusable_episode_id"] is not None for probe in plan["probes"]
                    ),
                    "skipped": len(plan["skipped"]),
                }
            ),
            flush=True,
        )
        return
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan["source_manifest_sha256"] != file_sha256(source_root / "episodes_manifest.json"):
        raise ValueError("probe plan source changed")
    if args.stage == "prepare":
        import torch

        torch.set_num_threads(1)
        prepare(source_root, output, plan, source)
    else:
        if args.model is None or args.accuracy_root is None:
            parser.error("forecast requires model and accuracy-root")
        forecast(source_root, args.accuracy_root.resolve(), output, plan, source, args.model)


if __name__ == "__main__":
    main()
