"""Control for longer history and prefix-standardized forecasting inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_timesfm_vendor_missing import TimesFMVendorMissingAdapter  # noqa: E402
from run_recent_forecast_probes import FrozenCandidatePool  # noqa: E402

from tsfm_fais.contracts import ForecastSpec  # noqa: E402
from tsfm_fais.data import (  # noqa: E402
    MaskingSpec,
    load_dataset,
    load_manifest,
    mask_time_series,
    stable_seed,
)
from tsfm_fais.forecasting import ForecastRunner, default_forecast_registry  # noqa: E402
from tsfm_fais.forecasting.accuracy import PrefixStandardizer, guarded_direct_forecast  # noqa: E402
from tsfm_fais.utility_experiment import (  # noqa: E402
    UtilityExperimentConfig,
    _save_npz,
    _write_json,
    file_sha256,
)


def carry_forward(values, defaults):
    array = np.asarray(values, float)
    observed = np.isfinite(array)
    indices = np.maximum.accumulate(np.where(observed, np.arange(len(array))[:, None], -1), axis=0)
    filled = array[np.maximum(indices, 0), np.arange(array.shape[1])[None, :]]
    return np.where(indices >= 0, filled, np.asarray(defaults)[None, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--accuracy-root", required=True, type=Path)
    parser.add_argument("--probe-plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("chronos2", "timesfm2p5"))
    parser.add_argument("--lengths", default="288,1024")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    root, accuracy_root, output = (
        args.source_root.resolve(),
        args.accuracy_root.resolve(),
        args.output_root.resolve() / args.model,
    )
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((root / "episodes_manifest.json").read_text(encoding="utf-8"))
    accuracy = json.loads((accuracy_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(args.probe_plan.read_text(encoding="utf-8"))
    expected = file_sha256(root / "episodes_manifest.json")
    if (
        plan["source_manifest_sha256"] != expected
        or accuracy["source_episode_manifest_sha256"] != expected
    ):
        raise ValueError("history controls must share the original evaluation episodes")
    config = UtilityExperimentConfig.model_validate(source["identity"]["config"])
    lengths = tuple(map(int, args.lengths.split(",")))
    if not lengths or min(lengths) < config.context_length:
        parser.error("control lengths must be at least the original context length")
    identity = {
        "source_manifest_sha256": expected,
        "accuracy_manifest_sha256": file_sha256(accuracy_root / "manifest.json"),
        "plan_sha256": file_sha256(args.probe_plan),
        "script_sha256": file_sha256(Path(__file__)),
        "lengths": lengths,
        "model_id": args.model,
    }
    path = output / "identity.json"
    serial = json.loads(json.dumps(identity))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != serial:
        raise ValueError("history-control identity changed")
    _write_json(path, identity)
    decisions = set(plan["decision_episode_ids"])
    source_indices = {
        record["episode_id"]: index for index, record in enumerate(source["episodes"])
    }
    groups = defaultdict(list)
    for record in source["episodes"]:
        if record["episode_id"] in decisions:
            groups[(record["dataset_id"], record["item_id"])].append(record)
    data_manifest = load_manifest(config.data_manifest)
    truth = np.load(accuracy_root / "truth_z.npy", mmap_mode="r")
    registry = default_forecast_registry()
    adapter = (
        TimesFMVendorMissingAdapter(
            model_name=str(config.forecaster_artifacts[args.model]), device="cuda", batch_size=8
        )
        if args.model == "timesfm2p5"
        else registry.build(
            args.model,
            model_name=str(config.forecaster_artifacts[args.model]),
            device="cuda",
            batch_size=8,
        )
    )
    runner = ForecastRunner(registry, {args.model: adapter})
    rows, files = [], []
    targets = list(config.target_indices)
    for (dataset_id, item_id), records in groups.items():
        dataset = next(entry for entry in source["datasets"] if entry["dataset_id"] == dataset_id)
        for file, digest in dataset["sources"].items():
            if file_sha256(Path(file)) != digest:
                raise ValueError("raw source changed")
        item = next(
            item for item in load_dataset(data_manifest.get(dataset_id)) if item.item_id == item_id
        )
        prefix_end = next(
            entry["prefix_end"] for entry in dataset["items"] if entry["item_id"] == item_id
        )
        scaler = PrefixStandardizer.fit(item.values[:prefix_end])
        mean, scale = scaler.mean[targets], scaler.scale[targets]
        prefix_z = (item.values[:prefix_end] - scaler.mean) / scaler.scale
        normalized_pool = None
        grouped = defaultdict(list)
        for record in records:
            grouped[(record["mechanism"], record["missing_rate"], record["mask_seed"])].append(
                record
            )
        for (mechanism, rate, seed), current_records in grouped.items():
            realization = mask_time_series(
                item.values,
                MaskingSpec(mechanism, rate, config.block_lengths),
                stable_seed(
                    config.protocol_id, dataset_id, item_id, "validation", mechanism, rate, seed
                ),
                calibration_values=item.values[:prefix_end],
            )
            for current in current_records:
                source_path = root / current["path"]
                if file_sha256(source_path) != current["sha256"]:
                    raise ValueError("current source episode changed")
                cache = output / "predictions" / source_path.name
                if not cache.exists():
                    with np.load(source_path, allow_pickle=False) as episode:
                        ids = episode["candidate_ids"].tolist()
                        context = episode["context"]
                        np.testing.assert_equal(
                            context,
                            realization.values[
                                current["origin"] - config.context_length : current["origin"]
                            ],
                        )
                        normalized = (episode["candidate_values"] - scaler.mean) / scaler.scale
                    base_spec = ForecastSpec(
                        args.model,
                        registry.get(args.model).mode,
                        config.horizon,
                        context_length=config.context_length,
                        target_indices=config.target_indices,
                    )
                    finite = runner.predict(normalized, base_spec).point
                    direct = runner.predict_missing(
                        ((context - scaler.mean) / scaler.scale)[None], base_spec
                    ).point[0]
                    guarded, _ = guarded_direct_forecast(
                        context,
                        targets,
                        direct,
                        finite[ids.index("locf")],
                        joint=args.model == "chronos2",
                    )
                    predictions = [*finite, direct, guarded]
                    names = [f"prefix_input_z_{action}" for action in ids] + [
                        "prefix_input_z_direct",
                        "prefix_input_z_guarded_direct",
                    ]
                    actual_lengths = [config.context_length] * len(names)
                    if normalized_pool is None:
                        local_config = config.model_copy(
                            update={"candidate_ids": ("locf", "linear_interp", "knn_multivariate")}
                        )
                        normalized_pool = FrozenCandidatePool(
                            local_config, dataset_id, prefix_z, current["period"]
                        )
                    normalized_candidates, _, _, _ = normalized_pool.complete(
                        (context - scaler.mean) / scaler.scale, current["origin"], seed
                    )
                    normalized_knn = runner.predict(normalized_candidates[2:3], base_spec).point[0]
                    predictions.append(normalized_knn)
                    names.append("prefix_input_z_standardized_knn")
                    actual_lengths.append(config.context_length)
                    for length in lengths:
                        start = max(prefix_end, current["origin"] - length)
                        extended = realization.values[start : current["origin"]]
                        extended_z = (extended - scaler.mean) / scaler.scale
                        extended_spec = ForecastSpec(
                            args.model,
                            registry.get(args.model).mode,
                            config.horizon,
                            context_length=len(extended),
                            target_indices=config.target_indices,
                        )
                        raw = runner.predict_missing(extended[None], extended_spec).point[0]
                        standardized = runner.predict_missing(
                            extended_z[None], extended_spec
                        ).point[0]
                        locf = carry_forward(extended_z, np.nanmedian(prefix_z, axis=0))
                        completed = runner.predict(locf[None], extended_spec).point[0]
                        guarded, _ = guarded_direct_forecast(
                            extended,
                            targets,
                            standardized,
                            completed,
                            joint=args.model == "chronos2",
                        )
                        predictions.extend([(raw - mean) / scale, standardized, completed, guarded])
                        names.extend(
                            [
                                f"history_{length}_raw_direct",
                                f"history_{length}_prefix_z_direct",
                                f"history_{length}_prefix_z_locf",
                                f"history_{length}_prefix_z_guarded_direct",
                            ]
                        )
                        actual_lengths.extend([len(extended)] * 4)
                    _save_npz(
                        cache,
                        point_z=np.stack(predictions),
                        methods=np.asarray(names),
                        context_lengths=np.asarray(actual_lengths),
                        source_sha256=np.asarray(current["sha256"]),
                    )
                with np.load(cache, allow_pickle=False) as saved:
                    if str(saved["source_sha256"]) != current["sha256"]:
                        raise ValueError("history control belongs to another episode")
                    for name, length, prediction in zip(
                        saved["methods"], saved["context_lengths"], saved["point_z"], strict=True
                    ):
                        error = prediction - truth[source_indices[current["episode_id"]]]
                        rows.append(
                            {
                                key: current[key]
                                for key in ("episode_id", "origin_id", "family_id", "dataset_id")
                            }
                            | {
                                "model_id": args.model,
                                "method": str(name),
                                "actual_context_length": int(length),
                                "mae": float(np.mean(np.abs(error))),
                                "mse": float(np.mean(error**2)),
                            }
                        )
                files.append(
                    {
                        "episode_id": current["episode_id"],
                        "path": str(cache.relative_to(output)),
                        "sha256": file_sha256(cache),
                    }
                )
                print(
                    json.dumps(
                        {"model": args.model, "completed": len(files), "total": len(decisions)}
                    ),
                    flush=True,
                )
    frame = pd.DataFrame(rows)
    keys = ["model_id", "method"]
    family = (
        frame.groupby(keys + ["family_id", "dataset_id"])[["mae", "mse"]]
        .mean()
        .groupby(level=keys + ["family_id"])
        .mean()
        .reset_index()
    )
    summary = family.groupby(keys)[["mae", "mse"]].mean().reset_index()
    frame.to_parquet(output / "episode_results.parquet", index=False)
    family.to_csv(output / "family_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "evidence_role": "development",
            "identity": identity,
            "episodes": files,
            "resources_this_execution": runner.resource_metrics(),
            "role": "same-history and input-standardization controls; improvements here are not attributed to imputer-selection learning",
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
