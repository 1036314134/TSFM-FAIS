"""Prepare only registered additional histories using frozen prefix imputers."""

import argparse
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch
from latent_source_inputs import ROOT, read_json
from prepare_native_confirmation import complete_candidates

from tsfm_fais.contracts import SeriesBatch
from tsfm_fais.data import MaskingSpec, load_dataset, load_manifest, mask_time_series, stable_seed
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner
from tsfm_fais.utility_experiment import (
    _load_frozen_imputers,
    _save_npz,
    _write_json,
    file_sha256,
    load_utility_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/iclr27-r3/development_expanded.yaml"
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "artifacts/iclr27-r9/source-expansion-inventory-v001/manifest.json",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "docs/iclr2027/R9_SOURCE_EXPANSION_PROTOCOL.md"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_utility_config(args.config)
    inventory = read_json(args.inventory)
    source = read_json(config.output_root / "episodes_manifest.json")
    if inventory["status"] != "completed" or inventory["source_sha256"] != file_sha256(
        config.output_root / "episodes_manifest.json"
    ):
        raise ValueError("complete the unchanged source inventory first")
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed supplemental inputs")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "inventory_sha256": file_sha256(args.inventory),
        "protocol_sha256": file_sha256(args.protocol),
        "config_sha256": file_sha256(args.config),
        "completion_helper_sha256": file_sha256(ROOT / "scripts/prepare_native_confirmation.py"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial supplemental input definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    data, runner = load_manifest(config.data_manifest), CandidateRunner()
    records, replays = [], []
    for row in inventory["items"]:
        if not row["added_train"]:
            continue
        dataset_id = row["dataset_id"]
        for name, digest in row["sources"].items():
            if file_sha256(Path(name)) != digest:
                raise ValueError("an inventoried raw source changed")
        spec = data.get(dataset_id)
        item = next(
            item
            for item in load_dataset(spec)[: config.max_items]
            if item.item_id == row["item_id"]
        )
        if len(item.values) != row["length"]:
            raise ValueError("an inventoried source length changed")
        artifacts, failures, fit_record = _load_frozen_imputers(config, dataset_id)
        old_dataset = next(part for part in source["datasets"] if part["dataset_id"] == dataset_id)
        if fit_record["manifest_sha256"] != old_dataset["fit_artifacts"]["manifest_sha256"]:
            raise ValueError("a frozen imputer artifact changed")
        prefix = item.values[: row["prefix_end"]]
        defaults = np.nanmedian(prefix, axis=0)
        params = {"seasonal_lag": {"period": max(2, spec.period)}}
        batch = SeriesBatch(
            prefix[None], np.isfinite(prefix[None]), metadata={"period": spec.period}
        )
        for candidate in config.candidate_ids:
            if (
                candidate not in artifacts
                and candidate not in failures
                and (
                    candidate == "seasonal_lag"
                    or DEFAULT_REGISTRY.get_spec(candidate).fit_scope == "dataset"
                )
            ):
                artifacts[candidate] = runner.fit(
                    candidate, batch, {"period": spec.period}, params=params.get(candidate)
                )

        def complete(
            context,
            origin,
            artifacts=artifacts,
            spec=spec,
            params=params,
            failures=failures,
            defaults=defaults,
        ):
            local = dict(artifacts)
            if "seasonal_lag" in local:
                seasonal = dict(local["seasonal_lag"])
                seasonal["profiles"] = np.roll(
                    seasonal["profiles"], -(origin - 96) % seasonal["period"], axis=1
                )
                local["seasonal_lag"] = seasonal
            batch = SeriesBatch(
                context[None], np.isfinite(context[None]), metadata={"period": spec.period}
            )
            results = runner.run_many(
                config.candidate_ids,
                batch,
                local,
                seed=stable_seed(6101, origin),
                params=params,
                artifact_failures=failures,
            )
            return complete_candidates(context, results, defaults, config.candidate_ids)

        for mechanism in config.mechanisms:
            for rate in config.missing_rates:
                realization = mask_time_series(
                    item.values,
                    MaskingSpec(mechanism, rate, config.block_lengths),
                    stable_seed(
                        config.protocol_id, dataset_id, item.item_id, "train", mechanism, rate, 6101
                    ),
                    calibration_values=prefix,
                )
                if mechanism == config.mechanisms[0] and rate == config.missing_rates[0]:
                    old = next(
                        part
                        for part in source["episodes"]
                        if part["dataset_id"] == dataset_id
                        and part["item_id"] == item.item_id
                        and part["origin"] == row["old_train"][0]
                        and part["split"] == "train"
                        and part["mechanism"] == mechanism
                        and part["missing_rate"] == rate
                        and part["mask_seed"] == 6101
                    )
                    old_path = config.output_root / old["path"]
                    if file_sha256(old_path) != old["sha256"]:
                        raise ValueError("a prespecified old input changed")
                    context = realization.values[old["origin"] - 96 : old["origin"]].copy()
                    with np.load(old_path, allow_pickle=False) as saved:
                        np.testing.assert_array_equal(context, saved["context"])
                        values, coverage, _ = complete(context, old["origin"])
                        np.testing.assert_allclose(
                            values, saved["candidate_values"], rtol=1e-6, atol=1e-6
                        )
                        np.testing.assert_array_equal(coverage, saved["native_coverage"])
                        replays.append(
                            {
                                "dataset_id": dataset_id,
                                "episode_id": old["episode_id"],
                                "maximum_candidate_difference": float(
                                    abs(values - saved["candidate_values"]).max()
                                ),
                            }
                        )
                for origin in row["added_train"]:
                    origin_id = f"{dataset_id}|{item.item_id}|{origin}"
                    episode_id = f"{origin_id}|train|{mechanism}|{rate}|6101"
                    key = hashlib.sha256(episode_id.encode()).hexdigest()[:24]
                    path, record_path = (
                        output / "episodes" / f"{key}.npz",
                        output / "records" / f"{key}.json",
                    )
                    context = realization.values[origin - 96 : origin].copy()
                    clean, future = (
                        item.values[origin - 96 : origin].copy(),
                        item.values[origin : origin + 96].copy(),
                    )
                    if record_path.exists():
                        record = read_json(record_path)
                        if (
                            record["identity_sha256"] != identity_sha
                            or file_sha256(path) != record["sha256"]
                        ):
                            raise ValueError("a resumed supplemental input changed")
                        with np.load(path, allow_pickle=False) as saved:
                            for name, value in (
                                ("context", context),
                                ("clean_context", clean),
                                ("future", future),
                            ):
                                np.testing.assert_array_equal(saved[name], value)
                    else:
                        values, coverage, statuses = complete(context, origin)
                        _save_npz(
                            path,
                            context=context,
                            clean_context=clean,
                            future=future,
                            candidate_values=values,
                            candidate_ids=np.asarray(config.candidate_ids),
                            native_coverage=coverage,
                        )
                        record = {
                            "episode_id": episode_id,
                            "origin_id": origin_id,
                            "dataset_id": dataset_id,
                            "family_id": spec.family_id,
                            "item_id": item.item_id,
                            "origin": origin,
                            "split": "train",
                            "mechanism": mechanism,
                            "missing_rate": rate,
                            "mask_seed": 6101,
                            "period": spec.period,
                            "path": str(path.relative_to(output)),
                            "sha256": file_sha256(path),
                            "identity_sha256": identity_sha,
                            "candidate_status": statuses,
                        }
                        _write_json(record_path, record)
                    records.append(record)
        print(f"{dataset_id}: supplemental candidates complete", flush=True)
        del artifacts, item, realization, complete
        gc.collect()
    if len(records) != 2826 or len({row["origin_id"] for row in records}) != 157:
        raise ValueError("the registered source supplement is incomplete")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "episodes": records,
            "original_input_replays": replays,
            "new_neural_imputer_fits": 0,
        },
    )


if __name__ == "__main__":
    main()
