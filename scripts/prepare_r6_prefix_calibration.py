"""Build and check historical teacher-calibration inputs without evaluation futures."""

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
from freeze_followup_cohort import evenly_spaced  # noqa: E402
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
from tsfm_fais.imputers.registry import DEFAULT_REGISTRY  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.stage_execution import _training_batch  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def calibration_origins(prefix, fit_end):
    eligible = [
        origin
        for origin in range(fit_end + 96, len(prefix) + 1, 96)
        if np.isfinite(prefix[origin - 96 : origin]).all()
    ]
    selected = evenly_spaced(eligible, 4)
    return selected if len(selected) >= 2 else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort-root", "prepared-root", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed calibration inputs")
    cohort_path = args.cohort_root / "manifest.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    scaler_path = args.prepared_root / "standardizers.json"
    scalers = {
        (row["dataset_id"], row["item_id"]): row
        for row in json.loads(scaler_path.read_text(encoding="utf-8"))
    }
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "cohort_sha256": file_sha256(cohort_path),
        "protocol_sha256": file_sha256(args.protocol),
        "standardizers_sha256": file_sha256(scaler_path),
        "inner_fraction": 0.6,
        "context_length": 96,
        "horizon": 96,
        "maximum_histories": 4,
        "minimum_histories": 2,
        "mask_rates": [0.1, 0.3, 0.5],
        "mask_seed": 9103,
        "candidate_ids": list(ACTIONS),
        "deep_params": DEEP_PARAMS,
        "runtime_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/prepare_followup_inputs.py",
                "scripts/prepare_native_confirmation.py",
                "src/tsfm_fais/stage_execution.py",
                "src/tsfm_fais/data/masking.py",
                "src/tsfm_fais/imputers/runner.py",
                "src/tsfm_fais/imputers/pypots.py",
                "src/tsfm_fais/imputers/structured.py",
                "src/tsfm_fais/imputers/classical.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial calibration input identity changed")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    groups = defaultdict(list)
    for source in cohort["sources"]:
        groups[source["dataset_id"]].append(source)
    runner, histories, records, fitting, support = CandidateRunner(), [], [], [], []
    for dataset, sources in groups.items():
        items, inner_ends = [], {}
        for source in sources:
            source_path = Path(source["path"])
            if file_sha256(source_path) != source["sha256"]:
                raise ValueError("an original trajectory changed")
            # Only the original fitting prefix is materialized; later values are not inputs.
            prefix = np.load(source_path, mmap_mode="r")[: source["prefix_end"]].copy(order="K")
            inner_ends[source["item_id"]] = max(96, int(0.6 * len(prefix)))
            scaler = PrefixStandardizer.fit(prefix)
            expected = scalers[(dataset, source["item_id"])]
            np.testing.assert_array_equal(scaler.mean, expected["mean"])
            np.testing.assert_array_equal(scaler.scale, expected["scale"])
            items.append(
                TimeSeriesItem(
                    source["item_id"],
                    prefix,
                    tuple(source["columns"]),
                    pd.Timestamp(source["start"]),
                    source["frequency"],
                )
            )
        period = sources[0]["period"]
        batch = _training_batch(
            items,
            96,
            96,
            64,
            dataset_id=dataset,
            masking_specs=[
                MaskingSpec(m, r) for m in MECHANISMS for r in (0.1, 0.2, 0.3, 0.4, 0.5)
            ],
            configured_seeds=[1101, 1102, 1103],
            fit_fraction=0.6,
            training_stride=24,
        )
        item_map = {item.item_id: item for item in items}
        for index, identifier in enumerate(batch.item_ids):
            item_id, offset = identifier.rsplit("@", 1)
            start = int(offset.split("|", 1)[0])
            if start + 96 > inner_ends[item_id]:
                raise ValueError("a calibration imputer saw a later calibration history")
            observed = batch.observed_mask[index]
            np.testing.assert_array_equal(
                batch.values[index][observed],
                item_map[item_id].values[start : start + 96][observed],
            )
        directory = output / "imputers" / dataset
        artifacts, failures, fits = fit_deep(
            runner, batch, directory, identity_sha, period, dataset
        )
        if failures or any(row["status"] != "fitted" for row in fits):
            raise ValueError("a calibration neural imputer failed; preserve its fitting record")
        fitting.append(
            {
                "dataset_id": dataset,
                "imputers": fits,
                "training_sha256": file_sha256(directory / "training_batch.npz"),
            }
        )
        for item in items:
            prefix, fit_end = item.values, inner_ends[item.item_id]
            origins = calibration_origins(prefix, fit_end)
            support.append(
                {
                    "dataset_id": dataset,
                    "item_id": item.item_id,
                    "fit_end": fit_end,
                    "prefix_end": len(prefix),
                    "origins": origins,
                    "uses_source_fallback": not origins,
                }
            )
            if not origins:
                continue
            inner = prefix[:fit_end]
            defaults = np.nanmedian(inner, axis=0)
            local, params = dict(artifacts), {"seasonal_lag": {"period": max(2, period)}}
            prefix_batch = SeriesBatch(
                inner[None], np.isfinite(inner[None]), metadata={"period": period}
            )
            for name in ACTIONS:
                if name not in local and (
                    name == "seasonal_lag" or DEFAULT_REGISTRY.get_spec(name).fit_scope == "dataset"
                ):
                    local[name] = runner.fit(
                        name, prefix_batch, {"period": period}, params=params.get(name)
                    )
            for origin in origins:
                key = hashlib.sha256(f"{dataset}:{item.item_id}:{origin}".encode()).hexdigest()[:24]
                path = output / "histories" / f"{key}.npz"
                clean = prefix[origin - 96 : origin].copy()
                if origin - 96 < fit_end or origin > len(prefix) or not np.isfinite(clean).all():
                    raise ValueError("teacher history violates the registered boundary")
                if not path.exists():
                    _save_npz(path, clean=clean, identity_sha256=np.asarray(identity_sha))
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["clean"], clean)
                    if str(saved["identity_sha256"]) != identity_sha:
                        raise ValueError("teacher history identity changed")
                histories.append(
                    {
                        "history_id": key,
                        "dataset_id": dataset,
                        "item_id": item.item_id,
                        "origin": origin,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
            for mechanism in MECHANISMS:
                for rate in identity["mask_rates"]:
                    masked = mask_time_series(
                        prefix, MaskingSpec(mechanism, rate), 9103, calibration_values=inner
                    )
                    for origin in origins:
                        history_id = hashlib.sha256(
                            f"{dataset}:{item.item_id}:{origin}".encode()
                        ).hexdigest()[:24]
                        key = hashlib.sha256(
                            f"{history_id}:{mechanism}:{rate}:9103".encode()
                        ).hexdigest()[:24]
                        path = output / "episodes" / f"{key}.npz"
                        marker = path.with_suffix(".json")
                        context = masked.values[origin - 96 : origin].copy()
                        if not marker.exists():
                            case_artifacts = dict(local)
                            if "seasonal_lag" in case_artifacts:
                                seasonal = dict(case_artifacts["seasonal_lag"])
                                seasonal["profiles"] = np.roll(
                                    seasonal["profiles"],
                                    -(origin - 96) % seasonal["period"],
                                    axis=1,
                                )
                                case_artifacts["seasonal_lag"] = seasonal
                            results = runner.run_many(
                                ACTIONS,
                                SeriesBatch(
                                    context[None],
                                    np.isfinite(context[None]),
                                    metadata={"period": period},
                                ),
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
                                candidate_values=completed,
                                candidate_ids=np.asarray(ACTIONS),
                                native_coverage=coverage,
                                identity_sha256=np.asarray(identity_sha),
                            )
                            _write_json(
                                marker,
                                {
                                    "episode_id": key,
                                    "history_id": history_id,
                                    "dataset_id": dataset,
                                    "item_id": item.item_id,
                                    "origin": origin,
                                    "mechanism": mechanism,
                                    "missing_rate": rate,
                                    "seed": 9103,
                                    "path": str(path.relative_to(output)),
                                    "sha256": file_sha256(path),
                                    "identity_sha256": identity_sha,
                                    "candidate_statuses": statuses,
                                },
                            )
                        record = json.loads(marker.read_text(encoding="utf-8"))
                        if (
                            record["identity_sha256"] != identity_sha
                            or file_sha256(path) != record["sha256"]
                        ):
                            raise ValueError("a saved calibration input changed")
                        with np.load(path, allow_pickle=False) as saved:
                            np.testing.assert_array_equal(saved["context"], context)
                            observed = np.isfinite(context)
                            for completion in saved["candidate_values"]:
                                if not np.isfinite(completion).all():
                                    raise ValueError("a calibration completion is nonfinite")
                                np.testing.assert_array_equal(
                                    completion[observed], prefix[origin - 96 : origin][observed]
                                )
                        records.append(record)
            print(
                json.dumps(
                    {"dataset": dataset, "item": item.item_id, "completed_inputs": len(records)}
                ),
                flush=True,
            )
        del artifacts, local, batch, items, item_map, prefix_batch
        torch.cuda.empty_cache()
    if (
        len(histories) != 43
        or len(records) != 774
        or len({row["episode_id"] for row in records}) != 774
    ):
        raise ValueError("the preregistered calibration coverage changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "histories": histories,
            "episodes": records,
            "support": support,
            "fitting": fitting,
            "checked_training_boundaries": True,
            "checked_original_observations": True,
            "evaluation_future_arrays_read": False,
            "forecaster_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
