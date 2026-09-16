"""Fit native-prefix L192 imputer controls and prepare the fixed expanded development panel."""

import argparse
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources, timestamp
from prepare_followup_inputs import fit_deep
from prepare_native_confirmation import ACTIONS, DEEP_PARAMS, MECHANISMS, complete_candidates

from tsfm_fais.contracts import SeriesBatch, TimeSeriesItem
from tsfm_fais.data.masking import MaskingSpec
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner
from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.stage_execution import _training_batch
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def smoke_selection(plan):
    first = plan["datasets"][0]
    last = next(d for d in plan["datasets"] if d["dataset_id"].startswith("US_Term"))
    selected = []
    for dataset in (first, last):
        selected.extend(
            [
                r
                for r in plan["cases"]
                if (r["cohort"], r["dataset_id"]) == (dataset["cohort"], dataset["dataset_id"])
            ][:2]
        )
    return (first, last), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed L192 imputation inputs")
    plan_root = ROOT / "artifacts/iclr27-r25/long-plan-v001"
    plan = read_json(plan_root / "manifest.json")
    identity = {
        "files": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in (
                Path(__file__),
                plan_root / "manifest.json",
                ROOT / "docs/iclr2027/R25_LONG_POOL_PROTOCOL.md",
                ROOT / "src/tsfm_fais/stage_execution.py",
                ROOT / "scripts/prepare_followup_inputs.py",
                ROOT / "src/tsfm_fais/imputers/motm.py",
            )
        },
        "context_length": 192,
        "horizon": 96,
        "deep_params": DEEP_PARAMS,
        "maximum_training_windows": 64,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial long-pool preparation definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    groups, cases = smoke_selection(plan) if args.smoke else (plan["datasets"], plan["cases"])
    sources = native_sources()
    source_map = {(r["cohort"], r["dataset_id"], r["item_id"]): r for r in sources}
    torch.set_num_threads(1)
    runner = CandidateRunner()
    motm = MOTMReference(
        ROOT / "artifacts/iclr27-r5/motm-reference-v001",
        ROOT / "artifacts/iclr27-r5/motm-runtime-v001",
        device="cuda",
        ridge=0.5,
        batch_size=32,
    )
    records, fits = [], []
    for dataset in groups:
        peers = [
            source_map[(dataset["cohort"], dataset["dataset_id"], name)]
            for name in dataset["fit_items"]
        ]
        items = [
            TimeSeriesItem(
                s["item_id"],
                s["values"],
                tuple(s["columns"]),
                pd.Timestamp(s["start"]),
                s["frequency"],
            )
            for s in peers
        ]
        period = peers[0]["period"]
        batch = _training_batch(
            items,
            192,
            96,
            64,
            dataset_id=dataset["dataset_id"],
            masking_specs=[
                MaskingSpec(name, rate, (6, 12, 24, 48))
                for name in MECHANISMS
                for rate in dataset["training_missing_rates"]
            ],
            configured_seeds=dataset["training_mask_seeds"],
            fit_fraction=0.2,
            training_stride=24,
        )
        peer_map = {s["item_id"]: s for s in peers}
        cutoffs = []
        for identifier in batch.item_ids:
            item, suffix = identifier.rsplit("@", 1)
            start = int(suffix.split("|", 1)[0])
            if start + 192 > peer_map[item]["prefix_end"]:
                raise ValueError("a new imputer window extends beyond its original prefix")
            cutoffs.append(timestamp(peer_map[item], start + 192))
        selected = [
            r
            for r in cases
            if (r["cohort"], r["dataset_id"]) == (dataset["cohort"], dataset["dataset_id"])
        ]
        if max(cutoffs) > min(
            timestamp(source_map[(r["cohort"], r["dataset_id"], r["item_id"])], r["origin"])
            for r in selected
        ):
            raise ValueError(
                "a dataset-level imputer uses information later than a prediction origin"
            )
        directory = output / "imputers" / dataset["cohort"] / dataset["dataset_id"]
        artifacts, failures, deep_records = fit_deep(
            runner, batch, directory, identity_sha, period, dataset["dataset_id"]
        )
        fits.append(
            {
                **dataset,
                "training_batch_path": str((directory / "training_batch.npz").relative_to(output)),
                "training_batch_sha256": file_sha256(directory / "training_batch.npz"),
                "deep_fits": deep_records,
                "latest_training_end": str(max(cutoffs)),
                "failure_count": len(failures),
            }
        )
        for source in peers:
            current = [r for r in selected if r["item_id"] == source["item_id"]]
            if not current:
                continue
            prefix = source["values"][: source["prefix_end"]]
            local = dict(artifacts)
            params = {"seasonal_lag": {"period": max(2, period)}}
            fit_start = perf_counter()
            classical = []
            for action in ACTIONS:
                if (
                    action not in local
                    and action not in failures
                    and (
                        action == "seasonal_lag"
                        or DEFAULT_REGISTRY.get_spec(action).fit_scope == "dataset"
                    )
                ):
                    local[action] = runner.fit(
                        action,
                        SeriesBatch(
                            prefix[None], np.isfinite(prefix[None]), metadata={"period": period}
                        ),
                        {"period": period},
                        params=params.get(action),
                    )
                    classical.append(action)
            classical_seconds = perf_counter() - fit_start
            for row in current:
                path = output / "cases" / f"{row['case_id']}.npz"
                marker = path.with_suffix(".json")
                if marker.exists():
                    saved = read_json(marker)
                    if (
                        saved["identity_sha256"] != identity_sha
                        or file_sha256(path) != saved["sha256"]
                    ):
                        raise ValueError("a prepared L192 case changed")
                    records.append(saved)
                    continue
                started = perf_counter()
                if file_sha256(plan_root / row["context_path"]) != row["context_sha256"]:
                    raise ValueError("a frozen original long context changed")
                with np.load(plan_root / row["context_path"], allow_pickle=False) as saved:
                    context, mean, scale, defaults = (
                        saved["context"],
                        saved["mean"],
                        saved["scale"],
                        saved["defaults"],
                    )
                case_artifacts = dict(local)
                seasonal = dict(local["seasonal_lag"])
                seasonal["profiles"] = np.roll(
                    seasonal["profiles"], -(row["origin"] - 192) % seasonal["period"], axis=1
                )
                case_artifacts["seasonal_lag"] = seasonal
                result = runner.run_many(
                    ACTIONS,
                    SeriesBatch(
                        context[None], np.isfinite(context[None]), metadata={"period": period}
                    ),
                    case_artifacts,
                    seed=int(hashlib.sha256(row["episode_id"].encode()).hexdigest()[:8], 16),
                    params=params,
                    artifact_failures=failures,
                )
                completed, coverage, statuses = complete_candidates(context, result, defaults)
                motm_started = perf_counter()
                extra, diagnostics = motm.impute(context, completed[list(ACTIONS).index("locf")])
                motm_seconds = perf_counter() - motm_started
                values = np.concatenate([completed, extra[None]])
                for value in values:
                    if not np.isfinite(value).all():
                        raise ValueError("a long imputation candidate remains nonfinite")
                    np.testing.assert_array_equal(
                        value[np.isfinite(context)], context[np.isfinite(context)]
                    )
                _save_npz(
                    path,
                    context=context,
                    candidate_values=values,
                    candidate_ids=np.asarray([*ACTIONS, "motm_reference"]),
                    mean=mean,
                    scale=scale,
                    native_coverage=coverage,
                    identity_sha256=np.asarray(identity_sha),
                )
                record = {
                    **row,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "identity_sha256": identity_sha,
                    "statuses": statuses,
                    "native_coverage": coverage.tolist(),
                    "motm": diagnostics,
                    "motm_seconds": motm_seconds,
                    "seconds": perf_counter() - started,
                    "classical_fit_names": classical,
                    "classical_fit_seconds_series_shared": classical_seconds,
                }
                _write_json(marker, record)
                records.append(record)
            print(f"prepared {dataset['dataset_id']} {source['item_id']}", flush=True)
        del artifacts, local, batch
        gc.collect()
        torch.cuda.empty_cache()
    motm.verify_frozen()
    if not args.smoke and len(records) != 301:
        raise ValueError("the complete long-pool case population changed")
    _write_json(
        output / ("smoke.json" if args.smoke else "manifest.json"),
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "cases": records,
            "fits": fits,
            "prediction_future_values_read": False,
            "new_repair_models_trained": 0,
        },
    )


if __name__ == "__main__":
    main()
