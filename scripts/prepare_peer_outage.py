"""Prepare shared peer information and stronger imputation controls without forecast errors."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from peer_outage_core import (
    COHORT,
    PEERS,
    ROOT,
    PrefixRegression,
    augmented,
    case_context,
    population,
    sources,
)
from prepare_native_confirmation import ACTIONS, DEEP_PARAMS, MECHANISMS, complete_candidates

from tsfm_fais.contracts import SeriesBatch, TimeSeriesItem
from tsfm_fais.data.masking import MaskingSpec
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner
from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.stage_execution import _training_batch
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def fit_deep(runner, batch, params, directory, identity):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "training_batch.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["values"], batch.values)
            np.testing.assert_array_equal(saved["observed"], batch.observed_mask)
    else:
        _save_npz(
            path,
            values=batch.values,
            observed=batch.observed_mask,
            window_ids=np.asarray(batch.item_ids),
        )
    artifacts, records = {}, []
    for name in ("saits", "timemixerpp"):
        marker, folder = directory / f"{name}.json", directory / name
        adapter = DEFAULT_REGISTRY.create(name, **params)
        if marker.exists():
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record["identity"] != identity:
                raise ValueError("partial neural fit identity differs")
            if record["status"] != "fitted":
                raise ValueError("preserve and diagnose a failed baseline fit")
            for entry in record["files"]:
                if file_sha256(folder / entry["path"]) != entry["sha256"]:
                    raise ValueError("a baseline checkpoint changed")
            artifacts[name] = adapter.load_artifact(folder)
        else:
            started = perf_counter()
            try:
                artifact = runner.fit(
                    name, batch, {"period": 24, "dataset_id": "beijing_peer"}, params=params
                )
                adapter.save_artifact(artifact, folder)
            except Exception as error:
                _write_json(
                    marker,
                    {
                        "status": "failed",
                        "identity": identity,
                        "reason": f"{type(error).__name__}: {error}",
                    },
                )
                raise
            artifacts[name] = artifact
            record = {
                "status": "fitted",
                "identity": identity,
                "params": params,
                "training_windows": batch.shape[0],
                "seconds": perf_counter() - started,
                "files": [
                    {"path": str(p.relative_to(folder)), "sha256": file_sha256(p)}
                    for p in sorted(folder.rglob("*"))
                    if p.is_file()
                ],
            }
            _write_json(marker, record)
        records.append({"candidate_id": name, "marker": str(marker), "sha256": file_sha256(marker)})
    return artifacts, records, file_sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed peer inputs")
    records, peer_map = sources()
    rows, rejected = population(records)
    augmented_sources = {station: augmented(station, records, peer_map) for station in records}
    full = {station: value[0] for station, value in augmented_sources.items()}
    params = {**DEEP_PARAMS, "epochs": 1 if args.smoke else 50}
    maximum = 16 if args.smoke else 512
    if args.smoke:
        selected = []
        for panel in ("natural_outage_h24", "synthetic_outage_h24", "legacy_native_h96"):
            selected.extend([r for r in rows if r["panel"] == panel][:2])
        natural = [r for r in rows if r["panel"] == "natural_outage_h24"]
        selected.append(max(natural, key=lambda r: r["outage_age"]))
        selected.append(
            max(
                rows,
                key=lambda r: (
                    ~np.isfinite(case_context(full[r["station"]], r)[:, 11:]).any(0)
                ).sum(),
            )
        )
        selected = list({r["case_id"]: r for r in selected}.values())
        rows = selected
    identity = {
        "files": {
            str(p): file_sha256(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/peer_outage_core.py",
                COHORT,
                PEERS,
                ROOT / "docs/iclr2027/R30_PEER_OUTAGE_PROTOCOL.md",
                ROOT / "scripts/prepare_native_confirmation.py",
                ROOT / "src/tsfm_fais/stage_execution.py",
            )
        },
        "params": params,
        "training_window_cap": maximum,
        "case_ids": [r["case_id"] for r in rows],
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and json.loads(
        (output / "identity.json").read_text(encoding="utf-8")
    ) != identity:
        raise ValueError("partial peer input definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    _write_json(
        output / "case_plan.json",
        {"cases": rows, "rejected_origins": rejected, "future_prediction_errors_read": False},
    )
    torch.set_num_threads(1)
    items = [
        TimeSeriesItem(
            station,
            full[station],
            tuple(records[station]["columns"] + [f"peer_{i}" for i in range(6)]),
            pd.Timestamp(records[station]["start"]),
            records[station]["frequency"],
        )
        for station in sorted(records)
    ]
    batch = _training_batch(
        items,
        192,
        96,
        maximum,
        dataset_id="beijing_peer",
        masking_specs=[
            MaskingSpec(name, rate, (6, 12, 24, 48))
            for name in MECHANISMS
            for rate in (0.1, 0.2, 0.3, 0.4, 0.5)
        ],
        configured_seeds=[1101, 1102, 1103],
        fit_fraction=0.2,
        training_stride=24,
    )
    for identifier in batch.item_ids:
        station, tail = identifier.rsplit("@", 1)
        start = int(tail.split("|", 1)[0])
        if start + 192 > records[station]["prefix_end"]:
            raise ValueError("a neural training window crosses its prefix")
    runner = CandidateRunner()
    started = perf_counter()
    deep, deep_records, training_sha = fit_deep(
        runner, batch, params, output / "imputers", identity_sha
    )
    motm = MOTMReference(
        ROOT / "artifacts/iclr27-r5/motm-reference-v001",
        ROOT / "artifacts/iclr27-r5/motm-runtime-v001",
        device="cuda",
        ridge=0.5,
        batch_size=32,
    )
    completed_rows, fits = [], []
    for station, source in sorted(records.items()):
        selected = [r for r in rows if r["station"] == station]
        if not selected:
            continue
        prefix = full[station][: source["prefix_end"]]
        stats = PrefixRegression(prefix)
        defaults = np.nanmedian(prefix, axis=0)
        if not np.isfinite(stats.mean).all() or not np.isfinite(defaults).all():
            raise ValueError("a registered prefix column has no support")
        local = dict(deep)
        impute_params = {"seasonal_lag": {"period": 24}}
        for action in ACTIONS:
            if action not in local and (
                action == "seasonal_lag" or DEFAULT_REGISTRY.get_spec(action).fit_scope == "dataset"
            ):
                local[action] = runner.fit(
                    action,
                    SeriesBatch(prefix[None], np.isfinite(prefix[None]), metadata={"period": 24}),
                    {"period": 24},
                    params=impute_params.get(action),
                )
        for row in selected:
            x = case_context(full[station], row)
            artifacts = dict(local)
            seasonal = dict(local["seasonal_lag"])
            seasonal["profiles"] = np.roll(
                seasonal["profiles"], -(row["origin"] - 192) % 24, axis=1
            )
            artifacts["seasonal_lag"] = seasonal
            outputs = runner.run_many(
                ACTIONS,
                SeriesBatch(x[None], np.isfinite(x[None]), metadata={"period": 24}),
                artifacts,
                seed=int(row["case_id"][:8], 16),
                params=impute_params,
            )
            candidates, coverage, status = complete_candidates(x, outputs, defaults)
            extra, motm_info = motm.impute(x, candidates[list(ACTIONS).index("locf")])
            candidates = np.concatenate([candidates, extra[None]])
            stats_targets = stats.targets(
                x, candidates[list(ACTIONS).index("linear_interp"), :, :2]
            )
            keep = np.r_[np.ones(11, bool), np.isfinite(x[:, 11:]).any(0)]
            future_models = stats.future_models(keep)
            for values in candidates:
                np.testing.assert_array_equal(values[np.isfinite(x)], x[np.isfinite(x)])
                if not np.isfinite(values).all():
                    raise ValueError("a baseline is not finite")
            path = output / "cases" / f"{row['case_id']}.npz"
            _save_npz(
                path,
                context=x,
                candidates=candidates,
                actions=np.asarray([*ACTIONS, "motm_reference"]),
                mean=stats.mean,
                scale=stats.scale,
                defaults=defaults,
                keep=keep,
                stat_names=np.asarray(list(stats_targets)),
                stat_targets=np.stack(list(stats_targets.values())),
                future_models=np.asarray(json.dumps(future_models)),
            )
            completed_rows.append(
                {
                    **row,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "statuses": status,
                    "native_coverage": coverage.tolist(),
                    "motm": motm_info,
                    "peer_columns": augmented_sources[station][1],
                    "dropped_peer_columns": np.flatnonzero(~keep).tolist(),
                }
            )
            _write_json(
                output / "progress.json", {"prepared": len(completed_rows), "total": len(rows)}
            )
        fit_path = output / "regressions" / f"{station}.json"
        stats.save(fit_path)
        fits.append(
            {
                "station": station,
                "path": str(fit_path.relative_to(output)),
                "sha256": file_sha256(fit_path),
                "prefix_end": source["prefix_end"],
                "subset_models": len(stats.cache),
            }
        )
        print(json.dumps({"prepared_station": station, "cases": len(selected)}), flush=True)
    motm.verify_frozen()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "smoke": args.smoke,
            "identity": identity,
            "cases": completed_rows,
            "deep_fits": deep_records,
            "training_batch_sha256": training_sha,
            "regression_fits": fits,
            "source_sha256": {s["path"]: s["sha256"] for s in records.values()},
            "rejected_origins": rejected,
            "future_prediction_errors_read": False,
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
