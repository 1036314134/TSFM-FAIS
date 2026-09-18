"""Cross-fit source imputers before the historical forecasting calibration interval."""

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401
import torch
from cooutage_calibration_core import (
    ROOT,
    calibration_context,
    calibration_population,
    calibration_sources,
    observable_features,
)
from peer_outage_core import PrefixRegression, augmented
from prepare_native_confirmation import ACTIONS, DEEP_PARAMS, MECHANISMS, complete_candidates
from prepare_peer_outage import fit_deep

from tsfm_fais.contracts import SeriesBatch, TimeSeriesItem
from tsfm_fais.data.masking import MaskingSpec
from tsfm_fais.imputers import DEFAULT_REGISTRY, CandidateRunner
from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.stage_execution import _training_batch
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed calibration inputs")
    started = perf_counter()
    records, peer_map, selection = calibration_sources()
    rows, eligibility = calibration_population(records)
    if len(rows) != 864:
        raise ValueError("the registered calibration count changed")
    if args.smoke:
        rows = [rows[i] for i in (0, 4, 8, 216, 220, 224)]
    augmented_sources = {s: augmented(s, records, peer_map) for s in records}
    full = {s: value[0] for s, value in augmented_sources.items()}
    params = {**DEEP_PARAMS, "epochs": 1 if args.smoke else 50}
    maximum = 16 if args.smoke else 512
    identity = {
        "files": {
            str(p): file_sha256(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/forecast_calibration_core.py",
                ROOT / "scripts/cooutage_calibration_core.py",
                ROOT / "scripts/prepare_peer_outage.py",
                ROOT / "scripts/peer_outage_core.py",
                ROOT / "scripts/prepare_native_confirmation.py",
                ROOT / "src/tsfm_fais/stage_execution.py",
                ROOT / "docs/iclr2027/R32_COOUTAGE_CALIBRATION_PROTOCOL.md",
            )
        },
        "params": params,
        "training_window_cap": maximum,
        "case_ids": [r["case_id"] for r in rows],
        "peer_selection": selection,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists():
        if json.loads((output / "identity.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("partial calibration input definitions changed")
    _write_json(output / "identity.json", identity)
    _write_json(output / "case_plan.json", {"cases": rows, "eligibility": eligibility})
    torch.set_num_threads(1)
    items = [
        TimeSeriesItem(
            s,
            full[s],
            tuple(records[s]["columns"] + [f"peer_{i}" for i in range(6)]),
            pd.Timestamp(records[s]["start"]),
            records[s]["frequency"],
        )
        for s in sorted(records)
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
        fit_fraction=0.5,
        training_stride=24,
    )
    for identifier in batch.item_ids:
        station, tail = identifier.rsplit("@", 1)
        if int(tail.split("|", 1)[0]) + 192 > records[station]["prefix_end"]:
            raise ValueError("source neural fitting crosses its earlier prefix")
    runner = CandidateRunner()
    parent = (
        ROOT
        / "artifacts/iclr27-r31"
        / ("calibration-smoke-inputs-v001" if args.smoke else "calibration-inputs-v001")
    )
    if not (parent / "manifest.json").is_file():
        raise ValueError("complete and preserve the registered R31 base fits first")
    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    parent_cases = {r["case_id"]: r for r in parent_manifest["cases"]}
    for name in ("saits", "timemixerpp"):
        if not (parent / "imputers" / f"{name}.json").is_file():
            raise ValueError("R32 reuses base fits and must not silently retrain")
    deep, deep_records, training_sha = fit_deep(
        runner, batch, params, parent / "imputers", file_sha256(parent / "identity.json")
    )
    (output / "imputers").mkdir(exist_ok=True)
    saved_batch = output / "imputers/training_batch.npz"
    if saved_batch.exists():
        if file_sha256(saved_batch) != training_sha:
            raise ValueError("partial R32 source batch differs")
    else:
        shutil.copy2(parent / "imputers/training_batch.npz", saved_batch)

    motm = MOTMReference(
        ROOT / "artifacts/iclr27-r5/motm-reference-v001",
        ROOT / "artifacts/iclr27-r5/motm-runtime-v001",
        device="cuda",
        ridge=0.5,
        batch_size=32,
    )
    completed, fits = [], []
    for station, record in sorted(records.items()):
        selected = [r for r in rows if r["station"] == station]
        if not selected:
            continue
        prefix = full[station][: record["prefix_end"]]
        stats = PrefixRegression(prefix)
        defaults = np.nanmedian(prefix, axis=0)
        if not np.isfinite(stats.mean).all() or not np.isfinite(defaults).all():
            raise ValueError("a half-prefix feature has no observations")
        fitted = dict(deep)
        impute_params = {"seasonal_lag": {"period": 24}}
        for action in ACTIONS:
            if action not in fitted and (
                action == "seasonal_lag" or DEFAULT_REGISTRY.get_spec(action).fit_scope == "dataset"
            ):
                fitted[action] = runner.fit(
                    action,
                    SeriesBatch(prefix[None], np.isfinite(prefix[None]), metadata={"period": 24}),
                    {"period": 24},
                    params=impute_params.get(action),
                )
        for row in selected:
            t, age = row["origin"], row["outage_age"]
            x = calibration_context(full[station], row)
            inherited = None
            if row["outage_pattern"] == "targets_only":
                old = parent_cases[row["parent_case_id"]]
                old_path = parent / old["path"]
                if file_sha256(old_path) != old["sha256"]:
                    raise ValueError("an unchanged source control changed")
                with np.load(old_path, allow_pickle=False) as saved:
                    inherited = {k: saved[k] for k in saved.files}
                np.testing.assert_array_equal(x, inherited["context"])
                candidates = inherited["candidates"]
                coverage, status = np.asarray(old["native_coverage"]), old["statuses"]
                motm_info = old["motm"]
            else:
                artifacts = dict(fitted)
                artifacts["seasonal_lag"] = {
                    **fitted["seasonal_lag"],
                    "profiles": np.roll(fitted["seasonal_lag"]["profiles"], -(t - 192) % 24, axis=1),
                }
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
            repairs = stats.targets(x, candidates[list(ACTIONS).index("linear_interp"), :, :2])
            keep = np.r_[np.ones(11, bool), np.isfinite(x[:, 11:]).any(0)]
            data = {
                "context": x,
                "candidates": candidates,
                "actions": np.asarray([*ACTIONS, "motm_reference"]),
                "mean": stats.mean,
                "scale": stats.scale,
                "defaults": defaults,
                "keep": keep,
                "stat_names": np.asarray(list(repairs)),
                "stat_targets": np.stack(list(repairs.values())),
                "future_models": np.asarray(json.dumps(stats.future_models(keep))),
            }
            data["features"] = observable_features(data, stats)
            if inherited is not None:
                for key in data:
                    np.testing.assert_array_equal(data[key], inherited[key])
            for values in candidates:
                np.testing.assert_array_equal(values[np.isfinite(x)], x[np.isfinite(x)])
                if not np.isfinite(values).all():
                    raise ValueError("a source baseline is incomplete")
            path = output / "cases" / f"{row['case_id']}.npz"
            _save_npz(path, **data)
            hidden = np.zeros((192, 2), bool)
            hidden[-age:] = True
            label_path = output / "training-labels" / f"{row['case_id']}.npz"
            _save_npz(
                label_path,
                original_history=full[station][t - 192 : t, :2],
                future=full[station][t : t + 24, :2],
                hidden=hidden,
            )
            completed.append(
                {
                    **row,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "label_path": str(label_path.relative_to(output)),
                    "label_sha256": file_sha256(label_path),
                    "statuses": status,
                    "native_coverage": coverage.tolist(),
                    "motm": motm_info,
                    "candidate_work_reused": inherited is not None,
                    "peer_columns": augmented_sources[station][1],
                    "dropped_peer_columns": np.flatnonzero(~keep).tolist(),
                }
            )
            _write_json(output / "progress.json", {"prepared": len(completed), "total": len(rows)})
        fit_path = output / "regressions" / f"{station}.json"
        stats.save(fit_path)
        fits.append(
            {
                "station": station,
                "path": str(fit_path.relative_to(output)),
                "sha256": file_sha256(fit_path),
                "prefix_end": record["prefix_end"],
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
            "cases": completed,
            "deep_fits": deep_records,
            "deep_fits_reused_from": str(parent),
            "parent_input_manifest_sha256": file_sha256(parent / "manifest.json"),
            "training_batch_sha256": training_sha,
            "regression_fits": fits,
            "source_sha256": {s["path"]: s["sha256"] for s in records.values()},
            "fit_prefix_end": 3506,
            "calibration_end": 7012,
            "evaluation_future_values_read": False,
            "training_label_files_separate": True,
            "wall_seconds": perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
