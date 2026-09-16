"""Prepare matched original-origin imputation controls for the fixed tail action."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from matched_replay_core import stable_seed
from matched_replay_pool import NativeReplayPool
from matched_replay_sources import native_sources
from prepare_native_confirmation import ACTIONS
from tail_bridge_core import supported_gaps

from tsfm_fais.forecasting.accuracy import PrefixStandardizer
from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def smoke_cases(cases):
    chosen = [
        next(row for row in cases if row["panel"] == "synthetic" and row["gap"] == gap)
        for gap in (8, 48)
    ]
    chosen.append(next(row for row in cases if row["panel"] == "native_common"))
    chosen.append(
        next(
            row
            for row in cases
            if row["panel"] == "native_target_only" and 0 in row["gaps"]["timesfm2p5"]
        )
    )
    chosen.append(
        next(
            row
            for row in cases
            if row["panel"] == "native_target_only"
            and min(row["gaps"]["timesfm2p5"]) > 0
            and len(set(row["gaps"]["timesfm2p5"])) == 2
        )
    )
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    plan_root = ROOT / "artifacts/iclr27-r20/tail-plan-v001"
    plan = read_json(plan_root / "manifest.json")
    if plan["core_module_sha256"] != file_sha256(ROOT / "scripts/tail_bridge_core.py"):
        raise ValueError("the tail action definition changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "plan_sha256": file_sha256(plan_root / "manifest.json"),
        "pool_module_sha256": file_sha256(ROOT / "scripts/matched_replay_pool.py"),
        "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R20_TAIL_BRIDGE_PROTOCOL.md"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed tail inputs")
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial tail preparation definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    cases = smoke_cases(plan["cases"]) if args.smoke else plan["cases"]
    sources = native_sources()
    source_map = {(s["cohort"], s["dataset_id"], s["item_id"]): s for s in sources}
    old_motm_root = ROOT / "artifacts/iclr27-r5/native-confirmation-v001/motm"
    old_motm = {
        row["episode_id"]: row for row in read_json(old_motm_root / "manifest.json")["episodes"]
    }
    torch.set_num_threads(1)
    motm = MOTMReference(
        ROOT / "artifacts/iclr27-r5/motm-reference-v001",
        ROOT / "artifacts/iclr27-r5/motm-runtime-v001",
        device="cuda",
        ridge=0.5,
        batch_size=32,
    )
    pools, records = {}, []
    for row in cases:
        path = output / "cases" / f"{row['case_id']}.npz"
        marker = path.with_suffix(".json")
        if marker.exists():
            done = read_json(marker)
            if done["identity_sha256"] != identity_sha or done["sha256"] != file_sha256(path):
                raise ValueError("a prepared tail case changed")
            records.append(done)
            continue
        started = perf_counter()
        key = row["cohort"], row["dataset_id"], row["item_id"]
        source = source_map[key]
        if key not in pools:
            pools[key] = NativeReplayPool(source, sources)
        pool = pools[key]
        if file_sha256(plan_root / row["context_path"]) != row["context_sha256"]:
            raise ValueError("a fixed tail context changed")
        with np.load(plan_root / row["context_path"], allow_pickle=False) as saved:
            context = saved["context"]
        for model_id in row["models"]:
            if supported_gaps(context, model_id == "chronos2") != row["gaps"][model_id]:
                raise ValueError("a registered model-scope tail length changed")
        seed = stable_seed(row.get("original_episode_id", "r20|" + row["case_id"]))
        values, coverage, statuses = pool.complete(context.copy(), row["origin"], seed)
        extra, diag = motm.impute(context, values[list(ACTIONS).index("locf")])
        max_delta = motm_delta = None
        if row["current_artifact"] is not None:
            if file_sha256(Path(row["current_artifact"])) != row["current_artifact_sha256"]:
                raise ValueError("a natural baseline input changed")
            with np.load(row["current_artifact"], allow_pickle=False) as saved:
                np.testing.assert_array_equal(context, saved["context"])
                old_values = saved["candidate_values"]
                np.testing.assert_allclose(values, old_values, rtol=1e-6, atol=1e-6)
                max_delta = float(abs(values - old_values).max())
                old_extra = saved["motm_values"] if row["cohort"] == "r6_native" else None
            if old_extra is None:
                entry = old_motm[row["original_episode_id"]]
                if file_sha256(old_motm_root / entry["path"]) != entry["sha256"]:
                    raise ValueError("a natural MoTM baseline changed")
                with np.load(old_motm_root / entry["path"], allow_pickle=False) as saved:
                    old_extra = saved["values"]
            np.testing.assert_allclose(extra, old_extra, rtol=2e-5, atol=2e-5)
            motm_delta = float(abs(extra - old_extra).max())
        scaler = PrefixStandardizer.fit(source["values"][: source["prefix_end"]])
        long_inputs = {}
        for length in (1024, 4096):
            start = max(source["prefix_end"], row["origin"] - length)
            long = source["values"][start : row["origin"]].copy()
            long[-96:] = context
            long_inputs[f"long{length}"] = long
        _save_npz(
            path,
            context=context,
            candidate_values=np.concatenate([values, extra[None]]),
            candidate_ids=np.asarray([*ACTIONS, "motm_reference"]),
            mean=scaler.mean,
            scale=scaler.scale,
            defaults=pool.defaults,
            identity_sha256=np.asarray(identity_sha),
            **long_inputs,
        )
        record = {
            "case_id": row["case_id"],
            "path": str(path.relative_to(output)),
            "sha256": file_sha256(path),
            "identity_sha256": identity_sha,
            "seconds": perf_counter() - started,
            "frozen_neural_imputers": pool.fit_records,
            "latest_neural_training_boundary": str(pool.latest_training_boundary),
            "current_replay_max_difference": max_delta,
            "motm_replay_max_difference": motm_delta,
            "statuses": statuses,
            "native_coverage": coverage.tolist(),
            "motm": diag,
        }
        _write_json(marker, record)
        records.append(record)
        print(f"prepared {row['case_id']}", flush=True)
    motm.verify_frozen()
    _write_json(
        output / ("smoke_preparation.json" if args.smoke else "manifest.json"),
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "cases": records,
            "neural_imputer_refits": 0,
            "current_future_values_read": False,
            "limits": "fixed-input development tail probe, no current-future scoring",
        },
    )
    del pools, motm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
