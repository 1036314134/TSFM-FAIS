"""Prepare common-anchor mask replays with unchanged native-prefix imputer fits."""

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
import torch
from latent_source_inputs import ROOT, read_json
from matched_replay_core import RULES, mask_rules, stable_seed
from matched_replay_pool import NativeReplayPool
from matched_replay_sources import native_sources
from prepare_native_confirmation import ACTIONS

from tsfm_fais.forecasting.accuracy import PrefixStandardizer
from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    plan_root = ROOT / "artifacts/iclr27-r19/pilot-plan-v001"
    plan = read_json(plan_root / "manifest.json")
    if plan["status"] != "completed" or plan["core_module_sha256"] != file_sha256(
        ROOT / "scripts/matched_replay_core.py"
    ):
        raise ValueError("the frozen replay plan changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "plan_sha256": file_sha256(plan_root / "manifest.json"),
        "pool_module_sha256": file_sha256(ROOT / "scripts/matched_replay_pool.py"),
        "source_module_sha256": file_sha256(ROOT / "scripts/matched_replay_sources.py"),
        "protocol_sha256": file_sha256(ROOT / "docs/iclr2027/R19_MATCHED_REPLAY_PROTOCOL.md"),
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial replay preparation definitions changed")
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed replay preparations")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    cases = plan["cases"]
    if args.smoke:
        cases = [
            next(row for row in cases if row["group_id"] == group)
            for group in plan["source_groups"]
        ]
    sources = native_sources()
    source_map = {(row["cohort"], row["dataset_id"], row["item_id"]): row for row in sources}
    legacy_motm = ROOT / "artifacts/iclr27-r5/native-confirmation-v001/motm"
    old_motm = {
        row["episode_id"]: row for row in read_json(legacy_motm / "manifest.json")["episodes"]
    }
    torch.set_num_threads(1)
    motm = MOTMReference(
        ROOT / "artifacts/iclr27-r5/motm-reference-v001",
        ROOT / "artifacts/iclr27-r5/motm-runtime-v001",
        device="cuda",
        ridge=0.5,
        batch_size=32,
    )
    records, pools = [], {}
    for row in cases:
        path = output / "cases" / f"{row['case_id']}.npz"
        marker = path.with_suffix(".json")
        if marker.exists():
            done = read_json(marker)
            if done["identity_sha256"] != identity_sha or file_sha256(path) != done["sha256"]:
                raise ValueError("a prepared replay case changed")
            records.append(done)
            continue
        started = perf_counter()
        key = row["cohort"], row["dataset_id"], row["item_id"]
        source = source_map[key]
        if key not in pools:
            pools[key] = NativeReplayPool(source, sources)
        pool = pools[key]
        if file_sha256(plan_root / row["mask_path"]) != row["mask_sha256"]:
            raise ValueError("a fixed query mask changed")
        with np.load(plan_root / row["mask_path"], allow_pickle=False) as saved:
            context, masks = saved["context"], saved["masks"]
        rebuilt, definition = mask_rules(~np.isfinite(context), row["episode_id"])
        np.testing.assert_array_equal(masks, rebuilt)
        if definition != row["mask_definition"]:
            raise ValueError("a mask definition changed")
        scaler = PrefixStandardizer.fit(source["values"][: source["prefix_end"]])
        candidates, coverage, statuses = pool.complete(
            context.copy(), row["origin"], stable_seed(row["episode_id"])
        )
        if file_sha256(Path(row["current_artifact"])) != row["current_artifact_sha256"]:
            raise ValueError("an original current input changed")
        with np.load(row["current_artifact"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["context"], context)
            old_candidates = saved["candidate_values"]
            np.testing.assert_allclose(candidates, old_candidates, rtol=1e-6, atol=1e-6)
            old_extra = saved["motm_values"] if row["cohort"] == "r6_native" else None
        extra, diagnostics = motm.impute(context, candidates[list(ACTIONS).index("locf")])
        if old_extra is None:
            entry = old_motm[row["episode_id"]]
            if file_sha256(legacy_motm / entry["path"]) != entry["sha256"]:
                raise ValueError("an original MoTM current completion changed")
            with np.load(legacy_motm / entry["path"], allow_pickle=False) as saved:
                old_extra = saved["values"]
        np.testing.assert_allclose(extra, old_extra, rtol=2e-5, atol=2e-5)
        current_candidates = np.concatenate([candidates, extra[None]])
        histories, historical_candidates, historical_future, details = [], [], [], []
        for rule_index, rule in enumerate(RULES):
            rule_contexts, rule_candidates = [], []
            for anchor_index, origin in enumerate(row["selected_anchors"]):
                if origin + 96 > row["origin"] or origin - 96 < source["prefix_end"]:
                    raise ValueError("a historical anchor violates the time boundary")
                clean = source["values"][origin - 96 : origin].copy()
                if not np.isfinite(clean).all():
                    raise ValueError("the exact-mask pilot requires complete historical contexts")
                masked = clean.copy()
                masked[masks[rule_index]] = np.nan
                completed, covered, status = pool.complete(
                    masked.copy(), origin, stable_seed(f"r19|{row['episode_id']}|{rule}|{origin}")
                )
                imputed, diag = motm.impute(masked, completed[list(ACTIONS).index("locf")])
                values = np.concatenate([completed, imputed[None]])
                np.testing.assert_array_equal(
                    values[:, np.isfinite(masked)],
                    np.broadcast_to(masked, values.shape)[:, np.isfinite(masked)],
                )
                rule_contexts.append(masked)
                rule_candidates.append(values)
                details.append(
                    {
                        "rule": rule,
                        "anchor_index": anchor_index,
                        "origin": origin,
                        "statuses": status,
                        "native_coverage": covered.tolist(),
                        "motm": diag,
                    }
                )
                if rule_index == 0:
                    future = source["values"][origin : origin + 96, :2].copy()
                    if (np.isfinite(future).sum(0) < 48).any():
                        raise ValueError("a past future lost observed-target support")
                    historical_future.append(future)
            histories.append(rule_contexts)
            historical_candidates.append(rule_candidates)
        _save_npz(
            path,
            context=context,
            current_candidates=current_candidates,
            candidate_ids=np.asarray([*ACTIONS, "motm_reference"]),
            masks=masks,
            historical_contexts=np.asarray(histories),
            historical_candidates=np.asarray(historical_candidates),
            historical_future=np.asarray(historical_future),
            history_origins=np.asarray(row["selected_anchors"]),
            long_context=source["values"][row["long_start"] : row["origin"]].copy(),
            mean=scaler.mean,
            scale=scaler.scale,
            defaults=pool.defaults,
            identity_sha256=np.asarray(identity_sha),
        )
        record = {
            "case_id": row["case_id"],
            "path": str(path.relative_to(output)),
            "sha256": file_sha256(path),
            "identity_sha256": identity_sha,
            "seconds": perf_counter() - started,
            "current_candidate_replay_max_difference": float(
                abs(candidates - old_candidates).max()
            ),
            "current_motm_replay_max_difference": float(abs(extra - old_extra).max()),
            "frozen_neural_imputers": pool.fit_records,
            "classical_fit_calls_for_series": pool.classical_fit_calls,
            "latest_neural_training_boundary": str(pool.latest_training_boundary),
            "current_statuses": statuses,
            "current_motm": diagnostics,
            "historical_imputations": details,
        }
        _write_json(marker, record)
        records.append(record)
        print(f"prepared {row['case_id']}: shared eight-anchor masks", flush=True)
    motm.verify_frozen()
    _write_json(
        output / ("smoke_preparation.json" if args.smoke else "manifest.json"),
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "cases": records,
            "case_count": len(records),
            "neural_imputer_refits": 0,
            "current_future_values_read": False,
            "current_predictions_scored": False,
            "limits": "development replay inputs with fixed native-prefix fits; contextual MoTM optimization counted separately",
        },
    )
    del motm, pools
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
