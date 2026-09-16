"""Freeze evaluation contexts, source-matched controls and real L192 seasonal inputs."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401
from audit_metric_source_gates import direct_control
from latent_source_inputs import ROOT, read_json
from matched_replay_sources import native_sources
from metric_source_gate import fit_fixed_metric
from patch_repair_eval_support import POOL, pool_catalog, source_bank
from patch_repair_inputs import (
    SOURCE,
    load_case,
    source_identity,
    source_population,
    training_weights,
)
from prepare_native_confirmation import complete_candidates

from tsfm_fais.contracts import SeriesBatch
from tsfm_fais.imputers import CandidateRunner
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed repair evaluation preparation")
    output.mkdir(parents=True, exist_ok=True)
    population, excluded = source_population()
    train, validation = (
        [r for r in population if r["split"] == "train"],
        [r for r in population if r["split"] == "validation"],
    )
    if len(train) != 2682 or len(validation) != 864:
        raise ValueError("the frozen training/validation split changed")
    matched = {}
    for model_id in ("chronos2", "timesfm2p5"):
        catalog = pool_catalog(model_id)
        points, truths = [], []
        for row in train:
            bank, actions = source_bank(row, model_id, catalog)
            _, _, truth = load_case(row)
            normal = (truth - np.asarray(row["scaler"]["mean"][:2])) / row["scaler"]["scale"][:2]
            points.append(bank)
            truths.append(normal)
        points, truths = np.stack(points), np.stack(truths)
        weights = training_weights(train)
        if model_id == "chronos2":
            vectors, target = points.reshape(len(train), 8, 192), truths.reshape(len(train), 192)
        else:
            vectors, target = (
                points.transpose(0, 3, 1, 2).reshape(-1, 8, 96),
                truths.transpose(0, 2, 1).reshape(-1, 96),
            )
            weights = np.repeat(weights, 2)
        mae = np.average(abs(vectors - target[:, None]).mean(-1), axis=0, weights=weights)
        controls = {
            kind: fit_fixed_metric(vectors, target, weights, kind) for kind in ("mae", "joint")
        }
        gaps = {}
        for kind, fitted in controls.items():
            probability = np.asarray(fitted["weights"])
            objective, gradient = direct_control(vectors, target, weights, probability, kind)
            gap = (gradient @ probability - gradient.min()) / max(abs(gradient).max(), 1.0)
            if gap > 1e-7 or abs(probability.sum() - 1) > 1e-10 or probability.min() < 0:
                raise ValueError("a source-matched fixed control failed optimality")
            np.testing.assert_allclose(objective, fitted["objective"], rtol=1e-12, atol=1e-12)
            gaps[kind] = float(gap)
        matched[model_id] = {
            "actions": actions,
            "single_index": int(mae.argmin()),
            "single_mae": mae.tolist(),
            "fixed_mae": controls["mae"],
            "fixed_joint": controls["joint"],
            "independent_optimality_gaps": gaps,
        }
    _write_json(
        output / "matched_controls.json",
        {
            "source_identity": source_identity(),
            "models": matched,
            "training_episode_ids": [r["episode_id"] for r in train],
            "evaluation_labels_used": False,
        },
    )
    cases = []
    trace_source = {
        validation[0]["episode_id"],
        max(validation, key=lambda r: r["dimensions"])["episode_id"],
    }
    for row in validation:
        original_path = SOURCE / row["path"]
        if file_sha256(original_path) != row["sha256"]:
            raise ValueError("a source validation input changed")
        with np.load(original_path, allow_pickle=False) as saved:
            context = saved["context"]
            base = saved["candidate_values"][saved["candidate_ids"].tolist().index("seasonal_lag")]
        case_id = "source_" + hashlib.sha256(row["episode_id"].encode()).hexdigest()[:20]
        path = output / "cases" / f"{case_id}.npz"
        _save_npz(
            path,
            context=context,
            base=base,
            mean=np.asarray(row["scaler"]["mean"]),
            scale=np.asarray(row["scaler"]["scale"]),
        )
        metadata = {
            name: row[name]
            for name in (
                "episode_id",
                "episode_index",
                "origin_id",
                "dataset_id",
                "family_id",
                "item_id",
                "origin",
            )
        }
        cases.append(
            {
                **metadata,
                "case_id": case_id,
                "panel": "source_validation",
                "context_length": 96,
                "group_id": row["family_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "original_path": str(original_path),
                "original_sha256": row["sha256"],
                "trace": row["episode_id"] in trace_source,
            }
        )
    sources = native_sources()
    source_map = {(r["cohort"], r["dataset_id"], r["item_id"]): r for r in sources}
    old_root = ROOT / "artifacts/iclr27-r21/provenance-inputs-v001"
    old = read_json(old_root / "manifest.json")
    fitting, fitted, trace_groups = [], {}, set()
    runner = CandidateRunner()
    for row in old["cases"]:
        key = row["cohort"], row["dataset_id"], row["item_id"]
        source = source_map[key]
        prefix = source["values"][: source["prefix_end"]]
        defaults = np.nanmedian(prefix, axis=0)
        period = max(2, source["period"])
        if key not in fitted:
            then = perf_counter()
            fitted[key] = runner.fit(
                "seasonal_lag",
                SeriesBatch(prefix[None], np.isfinite(prefix[None]), metadata={"period": period}),
                {},
                params={"period": period},
            )
            fitting.append(
                {
                    "cohort": key[0],
                    "dataset_id": key[1],
                    "item_id": key[2],
                    "seconds": perf_counter() - then,
                }
            )
        trace = row["group_id"] not in trace_groups
        trace_groups.add(row["group_id"])
        for length in (96, 192):
            start = row["origin"] - length
            if start < source["prefix_end"]:
                raise ValueError("native seasonal statistics overlap the current context")
            context = source["values"][start : row["origin"]]
            artifact = {
                **fitted[key],
                "profiles": np.roll(fitted[key]["profiles"], -start % period, axis=1),
            }
            result = runner.run_many(
                ("locf", "seasonal_lag"),
                SeriesBatch(context[None], np.isfinite(context[None]), metadata={"period": period}),
                {"seasonal_lag": artifact},
                seed=0,
                params={"seasonal_lag": {"period": period}},
            )
            completed, _, _ = complete_candidates(
                context, result, defaults, actions=("locf", "seasonal_lag")
            )
            base = completed[1]
            with np.load(old_root / row["path"], allow_pickle=False) as previous:
                if length == 96:
                    np.testing.assert_array_equal(
                        base,
                        previous["candidate_values"][
                            previous["candidate_ids"].tolist().index("seasonal_lag")
                        ],
                    )
                case_id = f"native_{row['case_id']}_l{length}"
                path = output / "cases" / f"{case_id}.npz"
                _save_npz(
                    path, context=context, base=base, mean=previous["mean"], scale=previous["scale"]
                )
            metadata = {
                name: row[name]
                for name in (
                    "episode_id",
                    "dataset_id",
                    "family_id",
                    "item_id",
                    "origin",
                    "group_id",
                    "cohort",
                )
            }
            cases.append(
                {
                    **metadata,
                    "origin_id": row["episode_id"],
                    "case_id": case_id,
                    "native_case_id": row["case_id"],
                    "panel": "native_development",
                    "context_length": length,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "original_path": row["original_path"],
                    "original_sha256": row["original_sha256"],
                    "trace": trace and length == 192,
                }
            )
    if len(cases) != 910:
        raise ValueError("the registered evaluation population changed")
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "script_sha256": file_sha256(Path(__file__)),
            "source_identity": source_identity(),
            "old_native_inputs_sha256": file_sha256(old_root / "manifest.json"),
            "source_pool_sha256": file_sha256(POOL / "manifest.json"),
            "matched_controls_sha256": file_sha256(output / "matched_controls.json"),
            "cases": cases,
            "classical_prefix_fits": fitting,
            "source_training_labels_used_for_fixed_controls": True,
            "evaluation_future_labels_read": False,
            "excluded_source_cases": len(excluded),
            "independent_confirmation": False,
        },
    )


if __name__ == "__main__":
    main()
