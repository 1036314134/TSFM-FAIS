"""Reuse fixed imputer budgets on every remaining registered development origin."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_imputer_budget_study import BUDGETS, DATASETS  # noqa: E402

from tsfm_fais.contracts import SeriesBatch  # noqa: E402
from tsfm_fais.data import stable_seed  # noqa: E402
from tsfm_fais.imputers.registry import DEFAULT_REGISTRY  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def select_additional_episodes(source, prepared):
    by_id = {row["episode_id"]: row for row in source["episodes"]}
    if len(by_id) != len(source["episodes"]):
        raise ValueError("duplicate source episode identities")
    old_ids = {row["episode_id"] for row in prepared["cases"]}
    old = [by_id[name] for name in sorted(old_ids)]
    selected, origin_map = [], {}
    for dataset in DATASETS:
        original = [row for row in old if row["dataset_id"] == dataset]
        origins = {row["origin"] for row in original}
        conditions = {(row["mechanism"], row["missing_rate"], row["mask_seed"]) for row in original}
        if (
            len(origins) != 1
            or len(conditions) != 6
            or any(row["split"] != "validation" for row in original)
        ):
            raise ValueError("the original six-condition development panel changed")
        rows = [
            (index, row)
            for index, row in enumerate(source["episodes"])
            if row["dataset_id"] == dataset
            and row["split"] == "validation"
            and row["origin"] not in origins
            and (row["mechanism"], row["missing_rate"], row["mask_seed"]) in conditions
        ]
        new_origins = sorted({row["origin"] for _, row in rows})
        if len(new_origins) != 3 or len(rows) != 18:
            raise ValueError("use all three remaining registered validation origins")
        for origin in new_origins:
            if {
                (r["mechanism"], r["missing_rate"], r["mask_seed"])
                for _, r in rows
                if r["origin"] == origin
            } != conditions:
                raise ValueError("an origin is missing a registered condition")
        if any(row["episode_id"] in old_ids for _, row in rows):
            raise ValueError("the original and additional panels overlap")
        selected.extend(rows)
        origin_map[dataset] = new_origins
    return selected, origin_map


def completed_bank(record, source_path, artifacts, runner):
    with np.load(source_path, allow_pickle=False) as saved:
        context = saved["context"]
        bank = saved["candidate_values"].copy()
        actions = saved["candidate_ids"].tolist()
        coverage = saved["native_coverage"].copy()
    batch = SeriesBatch(
        context[None], np.isfinite(context[None]), metadata={"period": record["period"]}
    )
    for name, artifact in artifacts.items():
        value = runner.run(
            name, batch, artifact, seed=stable_seed(record["mask_seed"], record["origin"])
        )
        if value.failure_reason is not None:
            raise ValueError(f"budget imputation failed: {name}: {value.failure_reason}")
        complete, valid = value.values[0].copy(), value.native_valid_mask[0]
        complete[~valid] = bank[actions.index("locf")][~valid]
        np.testing.assert_array_equal(complete[np.isfinite(context)], context[np.isfinite(context)])
        if not np.isfinite(complete).all():
            raise ValueError("a budget completion remains nonfinite")
        bank[actions.index(name)] = complete
        coverage[actions.index(name)] = valid[~np.isfinite(context)].mean()
    return bank, actions, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "source-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    parent, source_root, output = (
        args.prepared_root.resolve(),
        args.source_root.resolve(),
        args.output_root.resolve(),
    )
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed origin extensions")
    base = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads((source_root / "episodes_manifest.json").read_text(encoding="utf-8"))
    if (
        base["status"] != "completed"
        or file_sha256(source_root / "episodes_manifest.json")
        != base["identity"]["source_manifest_sha256"]
    ):
        raise ValueError("the completed budget study and development source must agree")
    selected, origin_map = select_additional_episodes(source, base)
    identity = {
        "source_manifest_sha256": file_sha256(source_root / "episodes_manifest.json"),
        "parent_preparation_sha256": file_sha256(parent / "manifest.json"),
        "datasets": list(DATASETS),
        "budgets": BUDGETS,
        "evaluation_episode_ids": [row["episode_id"] for _, row in selected],
        "additional_origins": origin_map,
        "script_sha256": file_sha256(Path(__file__)),
        "imputer_source_sha256": file_sha256(ROOT / "src/tsfm_fais/imputers/pypots.py"),
        "new_fits": 0,
    }
    identity = json.loads(json.dumps(identity))
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("the origin extension changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    fits, fit_map, batches = [], {}, []
    for item in base["fits"]:
        path = parent / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("an original fitted-model record changed")
        record = json.loads(path.read_text(encoding="utf-8"))
        directory = path.parent / record["candidate_id"]
        for entry in record["files"]:
            if file_sha256(directory / entry["path"]) != entry["sha256"]:
                raise ValueError("an original imputer checkpoint changed")
        fits.append({**item, "path": str(path)})
        fit_map[(record["dataset_id"], record["budget"], record["candidate_id"])] = directory
    for item in base["training_batches"]:
        path = parent / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("a historical training batch changed")
        batches.append({**item, "path": str(path)})
    torch.set_num_threads(1)
    runner, cases, parity = CandidateRunner(), [], []
    for dataset in DATASETS:
        for budget in BUDGETS:
            artifacts = {
                name: DEFAULT_REGISTRY.create(name, device="cuda").load_artifact(
                    fit_map[(dataset, budget, name)]
                )
                for name in ("saits", "timemixerpp")
            }
            old = next(
                row
                for row in base["cases"]
                if row["dataset_id"] == dataset and row["budget"] == budget
            )
            old_source = source["episodes"][old["episode_index"]]
            if (
                file_sha256(parent / old["path"]) != old["sha256"]
                or file_sha256(source_root / old_source["path"]) != old_source["sha256"]
            ):
                raise ValueError("an imputer-replay reference changed")
            check, _, _ = completed_bank(
                old_source, source_root / old_source["path"], artifacts, runner
            )
            with np.load(parent / old["path"], allow_pickle=False) as saved:
                np.testing.assert_allclose(check, saved["candidate_values"], rtol=1e-6, atol=1e-6)
                difference = float(np.max(np.abs(check - saved["candidate_values"])))
            parity.append(
                {"dataset_id": dataset, "budget": budget, "maximum_difference": difference}
            )
            for index, record in [pair for pair in selected if pair[1]["dataset_id"] == dataset]:
                source_path = source_root / record["path"]
                if file_sha256(source_path) != record["sha256"]:
                    raise ValueError("an additional source input changed")
                key = hashlib.sha256(record["episode_id"].encode()).hexdigest()[:24]
                path = output / "episodes" / budget / f"{key}.npz"
                if not path.exists():
                    bank, actions, coverage = completed_bank(record, source_path, artifacts, runner)
                    _save_npz(
                        path,
                        candidate_values=bank,
                        candidate_ids=np.asarray(actions),
                        native_coverage=coverage,
                        identity_sha256=np.asarray(identity_sha),
                        source_sha256=np.asarray(record["sha256"]),
                    )
                with np.load(path, allow_pickle=False) as saved:
                    if (
                        str(saved["identity_sha256"]) != identity_sha
                        or str(saved["source_sha256"]) != record["sha256"]
                    ):
                        raise ValueError("an additional input cache changed identity")
                cases.append(
                    {
                        "episode_id": record["episode_id"],
                        "episode_index": index,
                        "dataset_id": dataset,
                        "budget": budget,
                        "path": str(path.relative_to(output)),
                        "sha256": file_sha256(path),
                    }
                )
                _write_json(
                    output / "progress.json",
                    {"status": "preparing", "completed_cases": len(cases), "total_cases": 162},
                )
            del artifacts
            torch.cuda.empty_cache()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "training_batches": batches,
            "fits": fits,
            "cases": cases,
            "checkpoint_replay": parity,
            "evaluation_episode_count": 54,
            "new_fits": 0,
            "new_forecaster_calls": 0,
            "panel_description": "three additional validation origins per dataset; 54 episodes on nine histories, development follow-up",
        },
    )
    print(
        json.dumps({"status": "completed", "new_histories": 9, "episodes": 54, "new_fits": 0}),
        flush=True,
    )


if __name__ == "__main__":
    main()
