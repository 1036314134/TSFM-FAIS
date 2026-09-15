"""Prepare matched source-prefix training-budget controls on development data."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_native_confirmation import MECHANISMS, saved_files  # noqa: E402

from tsfm_fais.contracts import SeriesBatch  # noqa: E402
from tsfm_fais.data import MaskingSpec, load_dataset, load_manifest, stable_seed  # noqa: E402
from tsfm_fais.imputers.registry import DEFAULT_REGISTRY  # noqa: E402
from tsfm_fais.imputers.runner import CandidateRunner  # noqa: E402
from tsfm_fais.stage_execution import _training_batch, _training_batch_summary  # noqa: E402
from tsfm_fais.utility_experiment import (  # noqa: E402
    _save_npz,
    _write_json,
    file_sha256,
    load_utility_config,
)

DATASETS = ("ETTh1", "current_velocity_H", "electricity")
BUDGETS = {
    "epochs10_windows64": (10, 64),
    "epochs50_windows64": (50, 64),
    "epochs50_windows512": (50, 512),
}


def nested_training_batch(base, larger, limit=512):
    """Keep every original window and add a deterministic subset of new windows."""
    if base.shape[1:] != larger.shape[1:] or limit < base.shape[0]:
        raise ValueError("nested training batches need aligned shapes and a sufficient limit")
    existing = set(base.item_ids)
    choices = [index for index, name in enumerate(larger.item_ids) if name not in existing]
    needed = limit - base.shape[0]
    if len(choices) < needed:
        raise ValueError("not enough distinct historical windows for the larger budget")
    selected = (
        np.asarray(choices)[np.linspace(0, len(choices) - 1, needed, dtype=int)]
        if needed
        else np.array([], int)
    )
    names = (*base.item_ids, *(larger.item_ids[index] for index in selected))
    if len(set(names)) != len(names):
        raise ValueError("nested sampling duplicated a historical descriptor")
    return SeriesBatch(
        np.concatenate([base.values, larger.values[selected]]),
        np.concatenate([base.observed_mask, larger.observed_mask[selected]]),
        item_ids=names,
        metadata={"training_sampling_protocol": "legacy64_plus_uniform_new448_v1"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-root", "plan", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed imputer-budget preparations")
    output.mkdir(parents=True, exist_ok=True)
    config = load_utility_config(args.config)
    source_path = args.source_root / "episodes_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["source_manifest_sha256"] != file_sha256(source_path):
        raise ValueError("the development panel source changed")
    selected = [
        (index, record)
        for index, record in enumerate(source["episodes"])
        if record["episode_id"] in plan["decision_episode_ids"] and record["dataset_id"] in DATASETS
    ]
    if len(selected) != 18 or any(record["split"] != "validation" for _, record in selected):
        raise ValueError("use the three preselected development datasets and their 18 tasks")
    identity = {
        "source_manifest_sha256": file_sha256(source_path),
        "plan_sha256": file_sha256(args.plan),
        "config_sha256": file_sha256(args.config),
        "datasets": list(DATASETS),
        "budgets": BUDGETS,
        "imputers": ["saits", "timemixerpp"],
        "model_structure": "unchanged registered defaults",
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "src/tsfm_fais/stage_execution.py",
                "src/tsfm_fais/data/masking.py",
                "src/tsfm_fais/imputers/pypots.py",
                "src/tsfm_fais/imputers/runner.py",
                "scripts/prepare_native_confirmation.py",
            )
        },
        "script_sha256": file_sha256(Path(__file__)),
        "scope": "development-only training-budget sensitivity",
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(
        identity_path.read_text(encoding="utf-8")
    ) != json.loads(json.dumps(identity)):
        raise ValueError("the budget-study preparation changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    catalog = load_manifest(config.data_manifest)
    torch.set_num_threads(1)
    runner, fits, cases, batches = CandidateRunner(), [], [], []
    for dataset in DATASETS:
        source_dataset = next(row for row in source["datasets"] if row["dataset_id"] == dataset)
        for path, expected in source_dataset["sources"].items():
            if file_sha256(Path(path)) != expected:
                raise ValueError("a development data source changed")
        legacy_root = Path(source_dataset["fit_artifacts"]["root"])
        if (
            file_sha256(legacy_root / "manifest.json")
            != source_dataset["fit_artifacts"]["manifest_sha256"]
        ):
            raise ValueError("legacy imputer provenance changed")
        legacy = json.loads((legacy_root / "manifest.json").read_text(encoding="utf-8"))
        summary = legacy["datasets"][dataset]["training_summary"]
        item_ids = list(dict.fromkeys(name.rsplit("@", 1)[0] for name in summary["item_ids"]))
        spec = catalog.get(dataset)
        item_map = {item.item_id: item for item in load_dataset(spec)}
        items = [item_map[name] for name in item_ids]
        seeds = [2101, 2102, 2103] if dataset == "ETTh1" else [1101, 1102, 1103]
        settings = dict(
            dataset_id=dataset,
            masking_specs=[
                MaskingSpec(name, rate, (6, 12, 24, 48))
                for name in MECHANISMS
                for rate in (0.1, 0.2, 0.3, 0.4, 0.5)
            ],
            configured_seeds=seeds,
            fit_fraction=0.2,
            training_stride=24,
        )
        small = _training_batch(items, 96, 96, 64, **settings)
        if _training_batch_summary(small) != summary:
            _write_json(
                output / f"{dataset}_training_mismatch.json",
                {"expected": summary, "recreated": _training_batch_summary(small)},
            )
            raise ValueError(
                "the legacy 64-window batch did not reproduce; do not attribute differences to budget"
            )
        large = nested_training_batch(small, _training_batch(items, 96, 96, 512, **settings))
        batch_map = {64: small, 512: large}
        for count, batch in batch_map.items():
            path = output / "training_batches" / f"{dataset}_{count}.npz"
            _save_npz(
                path,
                values=batch.values,
                observed=batch.observed_mask,
                item_ids=np.asarray(batch.item_ids),
            )
            batches.append(
                {
                    "dataset_id": dataset,
                    "windows": count,
                    "path": str(path.relative_to(output)),
                    "sha256": file_sha256(path),
                    "summary": _training_batch_summary(batch),
                }
            )
        for budget, (epochs, count) in BUDGETS.items():
            artifacts = {}
            for name in identity["imputers"]:
                directory = output / "imputers" / dataset / budget / name
                marker = directory.parent / f"{name}.json"
                params = {
                    "epochs": epochs,
                    "batch_size": 16,
                    "random_state": 0,
                    "num_samples": 5,
                    "device": "cuda",
                }
                adapter = DEFAULT_REGISTRY.create(name, **params)
                if marker.exists():
                    record = json.loads(marker.read_text(encoding="utf-8"))
                    if record["identity_sha256"] != identity_sha:
                        raise ValueError("a budget checkpoint has a different identity")
                    for entry in record["files"]:
                        if file_sha256(directory / entry["path"]) != entry["sha256"]:
                            raise ValueError("a fitted budget checkpoint changed")
                    artifact = adapter.load_artifact(directory)
                else:
                    started = monotonic()
                    artifact = runner.fit(
                        name, batch_map[count], {"period": spec.period}, params=params
                    )
                    adapter.save_artifact(artifact, directory)
                    record = {
                        "identity_sha256": identity_sha,
                        "dataset_id": dataset,
                        "budget": budget,
                        "candidate_id": name,
                        "epochs": epochs,
                        "windows": count,
                        "seconds": monotonic() - started,
                        "training_best_loss": float(getattr(artifact.model, "best_loss", np.nan)),
                        "files": saved_files(directory),
                        "constructor_params": json.loads(
                            (directory / "metadata.json").read_text(encoding="utf-8")
                        )["constructor_params"],
                    }
                    _write_json(marker, record)
                artifacts[name] = artifact
                fits.append(
                    {"path": str(marker.relative_to(output)), "sha256": file_sha256(marker)}
                )
                _write_json(
                    output / "progress.json",
                    {
                        "status": "fitting",
                        "completed_fits": len(fits),
                        "total_fits": 18,
                        "dataset_id": dataset,
                        "budget": budget,
                        "candidate_id": name,
                    },
                )
            for index, record in [pair for pair in selected if pair[1]["dataset_id"] == dataset]:
                original = args.source_root / record["path"]
                if file_sha256(original) != record["sha256"]:
                    raise ValueError("an original development input changed")
                key = hashlib.sha256(record["episode_id"].encode()).hexdigest()[:24]
                destination = output / "episodes" / budget / f"{key}.npz"
                if not destination.exists():
                    with np.load(original, allow_pickle=False) as saved:
                        context, bank, actions, coverage = (
                            saved["context"],
                            saved["candidate_values"].copy(),
                            saved["candidate_ids"].tolist(),
                            saved["native_coverage"].copy(),
                        )
                    batch = SeriesBatch(
                        context[None], np.isfinite(context[None]), metadata={"period": spec.period}
                    )
                    for name, artifact in artifacts.items():
                        value = runner.run(
                            name,
                            batch,
                            artifact,
                            seed=stable_seed(record["mask_seed"], record["origin"]),
                        )
                        if value.failure_reason is not None:
                            raise ValueError(
                                f"budget imputation failed: {name}: {value.failure_reason}"
                            )
                        completed = value.values[0].copy()
                        valid = value.native_valid_mask[0]
                        completed[~valid] = bank[actions.index("locf")][~valid]
                        np.testing.assert_array_equal(
                            completed[np.isfinite(context)], context[np.isfinite(context)]
                        )
                        bank[actions.index(name)] = completed
                        coverage[actions.index(name)] = valid[~np.isfinite(context)].mean()
                    _save_npz(
                        destination,
                        candidate_values=bank,
                        candidate_ids=np.asarray(actions),
                        native_coverage=coverage,
                        identity_sha256=np.asarray(identity_sha),
                        source_sha256=np.asarray(record["sha256"]),
                    )
                with np.load(destination, allow_pickle=False) as saved:
                    if (
                        str(saved["identity_sha256"]) != identity_sha
                        or str(saved["source_sha256"]) != record["sha256"]
                    ):
                        raise ValueError("a budget input cache changed provenance")
                cases.append(
                    {
                        "episode_id": record["episode_id"],
                        "episode_index": index,
                        "dataset_id": dataset,
                        "budget": budget,
                        "path": str(destination.relative_to(output)),
                        "sha256": file_sha256(destination),
                    }
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
            "new_forecaster_calls": 0,
        },
    )
    _write_json(
        output / "progress.json",
        {"status": "completed", "completed_fits": len(fits), "total_fits": 18},
    )


if __name__ == "__main__":
    main()
