"""Prepare the pinned MoTM candidate on every unchanged R7 source context."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset  # noqa: F401 - initialize Arrow before Torch on Windows.
import torch
from latent_source_inputs import ROOT, read_json
from pool_gate_inputs import motm_coverage

from tsfm_fais.imputers.motm import MOTMReference
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, path in {
        "source-root": "artifacts/iclr27-r3/development-expanded-v001",
        "base-root": "artifacts/iclr27-r7/latent-source-v001",
        "reference-root": "artifacts/iclr27-r5/motm-reference-v001",
        "runtime-root": "artifacts/iclr27-r5/motm-runtime-v001",
        "probe-root": "artifacts/iclr27-r5/motm-source-feasibility-v001",
        "protocol": "docs/iclr2027/R12_MOTM_POOL_PROTOCOL.md",
    }.items():
        parser.add_argument("--" + name, type=Path, default=ROOT / path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed source MoTM inputs")
    source = read_json(args.source_root / "episodes_manifest.json")
    base = read_json(args.base_root / "manifest.json")
    probe = read_json(args.probe_root / "manifest.json")
    if (
        base["identity"]["source_manifest_sha256"]
        != file_sha256(args.source_root / "episodes_manifest.json")
        or probe["status"] != "completed"
        or probe["identity"]["module_sha256"]
        != file_sha256(ROOT / "src/tsfm_fais/imputers/motm.py")
    ):
        raise ValueError("source population or verified MoTM implementation changed")
    chosen = [row for row in source["episodes"] if row["mask_seed"] == 6101]
    if len(chosen) != 3906:
        raise ValueError("the original source population changed")
    identity = {
        "script_sha256": file_sha256(Path(__file__)),
        "source_sha256": file_sha256(args.source_root / "episodes_manifest.json"),
        "base_sha256": file_sha256(args.base_root / "manifest.json"),
        "reference_sha256": file_sha256(args.reference_root / "manifest.json"),
        "runtime_sha256": file_sha256(args.runtime_root / "manifest.json"),
        "probe_sha256": file_sha256(args.probe_root / "manifest.json"),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/imputers/motm.py"),
        "protocol_sha256": file_sha256(args.protocol),
        "ridge": 0.5,
        "batch_size": 32,
        "context_duplication_seed": 42,
    }
    for field, key in (
        ("reference_sha256", "reference_manifest_sha256"),
        ("runtime_sha256", "runtime_manifest_sha256"),
    ):
        if identity[field] != probe["identity"][key]:
            raise ValueError("the published MoTM reference changed")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("partial source MoTM definitions changed")
    _write_json(output / "identity.json", identity)
    identity_sha = file_sha256(output / "identity.json")
    torch.set_num_threads(1)
    imputer = MOTMReference(
        args.reference_root, args.runtime_root, device="cuda", ridge=0.5, batch_size=32
    )
    replay_cache, replays = {}, []
    by_id = {row["episode_id"]: row for row in chosen}
    for episode_id in probe["identity"]["episodes"]:
        row = by_id[episode_id]
        path = args.source_root / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError("a fixed source parity input changed")
        with np.load(path, allow_pickle=False) as raw:
            context = raw["context"]
            actions = raw["candidate_ids"].tolist()
            fallback = raw["candidate_values"][actions.index("locf")]
        started = perf_counter()
        values, diagnostics = imputer.impute(context, fallback)
        seconds = perf_counter() - started
        reference = args.probe_root / (
            hashlib.sha256(episode_id.encode()).hexdigest()[:20] + "-imputation.npz"
        )
        with np.load(reference, allow_pickle=False) as saved:
            np.testing.assert_array_equal(context, saved["context"])
            np.testing.assert_allclose(values, saved["completed"], rtol=2e-5, atol=2e-5)
            delta = float(abs(values - saved["completed"]).max())
        replay_cache[episode_id] = (values, diagnostics, seconds)
        replays.append(
            {
                "episode_id": episode_id,
                "reference_file_sha256": file_sha256(reference),
                "maximum_difference": delta,
            }
        )
    records = []
    for row in chosen:
        source_path = args.source_root / row["path"]
        if file_sha256(source_path) != row["sha256"]:
            raise ValueError("a source context changed")
        path = output / "episodes" / source_path.name
        with np.load(source_path, allow_pickle=False) as raw:
            context = raw["context"]
            actions = raw["candidate_ids"].tolist()
            fallback = raw["candidate_values"][actions.index("locf")]
        if not path.exists():
            if row["episode_id"] in replay_cache:
                values, diagnostics, seconds = replay_cache[row["episode_id"]]
            else:
                started = perf_counter()
                values, diagnostics = imputer.impute(context, fallback)
                seconds = perf_counter() - started
            np.testing.assert_array_equal(
                values[np.isfinite(context)], context[np.isfinite(context)]
            )
            if not np.isfinite(values).all():
                raise ValueError("the MoTM candidate is incomplete")
            _save_npz(
                path,
                values=values,
                diagnostics=np.asarray(json.dumps(diagnostics)),
                native_coverage=np.asarray(motm_coverage(context, diagnostics["fallback_columns"])),
                seconds=np.asarray(seconds),
                identity_sha256=np.asarray(identity_sha),
                source_sha256=np.asarray(row["sha256"]),
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["source_sha256"]) != row["sha256"]
            ):
                raise ValueError("a source MoTM cache changed")
            np.testing.assert_array_equal(
                saved["values"][np.isfinite(context)], context[np.isfinite(context)]
            )
            diagnostics = json.loads(str(saved["diagnostics"]))
            seconds = float(saved["seconds"])
        records.append(
            {
                "episode_id": row["episode_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
                "fitted_variables": diagnostics["fitted_variables"],
                "fallback_columns": diagnostics["fallback_columns"],
                "seconds": seconds,
            }
        )
        if len(records) % 100 == 0:
            print(f"MoTM source: {len(records)}/3906 contexts", flush=True)
    imputer.verify_frozen()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "episodes": records,
            "original_parity_replays": replays,
            "pretrained_networks_unchanged": True,
            "network_parameter_sha256": imputer.initial_digest,
            "context_optimization": imputer.settings,
            "future_arrays_read": False,
            "clean_context_arrays_read": False,
            "new_forecaster_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
