"""Prepare the pinned modern-imputer comparator for every registered follow-up history."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.imputers.motm import MOTMReference  # noqa: E402
from tsfm_fais.utility_experiment import _save_npz, _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "reference-root", "runtime-root", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed MoTM follow-up inputs")
    prepared = json.loads((args.prepared_root / "manifest.json").read_text(encoding="utf-8"))
    if prepared["status"] != "completed" or len(prepared["episodes"]) != 823:
        raise ValueError("the complete registered follow-up input bank is required")
    identity = {
        "prepared_manifest_sha256": file_sha256(args.prepared_root / "manifest.json"),
        "reference_manifest_sha256": file_sha256(args.reference_root / "manifest.json"),
        "runtime_manifest_sha256": file_sha256(args.runtime_root / "manifest.json"),
        "protocol_sha256": file_sha256(args.protocol),
        "script_sha256": file_sha256(Path(__file__)),
        "module_sha256": file_sha256(ROOT / "src/tsfm_fais/imputers/motm.py"),
        "ridge": 0.5,
        "batch_size": 32,
        "input_recipe": "published observed-context adaptation and internal normalization; shared LOCF/prefix-median fallback",
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("a partial supplemental input bank changed identity")
    _write_json(identity_path, identity)
    identity_sha = file_sha256(identity_path)
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(1)
    imputer = MOTMReference(
        args.reference_root, args.runtime_root, device="cuda", ridge=0.5, batch_size=32
    )
    records = []
    for record in prepared["episodes"]:
        source = args.prepared_root / record["path"]
        if file_sha256(source) != record["sha256"]:
            raise ValueError("a registered history changed")
        path = output / "episodes" / source.name
        if not path.exists():
            with np.load(source, allow_pickle=False) as saved:
                context = saved["context"]
                fallback = saved["candidate_values"][saved["candidate_ids"].tolist().index("locf")]
            completed, diagnostics = imputer.impute(context, fallback)
            np.testing.assert_array_equal(
                completed[np.isfinite(context)], context[np.isfinite(context)]
            )
            if not np.isfinite(completed).all():
                raise ValueError("the supported MoTM completion is not finite")
            _save_npz(
                path,
                values=completed,
                diagnostics=np.asarray(json.dumps(diagnostics)),
                identity_sha256=np.asarray(identity_sha),
                source_sha256=np.asarray(record["sha256"]),
            )
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["identity_sha256"]) != identity_sha
                or str(saved["source_sha256"]) != record["sha256"]
            ):
                raise ValueError("supplemental input provenance changed")
        records.append(
            {
                "episode_id": record["episode_id"],
                "path": str(path.relative_to(output)),
                "sha256": file_sha256(path),
            }
        )
        _write_json(
            output / "progress.json",
            {"status": "preparing", "completed_episodes": len(records), "total_episodes": 823},
        )
    imputer.verify_frozen()
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "identity": identity,
            "identity_sha256": identity_sha,
            "episodes": records,
            "networks_unchanged": True,
            "forecaster_calls": 0,
            "forecast_scores_computed": False,
            "pretraining_caveat": "the reference includes a Solar-trained network; Alabama solar is not claimed absent from its pretraining",
        },
    )


if __name__ == "__main__":
    main()
