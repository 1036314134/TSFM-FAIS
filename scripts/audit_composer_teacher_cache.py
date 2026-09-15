"""Audit complete teacher coverage without modifying historical training outputs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.routing.teacher_cache import collect_teacher_records  # noqa: E402
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    if (root / "teacher_cache_audit.json").exists():
        raise ValueError("preserve completed teacher cache audit")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or not manifest["parameter_digest_unchanged"]:
        raise ValueError("training and its frozen-parameter check must be complete")
    source_path = args.source_root / "episodes_manifest.json"
    if file_sha256(source_path) != identity["source_manifest_sha256"]:
        raise ValueError("source episodes changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = {row["episode_id"]: row for row in source["episodes"]}
    expected = set()
    for episode in identity["training_ids"]:
        record = records[episode]
        if record["split"] != "train":
            raise ValueError("teacher request contains a non-training episode")
        path = args.source_root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("source candidate data changed")
        with np.load(path, allow_pickle=False) as saved:
            if not np.isfinite(saved["context"]).all():
                expected.add(episode)
    config = source["identity"]["config"]
    teachers = collect_teacher_records(
        root / "teacher-cache",
        expected,
        file_sha256(root / "identity.json"),
        manifest["forecaster_parameter_sha256"],
        (config["horizon"], len(config["target_indices"])),
    )
    logical = len(teachers) * (len(config["candidate_ids"]) + 1)
    _write_json(
        root / "teacher_manifest_v002.json",
        {
            "identity_sha256": file_sha256(root / "identity.json"),
            "records": teachers,
            "logical_forecast_calls": logical,
        },
    )
    audit = {
        "status": "passed",
        "run_manifest_sha256": file_sha256(root / "manifest.json"),
        "script_sha256": file_sha256(Path(__file__)),
        "teacher_cache_module_sha256": file_sha256(ROOT / "src/tsfm_fais/routing/teacher_cache.py"),
        "expected_teacher_episodes": len(expected),
        "verified_teacher_episodes": len(teachers),
        "corrected_teacher_logical_forecast_calls": logical,
        "original_reported_teacher_episodes": manifest["teacher_cached_episodes"],
        "original_reported_logical_forecast_calls": manifest["teacher_logical_forecast_calls"],
        "correction_required": manifest["teacher_cached_episodes"] != len(teachers)
        or manifest["teacher_logical_forecast_calls"] != logical,
        "corrected_teacher_manifest_sha256": file_sha256(root / "teacher_manifest_v002.json"),
        "interpretation": "global logical teacher coverage replaces the final-process-only count; checkpointed physical counters still exclude discarded interrupted work; predictions and evaluation errors are unchanged",
    }
    _write_json(root / "teacher_cache_audit.json", audit)
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "status",
                    "verified_teacher_episodes",
                    "corrected_teacher_logical_forecast_calls",
                    "correction_required",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
