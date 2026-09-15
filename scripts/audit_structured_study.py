"""Run completed-study audits sequentially under one resource-guarded process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tsfm_fais.utility_experiment import _write_json, file_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "accuracy-root",
        "feature-root",
        "base-root",
        "temporal-root",
        "dependency-root",
        "replay-root",
        "plan",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path)
    parser.add_argument("--previous-readout-root", type=Path)
    parser.add_argument("--feature-audit-root", type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed study audits")
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        name: file_sha256(root / "manifest.json")
        for name, root in (
            ("features", args.feature_root),
            ("base_static", args.base_root),
            ("target_temporal", args.temporal_root),
            ("full_dependency", args.dependency_root),
        )
    }
    steps = [
        (
            "features",
            "audit_structured_preforecast.py",
            ["--accuracy-root", args.accuracy_root, "--input-root", args.feature_root],
        ),
        (
            "target_temporal",
            "audit_preforecast_student.py",
            [
                "--accuracy-root",
                args.accuracy_root,
                "--input-root",
                args.temporal_root,
                "--structured-root",
                args.feature_root,
            ],
        ),
        (
            "full_dependency",
            "audit_preforecast_student.py",
            [
                "--accuracy-root",
                args.accuracy_root,
                "--input-root",
                args.dependency_root,
                "--structured-root",
                args.feature_root,
            ],
        ),
        (
            "readout",
            "summarize_structured_students.py",
            [
                "--accuracy-root",
                args.accuracy_root,
                "--base-root",
                args.base_root,
                "--temporal-root",
                args.temporal_root,
                "--dependency-root",
                args.dependency_root,
                "--replay-root",
                args.replay_root,
                "--plan",
                args.plan,
            ],
        ),
    ]
    inherited_audit = None
    if args.feature_audit_root is not None:
        marker = args.feature_audit_root / "manifest.json"
        result = json.loads(marker.read_text(encoding="utf-8"))
        if (
            result["status"] != "completed"
            or result["source_manifest_sha256"] != sources["features"]
        ):
            raise ValueError("the reusable feature audit belongs to different inputs")
        inherited_audit = {"path": str(marker.resolve()), "sha256": file_sha256(marker)}
        steps = [step for step in steps if step[0] != "features"]
    if args.teacher_root is not None:
        steps.insert(
            0,
            (
                "base_static",
                "audit_preforecast_student.py",
                ["--accuracy-root", args.accuracy_root, "--input-root", args.base_root],
            ),
        )
        for _name, script, arguments in steps:
            if script == "audit_preforecast_student.py":
                arguments.extend(["--teacher-root", args.teacher_root])
    if args.previous_readout_root is not None:
        next(arguments for name, _, arguments in steps if name == "readout").extend(
            ["--previous-readout-root", args.previous_readout_root]
        )
    completed = []
    for name, script, arguments in steps:
        destination = output / name
        marker = destination / "manifest.json"
        if not marker.exists():
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / script),
                *map(str, arguments),
                "--output-root",
                str(destination),
            ]
            with (output / f"{name}.log").open("w", encoding="utf-8") as handle:
                subprocess.run(
                    command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True
                )
        result = json.loads(marker.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            raise ValueError("an audit did not complete")
        if name == "readout":
            if result["source_manifests"] != {
                key: sources[key] for key in ("base_static", "target_temporal", "full_dependency")
            }:
                raise ValueError("the readout references another study")
        elif result["source_manifest_sha256"] != sources[name]:
            raise ValueError("a completed audit references another source")
        completed.append(
            {
                "stage": name,
                "manifest": str(marker.relative_to(output)),
                "sha256": file_sha256(marker),
                "script_sha256": file_sha256(ROOT / "scripts" / script),
            }
        )
        print(json.dumps({"stage": name, "status": "completed"}), flush=True)
    _write_json(
        output / "manifest.json",
        {
            "status": "completed",
            "sources": sources,
            "audits": completed,
            "reused_feature_audit": inherited_audit,
            "script_sha256": file_sha256(Path(__file__)),
        },
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
