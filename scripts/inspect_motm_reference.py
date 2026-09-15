"""Check prepared MoTM source integrity, syntax and dependency availability."""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference_root.resolve()
    output = root / "compatibility_audit.json"
    if output.exists():
        raise ValueError("preserve completed source compatibility audit")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("complete source preparation before auditing compatibility")
    failures, imports, checked = [], {}, 0
    for record in manifest["files"]:
        path = root / record["path"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("prepared source or weights changed")
        if path.suffix != ".py":
            continue
        checked += 1
        try:
            parsed = ast.parse(data.decode("utf-8"), filename=record["path"])
        except SyntaxError as error:
            failures.append({"path": record["path"], "line": error.lineno, "message": error.msg})
            continue
        for node in ast.walk(parsed):
            names = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            for name in names:
                imports.setdefault(name, set()).add(record["path"])
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch",
            "numpy",
            "pandas",
            "einops",
            "einx",
            "omegaconf",
            "hydra",
            "sklearn",
            "matplotlib",
            "seaborn",
        )
    }
    result = {
        "status": "audited",
        "runtime_python": sys.version,
        "declared_python_requirement": ">=3.12",
        "verified_source_files": len(manifest["files"]),
        "python_files_checked": checked,
        "syntax_failures": failures,
        "available_dependencies": dependencies,
        "source_imports": {name: sorted(paths) for name, paths in sorted(imports.items())},
        "source_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "model_inference_status": "not_run",
        "interpretation": "static compatibility and installed-module discovery only; successful parsing does not verify checkpoint loading or numerical equivalence",
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "python_files_checked",
                    "syntax_failures",
                    "available_dependencies",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
