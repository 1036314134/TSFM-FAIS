"""Prepare private configuration dependencies and inspect pinned MoTM checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    reference, output = args.reference_root.resolve(), args.output_root.resolve()
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("the pinned source and weights must be fully prepared")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("preserve completed runtime preparation")
    deps = output / "dependencies"
    identity = {
        "reference_manifest_sha256": sha256(reference / "manifest.json"),
        "base_python": sys.executable,
        "packages": ["omegaconf==2.3.0", "antlr4-python3-runtime==4.9.3"],
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("private runtime identity changed")
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    if (
        not (deps / "omegaconf/__init__.py").is_file()
        or not (deps / "antlr4/__init__.py").is_file()
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(deps),
                "--upgrade",
                "--no-deps",
                "--no-compile",
                "--no-build-isolation",
                "--disable-pip-version-check",
                "--cache-dir",
                str(output / "package-cache"),
                "--report",
                str(output / "pip_report.json"),
                *identity["packages"],
            ],
            check=True,
        )
    sys.path.insert(0, str(deps))
    import omegaconf
    import torch
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    torch.set_num_threads(1)
    allowed = [
        DictConfig,
        ListConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        Any,
        defaultdict,
        dict,
        list,
        int,
    ]
    checkpoints = []
    for record in manifest["files"]:
        if not record["path"].endswith("/ckpt/best.pt"):
            continue
        path = reference / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("published checkpoint changed")
        with torch.serialization.safe_globals(allowed):
            saved = torch.load(path, map_location="cpu", weights_only=True)
        config = saved["cfg_inr"]
        root = config._get_root() if OmegaConf.is_config(config) else config
        root_plain = (
            OmegaConf.to_container(root, resolve=False) if OmegaConf.is_config(root) else root
        )
        config_plain = (
            OmegaConf.to_container(config, resolve=False) if OmegaConf.is_config(config) else config
        )
        if config_plain.get("_target_") != "src.modules.inr.ModulatedFourierFeatures":
            raise ValueError("unexpected published model class")
        state = saved["inr"]
        if not isinstance(state, dict) or not all(
            isinstance(value, torch.Tensor) for value in state.values()
        ):
            raise ValueError("checkpoint must contain a tensor state dictionary")
        checkpoints.append(
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "keys": sorted(saved),
                "inr_config": config_plain,
                "root_config_keys": sorted(root_plain),
                "saved_optim_config": root_plain.get("optim"),
                "saved_data_name": root_plain.get("data", {}).get("name"),
                "tensor_count": len(state),
                "parameter_values": sum(value.numel() for value in state.values()),
            }
        )
    result = {
        "status": "completed",
        "capability": "private_dependencies_and_checkpoint_metadata",
        "identity": identity,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "omegaconf_version": omegaconf.__version__,
        "checkpoints": checkpoints,
        "shared_environment_modified": False,
        "checkpoint_loading": "weights_only with explicitly listed configuration classes",
        "model_inference_status": "not_run",
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "completed",
                "checkpoints": len(checkpoints),
                "original_optim_recovered": [
                    row["saved_optim_config"] is not None for row in checkpoints
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
