"""Pin and download the official TiRex checkpoint through the installed hf CLI."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("preserve the completed TiRex reference")
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    else:
        info = HfApi().model_info("NX-AI/TiRex", files_metadata=True)
        selected = [
            item
            for item in info.siblings
            if item.rfilename in {"model.ckpt", "README.md", "LICENSE", "LICENSE.md"}
        ]
        checkpoint = next(item for item in selected if item.rfilename == "model.ckpt")
        lfs = checkpoint.lfs
        expected = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
        identity = {
            "repo_id": "NX-AI/TiRex",
            "revision": info.sha,
            "checkpoint_sha256": expected,
            "checkpoint_bytes": checkpoint.size,
            "files": [item.rfilename for item in selected],
            "purpose": "additional frozen-forecaster interface verification; no evaluation result yet",
        }
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "huggingface_hub.cli.hf",
        "download",
        identity["repo_id"],
        *identity["files"],
        "--revision",
        identity["revision"],
        "--local-dir",
        str(output / "model"),
    ]
    with (output / "download.log").open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    checkpoint = output / "model/model.ckpt"
    if (
        sha256(checkpoint) != identity["checkpoint_sha256"]
        or checkpoint.stat().st_size != identity["checkpoint_bytes"]
    ):
        raise ValueError("the downloaded checkpoint differs from the pinned Hub metadata")
    records = [
        {"path": str(Path("model") / name), "sha256": sha256(output / "model" / name)}
        for name in identity["files"]
    ]
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "identity": identity,
                "files": records,
                "script_sha256": sha256(Path(__file__)),
                "cli_entrypoint": "python -m huggingface_hub.cli.hf; existing Windows launcher exits before displaying output",
                "native_interface_verified": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "status": "completed",
                "revision": identity["revision"],
                "checkpoint_bytes": identity["checkpoint_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
