"""Download a pinned MoTM reference without changing installed packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def git_blob_sha1(path):
    size = path.stat().st_size
    result = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def checked_destination(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError("source paths must remain inside the reference directory")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("source paths must remain inside the reference directory")
    return target


def checkpoint_globals(path):
    """Inspect pickle instructions as data; never deserialize downloaded objects here."""
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if entry.filename.endswith("/data.pkl")]
        if len(entries) != 1 or entries[0].file_size > 4 * 1024 * 1024:
            raise ValueError("unexpected checkpoint metadata layout")
        code = archive.read(entries[0])
    return sorted(
        {
            argument
            for operation, argument, _ in pickletools.genops(code)
            if operation.name == "GLOBAL"
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-mib-per-second", type=float, default=2.0)
    args = parser.parse_args()
    if not 0 < args.maximum_mib_per_second <= 4:
        parser.error("download rate must be positive and no higher than 4 MiB/s")
    spec = json.loads(args.specification.read_text(encoding="utf-8"))
    if (
        spec["repository"] != "EDF-Lab/MoTM"
        or len(spec["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in spec["commit"])
    ):
        raise ValueError("only the pinned official MoTM repository is supported")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("preserve completed reference preparation")
    started, downloaded, records = time.monotonic(), 0, []
    for entry in spec["files"]:
        target = checked_destination(root, entry["path"])
        if target.exists():
            if (
                target.stat().st_size != entry["size_bytes"]
                or git_blob_sha1(target) != entry["git_blob_sha1"]
            ):
                raise ValueError(f"existing reference file differs: {entry['path']}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            url = (
                "https://raw.githubusercontent.com/EDF-Lab/MoTM/"
                + spec["commit"]
                + "/"
                + urllib.parse.quote(entry["path"], safe="/")
            )
            request = urllib.request.Request(
                url, headers={"User-Agent": "TSFM-FAIS-reference-preparation"}
            )
            count, file_started = 0, time.monotonic()
            with (
                urllib.request.urlopen(request, timeout=30) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(65536):
                    count += len(chunk)
                    if count > entry["size_bytes"]:
                        raise ValueError("download exceeds the pinned blob size")
                    output.write(chunk)
                    delay = count / (args.maximum_mib_per_second * 1024**2) - (
                        time.monotonic() - file_started
                    )
                    if delay > 0:
                        time.sleep(delay)
            if count != entry["size_bytes"] or git_blob_sha1(temporary) != entry["git_blob_sha1"]:
                raise ValueError(f"download does not match the pinned blob: {entry['path']}")
            temporary.replace(target)
            downloaded += count
        record = dict(entry)
        record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        if target.suffix == ".pt":
            record["pickle_globals_without_deserialization"] = checkpoint_globals(target)
        records.append(record)
        print(
            json.dumps(
                {
                    "completed_files": len(records),
                    "total_files": len(spec["files"]),
                    "path": entry["path"],
                }
            ),
            flush=True,
        )
    manifest = {
        "status": "completed",
        "capability": "source_and_weight_preparation_only",
        "model_inference_status": "not_verified",
        "repository": spec["repository"],
        "commit": spec["commit"],
        "specification_sha256": hashlib.sha256(args.specification.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "downloaded_bytes_this_execution": downloaded,
        "elapsed_seconds": time.monotonic() - started,
        "maximum_mib_per_second": args.maximum_mib_per_second,
        "files": records,
        "installed_packages_modified": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
