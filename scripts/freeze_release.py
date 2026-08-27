"""Create a verified, non-overwriting snapshot of experiment artifacts.

The destination is created from scratch. A resumed run may reuse byte-identical
files from a snapshot created by this script, but it never replaces payload
files. Source files are only opened for reading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATE_FILE = ".freeze-state.json"
BUFFER_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class PayloadFile:
    source: Path
    destination_relative: Path
    group: str
    size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True)
class FileRecord:
    group: str
    relative_path: str
    size_bytes: int
    modified_time_ns: int
    sha256: str
    reused: bool


@dataclass(frozen=True)
class ExternalFileRecord:
    name: str
    relative_path: str
    size_bytes: int
    modified_time_ns: int
    sha256: str
    is_symlink: bool


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy and SHA-256 verify a frozen experiment release."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--artifact-source", default="artifacts")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--repository-commit",
        help="Archive tracked source from this immutable Git commit instead of the workspace.",
    )
    parser.add_argument(
        "--paper",
        action="append",
        default=[],
        help="Additional workspace paper file to preserve; may be repeated.",
    )
    parser.add_argument(
        "--external-input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Record an external input without copying it; may be repeated.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-free-factor", type=float, default=1.10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--augment-external-ledger",
        action="store_true",
        help="Add verified external-input hashes to an already completed release.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def _git_metadata(repo_root: Path, repository_commit: str | None) -> dict[str, Any]:
    active_commit = _run_git(repo_root, "rev-parse", "HEAD").strip()
    branch = _run_git(repo_root, "branch", "--show-current").strip()
    status_lines = tuple(
        line for line in _run_git(repo_root, "status", "--porcelain=v1").splitlines() if line
    )
    tracked_dirty = tuple(line for line in status_lines if not line.startswith("??"))
    if tracked_dirty and repository_commit is None:
        details = "\n".join(tracked_dirty)
        raise RuntimeError(
            "tracked workspace files are modified; freeze from a clean code state:\n" + details
        )
    snapshot_commit = active_commit
    source_mode = "tracked_workspace"
    if repository_commit is not None:
        snapshot_commit = _run_git(
            repo_root, "rev-parse", "--verify", f"{repository_commit}^{{commit}}"
        ).strip()
        source_mode = "git_commit_archive"
    return {
        "commit": snapshot_commit,
        "active_workspace_commit": active_commit,
        "branch": branch,
        "status_porcelain": list(status_lines),
        "tracked_workspace_clean": not tracked_dirty,
        "source_mode": source_mode,
    }


def _tracked_files(repo_root: Path) -> tuple[Path, ...]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    paths = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"tracked file is absent from the workspace: {relative}")
        paths.append(relative)
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _regular_files(root: Path) -> tuple[Path, ...]:
    paths = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symbolic links are not supported in a frozen payload: {path}")
        if path.is_file():
            paths.append(path.relative_to(root))
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _payload_file(source: Path, destination_relative: Path, group: str) -> PayloadFile:
    stat = source.stat()
    return PayloadFile(
        source=source,
        destination_relative=destination_relative,
        group=group,
        size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
    )


def _build_payload(
    repo_root: Path,
    artifact_source: Path,
    paper_paths: tuple[Path, ...],
    repository_archive: Path | None,
    repository_commit: str | None,
) -> tuple[tuple[PayloadFile, ...], tuple[str, ...]]:
    artifact_relatives = _regular_files(artifact_source)
    payload = [
        _payload_file(
            artifact_source / relative,
            Path("artifacts") / relative,
            "artifacts",
        )
        for relative in artifact_relatives
    ]
    if repository_archive is not None:
        if repository_commit is None:
            raise ValueError("repository commit is required with a repository archive")
        payload.append(
            _payload_file(
                repository_archive,
                Path("repository") / f"source-{repository_commit[:12]}.tar",
                "repository_archive",
            )
        )
    else:
        for relative in _tracked_files(repo_root):
            payload.append(
                _payload_file(
                    repo_root / relative,
                    Path("repository") / relative,
                    "repository",
                )
            )

    utility_source = Path(__file__).resolve()
    try:
        utility_relative = utility_source.relative_to(repo_root)
    except ValueError:
        utility_relative = None
    if utility_relative is not None:
        utility_destination = Path("repository") / utility_relative
        if all(item.destination_relative != utility_destination for item in payload):
            payload.append(
                _payload_file(
                    utility_source,
                    utility_destination,
                    "freeze_utility",
                )
            )

    paper_destinations: set[Path] = set()
    for paper in paper_paths:
        source = paper if paper.is_absolute() else repo_root / paper
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"paper file does not exist: {source}")
        destination = Path("paper") / source.name
        if destination in paper_destinations:
            raise ValueError(f"duplicate paper destination name: {source.name}")
        paper_destinations.add(destination)
        payload.append(_payload_file(source, destination, "paper"))

    destination_paths = [item.destination_relative.as_posix() for item in payload]
    if len(destination_paths) != len(set(destination_paths)):
        raise RuntimeError("payload contains duplicate destination paths")
    payload.sort(key=lambda item: item.destination_relative.as_posix())
    return tuple(payload), tuple(relative.as_posix() for relative in artifact_relatives)


def _create_repository_archive(
    repo_root: Path,
    repository_commit: str,
    temporary_root: Path,
) -> Path:
    archive = temporary_root / f"source-{repository_commit[:12]}.tar"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--output={archive}",
            repository_commit,
        ],
        cwd=repo_root,
        check=True,
    )
    commit_timestamp = int(
        _run_git(repo_root, "show", "-s", "--format=%ct", repository_commit).strip()
    )
    os.utime(archive, (commit_timestamp, commit_timestamp))
    return archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"no existing ancestor for destination: {path}")
        candidate = candidate.parent
    return candidate


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_or_create_destination(
    destination: Path,
    *,
    release_id: str,
    repo_root: Path,
    artifact_source: Path,
    resume: bool,
) -> dict[str, Any]:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "repo_root": str(repo_root),
        "artifact_source": str(artifact_source),
        "destination": str(destination),
    }
    state_path = destination / STATE_FILE
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"destination already exists; refusing to write or replace files: {destination}"
            )
        if not state_path.is_file():
            raise RuntimeError(f"resume destination lacks {STATE_FILE}: {destination}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(
                    f"resume state mismatch for {key}: {state.get(key)!r} != {value!r}"
                )
        if state.get("status") == "completed":
            raise RuntimeError(f"snapshot is already completed: {destination}")
        return state

    destination.mkdir(parents=True, exist_ok=False)
    state = {
        **expected,
        "status": "copying",
        "started_at": _utc_now(),
    }
    _write_json(state_path, state)
    return state


def _copy_and_verify(item: PayloadFile, destination_root: Path) -> FileRecord:
    destination = destination_root / item.destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    reused = False
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"payload destination exists and is not a file: {destination}")
        if destination.stat().st_size != item.size_bytes:
            raise RuntimeError(f"existing payload has a different size; refusing overwrite: {destination}")
        source_sha256 = _sha256(item.source)
        destination_sha256 = _sha256(destination)
        if source_sha256 != destination_sha256:
            raise RuntimeError(f"existing payload differs; refusing overwrite: {destination}")
        reused = True
    else:
        shutil.copy2(item.source, destination)
        source_sha256 = _sha256(item.source)
        destination_sha256 = _sha256(destination)
        if source_sha256 != destination_sha256:
            raise RuntimeError(f"copied file failed SHA-256 verification: {destination}")

    source_stat = item.source.stat()
    if (
        source_stat.st_size != item.size_bytes
        or source_stat.st_mtime_ns != item.modified_time_ns
    ):
        raise RuntimeError(f"source file changed during the snapshot: {item.source}")
    if destination.stat().st_size != item.size_bytes:
        raise RuntimeError(f"destination size changed during the snapshot: {destination}")
    return FileRecord(
        group=item.group,
        relative_path=item.destination_relative.as_posix(),
        size_bytes=item.size_bytes,
        modified_time_ns=item.modified_time_ns,
        sha256=source_sha256,
        reused=reused,
    )


def _copy_payload(
    payload: tuple[PayloadFile, ...], destination: Path, workers: int
) -> tuple[FileRecord, ...]:
    if workers < 1:
        raise ValueError("workers must be positive")
    records = []
    started = time.monotonic()
    total_bytes = sum(item.size_bytes for item in payload)
    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, record in enumerate(
            executor.map(lambda item: _copy_and_verify(item, destination), payload),
            start=1,
        ):
            records.append(record)
            copied_bytes += record.size_bytes
            if index % 500 == 0 or index == len(payload):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    json.dumps(
                        {
                            "files_verified": index,
                            "files_total": len(payload),
                            "gib_verified": round(copied_bytes / (1024**3), 3),
                            "gib_total": round(total_bytes / (1024**3), 3),
                            "elapsed_seconds": round(elapsed, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    records.sort(key=lambda item: item.relative_path)
    return tuple(records)


def _write_file_ledger(destination: Path, records: tuple[FileRecord, ...]) -> tuple[str, str]:
    ledger_path = destination / "files.sha256.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "group",
                "relative_path",
                "size_bytes",
                "modified_time_ns",
                "sha256",
                "reused",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "group": record.group,
                    "relative_path": record.relative_path,
                    "size_bytes": record.size_bytes,
                    "modified_time_ns": record.modified_time_ns,
                    "sha256": record.sha256,
                    "reused": str(record.reused).lower(),
                }
            )

    tree_digest = hashlib.sha256()
    for record in records:
        tree_digest.update(
            f"{record.sha256}\0{record.size_bytes}\0{record.relative_path}\n".encode()
        )
    return _sha256(ledger_path), tree_digest.hexdigest()


def _payload_tree_from_ledger(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    size_bytes = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path = str(row["relative_path"])
            file_size = int(row["size_bytes"])
            file_sha256 = str(row["sha256"])
            digest.update(
                f"{file_sha256}\0{file_size}\0{relative_path}\n".encode()
            )
            count += 1
            size_bytes += file_size
    return count, size_bytes, digest.hexdigest()


def _formal_selector_evaluations(artifact_source: Path) -> list[str]:
    prefixes = (
        "main-selector-alors-sequence-eval-",
        "main-selector-bfais-sequence-eval-",
        "main-selector-dselect1-sequence-eval-",
        "main-selector-hybrid_lstm-sequence-eval-",
        "main-selector-metaod-sequence-eval-",
        "main-selector-neuralucb-sequence-eval-",
        "main-selector-random_valid_block-sequence-eval-",
    )
    return sorted(
        f"artifacts/{path.name}"
        for path in artifact_source.iterdir()
        if path.is_dir()
        and path.name.endswith("-v2")
        and path.name.startswith(prefixes)
    )


def _artifact_roles(release_id: str, artifact_source: Path) -> str:
    roles = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "snapshot_scope": "complete_artifacts_directory",
        "canonical_publication_dependencies": {
            "data_audits": [
                "artifacts/data-audit-main-seq96-opt9-v1.json",
                "artifacts/main-selector-baselines-data-audit-v1.json",
            ],
            "imputer_fit": ["artifacts/main-seq96-opt13-fit-v1"],
            "teacher_labels": [
                "artifacts/main-seq96-opt23-labels-merged-rolling-b128-v10",
                "artifacts/main-selector-sequence-labels-v1",
            ],
            "routers": [
                "artifacts/main-seq96-opt34-router-consensus-model-prior-correlated8-b128-v20",
                "artifacts/main-seq96-opt38-router-consensus-targetwise-times-b128-v24",
                "artifacts/main-selector-sequence-routers-v1",
            ],
            "candidate_sources": [
                "artifacts/main-seq96-opt24-impute-chronos2-consensus-b128-v10",
                "artifacts/main-seq96-opt35-impute-timesfm2p5-consensus-prior-reg08-b128-v21",
            ],
            "final_bfais_imputations": [
                "artifacts/main-seq96-opt115-impute-chronos2-seasonal090-linear075-b128-v100",
                "artifacts/main-seq96-opt99-impute-timesfm2p5-margin005-seasonal010-b128-v84",
                "artifacts/main-selector-bfais-sequence-impute-chronos2-v1",
                "artifacts/main-selector-bfais-sequence-impute-timesfm2p5-v1",
            ],
            "final_bfais_evaluations": [
                "artifacts/main-seq96-opt115-eval-chronos2-seasonal090-linear075-b128-v100",
                "artifacts/main-seq96-opt99-eval-timesfm2p5-margin005-seasonal010-b128-v84",
            ],
            "formal_selector_evaluations": _formal_selector_evaluations(artifact_source),
            "formal_summary": ["artifacts/main-selector-sequence-comparison-v2"],
        },
        "notes": [
            "All other paths under artifacts are retained as historical or engineering evidence.",
            "Absolute paths embedded in manifests are preserved verbatim in the copied payload.",
            "This role map identifies provenance; it does not change any result status.",
        ],
    }
    # JSON is a valid YAML 1.2 document and avoids a runtime dependency for this utility.
    return json.dumps(roles, indent=2, ensure_ascii=False) + "\n"


def _parse_external_inputs(values: list[str]) -> list[dict[str, Any]]:
    inputs = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"external input must use NAME=PATH: {value!r}")
        path = Path(raw_path).resolve()
        entry: dict[str, Any] = {
            "name": name.strip(),
            "path": str(path),
            "exists": path.exists(),
            "copied": False,
        }
        if path.exists():
            stat = path.stat()
            entry["kind"] = "directory" if path.is_dir() else "file"
            entry["modified_time_ns"] = stat.st_mtime_ns
            if path.is_file():
                entry["size_bytes"] = stat.st_size
                entry["sha256"] = _sha256(path)
            else:
                files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
                entry["file_count"] = len(files)
                entry["size_bytes"] = sum(candidate.stat().st_size for candidate in files)
        inputs.append(entry)
    return inputs


def _external_specs(values: list[str]) -> tuple[tuple[str, Path], ...]:
    specs = []
    names: set[str] = set()
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw_path.strip():
            raise ValueError(f"external input must use NAME=PATH: {value!r}")
        if name in names:
            raise ValueError(f"duplicate external input name: {name}")
        names.add(name)
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"external input does not exist: {path}")
        specs.append((name, path))
    if not specs:
        raise ValueError("at least one external input is required")
    return tuple(specs)


def _external_files(name: str, root: Path) -> tuple[tuple[Path, str], ...]:
    if root.is_file():
        return ((root, root.name),)
    paths = []
    for path in root.rglob("*"):
        if path.is_file():
            paths.append((path, path.relative_to(root).as_posix()))
    paths.sort(key=lambda item: item[1])
    if not paths:
        raise RuntimeError(f"external input contains no files: {name}={root}")
    return tuple(paths)


def _hash_external_file(name: str, path: Path, relative_path: str) -> ExternalFileRecord:
    stat = path.stat()
    size_bytes = stat.st_size
    modified_time_ns = stat.st_mtime_ns
    file_sha256 = _sha256(path)
    final_stat = path.stat()
    if final_stat.st_size != size_bytes or final_stat.st_mtime_ns != modified_time_ns:
        raise RuntimeError(f"external input changed while hashing: {path}")
    return ExternalFileRecord(
        name=name,
        relative_path=relative_path,
        size_bytes=size_bytes,
        modified_time_ns=modified_time_ns,
        sha256=file_sha256,
        is_symlink=path.is_symlink(),
    )


def _hash_external_inputs(
    specs: tuple[tuple[str, Path], ...], workers: int
) -> tuple[tuple[ExternalFileRecord, ...], list[dict[str, Any]]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    initial_files = {name: _external_files(name, path) for name, path in specs}
    tasks = [
        (name, path, relative_path)
        for name, _ in specs
        for path, relative_path in initial_files[name]
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = tuple(
            executor.map(
                lambda task: _hash_external_file(task[0], task[1], task[2]),
                tasks,
            )
        )
    records = tuple(sorted(records, key=lambda item: (item.name, item.relative_path)))

    summaries = []
    for name, root in specs:
        final_files = _external_files(name, root)
        if tuple(relative for _, relative in final_files) != tuple(
            relative for _, relative in initial_files[name]
        ):
            raise RuntimeError(f"external input file set changed while hashing: {root}")
        selected = tuple(record for record in records if record.name == name)
        digest = hashlib.sha256()
        for record in selected:
            digest.update(
                (
                    f"{record.sha256}\0{record.size_bytes}\0"
                    f"{record.relative_path}\n"
                ).encode()
            )
        summaries.append(
            {
                "name": name,
                "path": str(root),
                "exists": True,
                "copied": False,
                "kind": "directory" if root.is_dir() else "file",
                "file_count": len(selected),
                "size_bytes": sum(record.size_bytes for record in selected),
                "tree_sha256": digest.hexdigest(),
            }
        )
    return records, summaries


def _external_ledger_bytes(records: tuple[ExternalFileRecord, ...]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "name",
            "relative_path",
            "size_bytes",
            "modified_time_ns",
            "sha256",
            "is_symlink",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "name": record.name,
                "relative_path": record.relative_path,
                "size_bytes": record.size_bytes,
                "modified_time_ns": record.modified_time_ns,
                "sha256": record.sha256,
                "is_symlink": str(record.is_symlink).lower(),
            }
        )
    return buffer.getvalue().encode("utf-8")


def _expected_freeze_sha256(destination: Path) -> str:
    line = (destination / "FREEZE.sha256").read_text(encoding="ascii").strip()
    expected, separator, filename = line.partition("  ")
    if not separator or filename != "FREEZE.json":
        raise RuntimeError("invalid FREEZE.sha256 format")
    return expected


def _augment_external_ledger(
    destination: Path,
    release_id: str,
    external_values: list[str],
    workers: int,
) -> int:
    freeze_path = destination / "FREEZE.json"
    state_path = destination / STATE_FILE
    ledger_path = destination / "files.sha256.csv"
    if not freeze_path.is_file() or not state_path.is_file() or not ledger_path.is_file():
        raise FileNotFoundError("destination is not a completed freeze release")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "completed" or state.get("status") != "completed":
        raise RuntimeError("external hashes can only augment a completed release")
    if freeze.get("release_id") != release_id or state.get("release_id") != release_id:
        raise RuntimeError("release ID does not match completed destination")
    previous_freeze_sha256 = _expected_freeze_sha256(destination)
    if _sha256(freeze_path) != previous_freeze_sha256:
        raise RuntimeError("FREEZE.json does not match FREEZE.sha256")

    payload = freeze["payload"]
    ledger_sha256 = _sha256(ledger_path)
    if ledger_sha256 != payload["ledger_sha256"]:
        raise RuntimeError("main payload ledger SHA-256 changed")
    file_count, size_bytes, tree_sha256 = _payload_tree_from_ledger(ledger_path)
    if (
        file_count != payload["file_count"]
        or size_bytes != payload["size_bytes"]
        or tree_sha256 != payload["tree_sha256"]
    ):
        raise RuntimeError("main payload ledger no longer matches FREEZE.json")

    specs = _external_specs(external_values)
    records, summaries = _hash_external_inputs(specs, workers)
    ledger_bytes = _external_ledger_bytes(records)
    external_ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
    external_ledger_path = destination / "external_inputs.sha256.csv"
    if external_ledger_path.exists():
        if _sha256(external_ledger_path) != external_ledger_sha256:
            raise RuntimeError("existing external-input ledger differs; refusing overwrite")
    else:
        external_ledger_path.write_bytes(ledger_bytes)

    metadata = freeze.setdefault("metadata", {})
    if (
        metadata.get("external_inputs_ledger") == external_ledger_path.name
        and metadata.get("external_inputs_ledger_sha256") == external_ledger_sha256
        and freeze.get("external_inputs") == summaries
    ):
        print(
            json.dumps(
                {
                    "status": "already_augmented",
                    "release_id": release_id,
                    "payload_tree_sha256": tree_sha256,
                    "payload_tree_unchanged": True,
                    "external_file_count": len(records),
                    "external_inputs_ledger_sha256": external_ledger_sha256,
                    "freeze_sha256": previous_freeze_sha256,
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    amended_at = _utc_now()
    metadata.update(
        {
            "revision": int(metadata.get("revision", 1)) + 1,
            "previous_freeze_sha256": previous_freeze_sha256,
            "external_inputs_ledger": external_ledger_path.name,
            "external_inputs_ledger_sha256": external_ledger_sha256,
        }
    )
    freeze["external_inputs"] = summaries
    freeze["amended_at"] = amended_at
    freeze["payload"]["tree_sha256"] = tree_sha256
    _write_json(freeze_path, freeze)
    freeze_sha256 = _sha256(freeze_path)
    (destination / "FREEZE.sha256").write_text(
        f"{freeze_sha256}  FREEZE.json\n", encoding="ascii"
    )
    state.update(
        {
            "updated_at": amended_at,
            "freeze_sha256": freeze_sha256,
            "payload_tree_sha256": tree_sha256,
            "external_inputs_ledger_sha256": external_ledger_sha256,
        }
    )
    _write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": "completed",
                "release_id": release_id,
                "payload_tree_sha256": tree_sha256,
                "payload_tree_unchanged": True,
                "external_file_count": len(records),
                "external_inputs": summaries,
                "external_inputs_ledger_sha256": external_ledger_sha256,
                "freeze_sha256": freeze_sha256,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _verify_artifact_source_unchanged(
    artifact_source: Path, initial_relatives: tuple[str, ...]
) -> None:
    final_relatives = tuple(path.as_posix() for path in _regular_files(artifact_source))
    if final_relatives != initial_relatives:
        raise RuntimeError("artifact source file set changed during the snapshot")


def main() -> int:
    args = _arguments()
    repo_root = Path(args.repo_root).resolve()
    artifact_source = Path(args.artifact_source)
    if not artifact_source.is_absolute():
        artifact_source = repo_root / artifact_source
    artifact_source = artifact_source.resolve()
    destination = Path(args.destination).resolve()

    if not repo_root.is_dir() or not artifact_source.is_dir():
        raise FileNotFoundError("repo root and artifact source must be existing directories")
    if artifact_source == destination or artifact_source in destination.parents:
        raise ValueError("destination must be outside the artifact source")
    if destination in artifact_source.parents:
        raise ValueError("artifact source must not be inside the destination")
    if args.augment_external_ledger:
        if args.plan_only or args.resume:
            raise ValueError(
                "--augment-external-ledger cannot be combined with --plan-only or --resume"
            )
        return _augment_external_ledger(
            destination,
            args.release_id,
            args.external_input,
            args.workers,
        )

    git_metadata = _git_metadata(repo_root, args.repository_commit)
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    repository_archive = None
    if args.repository_commit is not None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="tsfm-fais-freeze-")
        repository_archive = _create_repository_archive(
            repo_root,
            str(git_metadata["commit"]),
            Path(temporary_directory.name),
        )
    payload, artifact_relatives = _build_payload(
        repo_root,
        artifact_source,
        tuple(Path(path) for path in args.paper),
        repository_archive,
        str(git_metadata["commit"]) if repository_archive is not None else None,
    )
    payload_bytes = sum(item.size_bytes for item in payload)
    external_inputs = _parse_external_inputs(args.external_input)
    existing_parent = _nearest_existing_parent(destination.parent)
    free_bytes = shutil.disk_usage(existing_parent).free
    required_bytes = int(payload_bytes * args.minimum_free_factor)
    plan = {
        "release_id": args.release_id,
        "repo_root": str(repo_root),
        "artifact_source": str(artifact_source),
        "destination": str(destination),
        "payload_file_count": len(payload),
        "payload_size_bytes": payload_bytes,
        "payload_size_gib": round(payload_bytes / (1024**3), 3),
        "free_size_bytes": free_bytes,
        "free_size_gib": round(free_bytes / (1024**3), 3),
        "minimum_free_factor": args.minimum_free_factor,
        "required_size_bytes": required_bytes,
        "git": git_metadata,
        "external_inputs": external_inputs,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if free_bytes < required_bytes and not args.resume:
        raise RuntimeError(
            f"insufficient free space: need {required_bytes} bytes, have {free_bytes}"
        )
    if args.plan_only:
        return 0

    state = _load_or_create_destination(
        destination,
        release_id=args.release_id,
        repo_root=repo_root,
        artifact_source=artifact_source,
        resume=args.resume,
    )
    started = time.monotonic()
    records = _copy_payload(payload, destination, args.workers)
    _verify_artifact_source_unchanged(artifact_source, artifact_relatives)

    ledger_sha256, tree_sha256 = _write_file_ledger(destination, records)
    roles_path = destination / "artifact_roles.yaml"
    roles_path.write_text(
        _artifact_roles(args.release_id, artifact_source),
        encoding="utf-8",
    )
    elapsed_seconds = time.monotonic() - started
    group_summary: dict[str, dict[str, int]] = {}
    for record in records:
        summary = group_summary.setdefault(record.group, {"file_count": 0, "size_bytes": 0})
        summary["file_count"] += 1
        summary["size_bytes"] += record.size_bytes

    freeze = {
        "schema_version": SCHEMA_VERSION,
        "release_id": args.release_id,
        "status": "completed",
        "created_at": state["started_at"],
        "completed_at": _utc_now(),
        "source_is_read_only_by_protocol": True,
        "source_files_modified": 0,
        "copy_policy": {
            "destination_created_for_release": True,
            "payload_overwrite_allowed": False,
            "resume_requires_identical_existing_payload": True,
            "permissions_changed": False,
        },
        "source": {
            "repo_root": str(repo_root),
            "artifact_source": str(artifact_source),
            "git": git_metadata,
        },
        "destination": str(destination),
        "payload": {
            "file_count": len(records),
            "size_bytes": sum(record.size_bytes for record in records),
            "groups": group_summary,
            "ledger": "files.sha256.csv",
            "ledger_sha256": ledger_sha256,
            "tree_sha256": tree_sha256,
            "all_source_destination_hashes_match": True,
            "reused_file_count": sum(record.reused for record in records),
        },
        "metadata": {
            "artifact_roles": "artifact_roles.yaml",
            "artifact_roles_sha256": _sha256(roles_path),
        },
        "external_inputs": external_inputs,
        "elapsed_seconds": elapsed_seconds,
    }
    freeze_path = destination / "FREEZE.json"
    _write_json(freeze_path, freeze)
    freeze_sha256 = _sha256(freeze_path)
    (destination / "FREEZE.sha256").write_text(
        f"{freeze_sha256}  FREEZE.json\n", encoding="ascii"
    )
    state.update(
        {
            "status": "completed",
            "completed_at": freeze["completed_at"],
            "tree_sha256": tree_sha256,
            "freeze_sha256": freeze_sha256,
        }
    )
    _write_json(destination / STATE_FILE, state)
    print(
        json.dumps(
            {
                "status": "completed",
                "destination": str(destination),
                "file_count": len(records),
                "size_bytes": freeze["payload"]["size_bytes"],
                "tree_sha256": tree_sha256,
                "freeze_sha256": freeze_sha256,
                "elapsed_seconds": round(elapsed_seconds, 1),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if temporary_directory is not None:
        temporary_directory.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
