import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "motm_reference_preparation",
    Path(__file__).resolve().parents[2] / "scripts/prepare_motm_reference.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reference_paths_cannot_leave_the_output_directory(tmp_path):
    for path in ("../outside", "src/../../outside", "E:/outside", "src\\..\\outside", "/outside"):
        with pytest.raises(ValueError, match="inside"):
            MODULE.checked_destination(tmp_path, path)
    assert MODULE.checked_destination(tmp_path, "src/model.py") == tmp_path / "src/model.py"


def test_reference_digest_includes_git_blob_header(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"x = 1\n")
    assert MODULE.git_blob_sha1(path) == hashlib.sha1(b"blob 6\0x = 1\n").hexdigest()


def test_checkpoint_inspection_never_executes_pickle_globals(tmp_path):
    path = tmp_path / "weights.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"cos\nsystem\n(S'never_execute_this_command'\ntR.")
    assert MODULE.checkpoint_globals(path) == ["os system"]
