"""Source provenance must never read protected data, and must not hide a dirty tree.

Everything here runs against a synthetic temporary tree. No real protected file
is opened, enumerated, or hashed by these tests.
"""

from __future__ import annotations

import pathlib
import subprocess
from pathlib import Path

import pytest

from interp.provenance import (
    PROTECTED_DIRECTORY_NAMES,
    is_protected_path,
    snapshot_digest,
    snapshot_files,
    source_revision,
)


def _tree(root: Path) -> Path:
    """A synthetic project: allowlisted sources plus a protected config directory."""

    (root / "src" / "interp").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "configs" / "protected").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "uv.lock").write_text("lock\n")
    (root / "src" / "interp" / "a.py").write_text("A = 1\n")
    (root / "scripts" / "run.py").write_text("print(1)\n")
    (root / "configs" / "train.yaml").write_text("version: 1\n")
    # A protected artifact that matches the configs glob: exactly the file the
    # old broad rglob would have opened and hashed.
    (root / "configs" / "protected" / "held_out_features.yaml").write_text(
        "SYNTHETIC-NOT-REAL-PROTECTED-CONTENT-v1\n"
    )
    return root


def test_snapshot_excludes_protected_directories(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    collected = snapshot_files(root)
    assert all(not is_protected_path(path, root) for path in collected)
    assert all("protected" not in path.parts for path in collected)
    names = {path.name for path in collected}
    assert names == {"pyproject.toml", "uv.lock", "a.py", "run.py", "train.yaml"}


def test_changing_a_protected_file_does_not_change_the_revision(tmp_path: Path) -> None:
    """Requirement 1: protected content cannot influence the source hash."""

    root = _tree(tmp_path)
    before = snapshot_digest(root)
    (root / "configs" / "protected" / "held_out_features.yaml").write_text(
        "COMPLETELY-DIFFERENT-SYNTHETIC-CONTENT\n" * 50
    )
    (root / "configs" / "protected" / "another.yaml").write_text("more synthetic\n")
    assert snapshot_digest(root) == before

    # ...while a change to an allowlisted file does move it, so the hash is not
    # simply insensitive to everything.
    (root / "src" / "interp" / "a.py").write_text("A = 2\n")
    assert snapshot_digest(root) != before


def test_protected_files_are_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 2: the bytes are not read, not merely excluded from the hash."""

    root = _tree(tmp_path)
    opened: list[Path] = []
    original = pathlib.Path.read_bytes

    def recording_read_bytes(self: Path) -> bytes:
        opened.append(Path(self))
        return original(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", recording_read_bytes)
    snapshot_digest(root)

    assert opened, "the snapshot must actually read the allowlisted files"
    assert not [path for path in opened if "protected" in path.parts]


def test_a_new_unlisted_directory_cannot_enter_the_hash(tmp_path: Path) -> None:
    """The allowlist is the point: adding a tree must not silently add bytes."""

    root = _tree(tmp_path)
    before = snapshot_digest(root)
    (root / "notebooks").mkdir()
    (root / "notebooks" / "scratch.py").write_text("whatever\n")
    assert snapshot_digest(root) == before


def test_is_protected_path_rejects_paths_outside_the_root(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert is_protected_path(Path("/etc/passwd"), root)
    assert is_protected_path(root / "configs" / "protected" / "x.yaml", root)
    assert not is_protected_path(root / "configs" / "train.yaml", root)
    assert "protected" in PROTECTED_DIRECTORY_NAMES


def test_snapshot_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert snapshot_digest(root) == snapshot_digest(root)
    assert snapshot_files(root) == sorted(snapshot_files(root))


def test_empty_tree_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no source files found"):
        snapshot_digest(tmp_path)


# --------------------------------------------------------------------------
# git dirty-tree handling
# --------------------------------------------------------------------------


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True
    )


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return False
    return True


@pytest.mark.skipif(not _git_available(), reason="git is not installed")
def test_dirty_tree_cannot_masquerade_as_the_clean_commit(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")

    clean = source_revision(root)
    assert clean.startswith("git:")
    assert "+dirty:" not in clean
    commit = clean.removeprefix("git:")

    (root / "src" / "interp" / "a.py").write_text("A = 999\n")
    dirty = source_revision(root)
    assert dirty != clean
    assert dirty.startswith(f"git:{commit}+dirty:")
    # the dirty marker carries the digest of what actually ran
    assert dirty.endswith(snapshot_digest(root))


@pytest.mark.skipif(not _git_available(), reason="git is not installed")
def test_a_dirty_protected_file_does_not_make_the_tree_look_dirty(tmp_path: Path) -> None:
    """Protected paths are excluded from the pathspec, so they are never listed."""

    root = _tree(tmp_path)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    clean = source_revision(root)

    (root / "configs" / "protected" / "held_out_features.yaml").write_text("changed\n")
    assert source_revision(root) == clean


def test_no_repository_falls_back_to_a_snapshot(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    revision = source_revision(root)
    # tmp_path is not inside a repository in normal CI; if it happens to be,
    # the git branch is exercised by the tests above instead.
    if revision.startswith("snapshot-sha256:"):
        assert revision == f"snapshot-sha256:{snapshot_digest(root)}"
