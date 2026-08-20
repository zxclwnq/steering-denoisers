"""Source provenance that works with either Git or a synchronized source snapshot.

Two rules govern this module, and both exist because provenance code runs in
every experiment and therefore touches the filesystem on every run:

1. **It never reads protected data.** Directories named in
   :data:`PROTECTED_DIRECTORY_NAMES` are pruned during the walk, before any file
   is opened. A run cannot claim ``held_out_accessed: false`` while its own
   revision hash was computed by reading held-out bytes.

2. **Uncommitted code cannot masquerade as a clean commit.** A dirty working
   tree gets a revision string that says so and carries a content digest of what
   actually ran.

The snapshot file set is an explicit allowlist, not a broad recursive glob, so
adding a new directory to the repository cannot silently pull new bytes into the
hash.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# Directory names that must never be walked, opened, or hashed here. Matched on
# any path component, so ``configs/protected/`` and any future nesting are both
# excluded. See docs/RESEARCH_GOVERNANCE.md section 2.
PROTECTED_DIRECTORY_NAMES = frozenset({"protected"})

# Explicit allowlist of what constitutes "the source that ran".
SNAPSHOT_FILES = ("pyproject.toml", "uv.lock")
SNAPSHOT_TREES = (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml"))

# Git pathspec restricting dirty-tree detection to the same allowlist, with
# protected paths excluded so their names are never even listed.
_GIT_PATHSPEC = [
    *SNAPSHOT_FILES,
    *(directory for directory, _ in SNAPSHOT_TREES),
    ":(exclude)configs/protected",
]


def is_protected_path(path: Path, root: Path) -> bool:
    """True when any component of ``path`` below ``root`` is a protected directory."""

    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        # Outside the project root: treat as protected rather than guess.
        return True
    return bool(PROTECTED_DIRECTORY_NAMES.intersection(relative.parts))


def snapshot_files(root: Path) -> list[Path]:
    """Collect the allowlisted source files, pruning protected directories.

    Pruning happens during traversal, so a protected file is never stat-ed for
    inclusion and never opened.
    """

    collected: list[Path] = []
    for name in SNAPSHOT_FILES:
        candidate = root / name
        if candidate.is_file():
            collected.append(candidate)

    for directory, pattern in SNAPSHOT_TREES:
        base = root / directory
        if not base.is_dir():
            continue
        stack = [base]
        while stack:
            current = stack.pop()
            for entry in sorted(current.iterdir()):
                if entry.is_dir():
                    if entry.name in PROTECTED_DIRECTORY_NAMES:
                        continue  # pruned: never descended into, never read
                    stack.append(entry)
                elif entry.match(pattern):
                    collected.append(entry)

    # Belt and braces: a path that somehow survived pruning is dropped here,
    # still before any file is opened.
    safe = [path for path in collected if not is_protected_path(path, root)]
    if not safe:
        raise RuntimeError(f"no source files found below {root}")
    return sorted(set(safe))


def snapshot_digest(root: Path) -> str:
    """Deterministic SHA-256 over the allowlisted, protected-free source files."""

    digest = hashlib.sha256()
    for path in snapshot_files(root):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def source_revision(root: Path | None = None) -> str:
    """Return the revision of the source that is about to run.

    ``git:<sha>``                      a clean working tree at that commit;
    ``git:<sha>+dirty:<digest>``       uncommitted changes; the digest identifies
                                       the source that actually ran, so a dirty
                                       tree can never be mistaken for the commit;
    ``snapshot-sha256:<digest>``       no Git repository at all.
    """

    project_root = root or Path(__file__).resolve().parents[2]
    head = _git(project_root, "rev-parse", "HEAD")
    if head.returncode == 0 and head.stdout.strip():
        commit = head.stdout.strip()
        # Restricted to the allowlist and excluding protected paths, so protected
        # filenames are never listed. Only the boolean is used.
        status = _git(project_root, "status", "--porcelain", "--", *_GIT_PATHSPEC)
        if status.returncode != 0:
            raise RuntimeError(
                f"git status failed while checking for a dirty tree: {status.stderr.strip()}"
            )
        if status.stdout.strip():
            return f"git:{commit}+dirty:{snapshot_digest(project_root)}"
        return f"git:{commit}"

    return f"snapshot-sha256:{snapshot_digest(project_root)}"
