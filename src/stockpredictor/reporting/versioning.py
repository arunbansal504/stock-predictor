"""Provenance helpers for published predictions (ML Review Board Part 1/8:
"model_version" and "git_commit_hash" columns).

Neither concept previously existed as a formal value anywhere in the
pipeline -- `config/model.yaml` has no version field, and no code shelled
out to git before now. Both are defined here rather than invented ad hoc at
each call site.
"""

from __future__ import annotations

import hashlib
import os
import subprocess

from stockpredictor.common.config import CONFIG_DIR, REPO_ROOT

_MODEL_CONFIG_PATH = CONFIG_DIR / "model.yaml"


def model_version() -> str:
    """A short, deterministic identifier for "which model configuration
    produced this prediction" -- a hash of `config/model.yaml`'s raw bytes.
    Changes iff the config that actually governs training changes, so it's a
    meaningful diff signal without needing a manually-bumped version field."""
    digest = hashlib.sha256(_MODEL_CONFIG_PATH.read_bytes()).hexdigest()[:12]
    return f"stacked-ranker-{digest}"


def git_commit_hash() -> str:
    """The commit this prediction was produced from. Prefers `GITHUB_SHA`
    (set by every GitHub Actions run, including on `push` events where a
    plain `git rev-parse HEAD` inside an Actions checkout can point at a
    detached merge commit rather than the reviewable commit) and falls back
    to a local `git rev-parse HEAD` for manual/dev runs."""
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
