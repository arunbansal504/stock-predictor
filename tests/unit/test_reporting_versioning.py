from __future__ import annotations

from stockpredictor.reporting.versioning import git_commit_hash, model_version


def test_model_version_is_deterministic():
    assert model_version() == model_version()
    assert model_version().startswith("stacked-ranker-")


def test_git_commit_hash_prefers_github_sha_env_var(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "abc123deadbeef")
    assert git_commit_hash() == "abc123deadbeef"


def test_git_commit_hash_falls_back_to_local_git(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    result = git_commit_hash()
    assert isinstance(result, str)
    assert len(result) == 40
