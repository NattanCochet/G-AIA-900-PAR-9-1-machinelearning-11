"""Checks the code version stored with every result (common.git_commit)."""

import shutil
import subprocess

import pytest

from common import git_commit

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def run_git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@test", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repository with one script and one results CSV, both committed."""
    run_git(tmp_path, "init")
    (tmp_path / "results").mkdir()
    (tmp_path / "script.py").write_text("print('v1')\n")
    (tmp_path / "results" / "log.csv").write_text("score\n")
    run_git(tmp_path, "add", ".")
    run_git(tmp_path, "commit", "-m", "v1")
    return tmp_path


def test_clean_checkout_gives_the_commit_hash(repo):
    assert git_commit(repo) == run_git(repo, "rev-parse", "--short", "HEAD")


def test_logging_a_result_does_not_make_the_code_dirty(repo):
    (repo / "results" / "log.csv").write_text("score\n0.97\n")
    assert not git_commit(repo).endswith("-dirty")


def test_modified_script_is_flagged_dirty(repo):
    (repo / "script.py").write_text("print('v2')\n")
    assert git_commit(repo).endswith("-dirty")


def test_outside_a_repository_gives_none(tmp_path):
    assert git_commit(tmp_path) is None
