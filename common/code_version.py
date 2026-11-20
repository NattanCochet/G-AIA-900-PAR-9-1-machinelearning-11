"""Identify the code that produced a result, so that rows logged before and
after a change to a script can be told apart in the results CSVs."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit(repo: str | Path = REPO_ROOT) -> str | None:
    """Short hash of the checked-out commit, e.g. "4c5ab66", suffixed with
    "-dirty" when tracked files differ from it.

    results/ is left out of the check: the CSV logs change after every run,
    which would otherwise flag every run but the first as dirty.
    Returns None outside a git checkout, or when git is not installed.
    """

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        )

    try:
        head = git("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return None
        diff = git("diff", "--quiet", "HEAD", "--", ".", ":(exclude)results")
    except OSError:  # git not installed
        return None

    commit = head.stdout.strip()
    # exit code 1 means "differences found"; any other failure counts as dirty
    # too, so that an unverified state is never reported as clean
    return commit if diff.returncode == 0 else f"{commit}-dirty"
