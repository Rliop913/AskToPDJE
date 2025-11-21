import os
from pathlib import Path
from git import Repo
from git.exc import InvalidGitRepositoryError, NoSuchPathError, GitCommandError


def clone_or_pull(
    url: str,
    dest: str,
    branch: str = "main",
    remote_name: str = "origin",
    verify_remote: bool = True,
):
    dpath = Path(dest)

    if not dpath.exists():
        dpath.parent.mkdir(parents=True, exist_ok=True)
        repo = Repo.clone_from(url, dpath, branch=branch)
        return repo

    try:
        repo = Repo(dest)
    except (InvalidGitRepositoryError, NoSuchPathError):
        raise RuntimeError(f"{dpath} exist but it's not a git repo.")
    try:
        remote = repo.remote(remote_name)
    except ValueError:
        raise RuntimeError(f"remote {remote_name} not found in {dpath}")

    
    try:
        remote.fetch(prune=True)
        if repo.head.is_detached:
            repo.git.checkout(branch)
        repo.git.pull(remote_name, branch)
    except GitCommandError as e:
        raise RuntimeError(f"git operation failed: {e}")
    return repo
