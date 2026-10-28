"""Shared HuggingFace offline-cache utilities.

Used by both training scripts and model modules so that offline HF snapshot
resolution is not duplicated across files.
"""

import os


def resolve_hf_snapshot_path(repo_id: str, cache_dir: str | None = None) -> str:
    """Resolve a HuggingFace repo ID to its local snapshot directory.

    Converts e.g. 'black-forest-labs/FLUX.2-klein-base-9B' to
    '{cache_dir}/models--black-forest-labs--FLUX.2-klein-base-9B/snapshots/{hash}'.
    If repo_id is already a local directory, returns it unchanged.
    """
    if os.path.isdir(repo_id):
        return repo_id

    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        )

    folder_name = "models--" + repo_id.replace("/", "--")
    model_dir = os.path.join(cache_dir, folder_name)
    refs_file = os.path.join(model_dir, "refs", "main")

    if not os.path.isfile(refs_file):
        raise FileNotFoundError(
            f"Cannot find cached model for '{repo_id}'.\n"
            f"Expected refs file at: {refs_file}\n"
            f"Download the model first:\n"
            f"  huggingface-cli download {repo_id}"
        )

    commit_hash = open(refs_file).read().strip()
    snapshot_dir = os.path.join(model_dir, "snapshots", commit_hash)

    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(
            f"Snapshot directory missing for '{repo_id}'.\n"
            f"Expected: {snapshot_dir}\n"
            f"Re-download it: huggingface-cli download {repo_id}"
        )

    print(f"Resolved '{repo_id}' → {snapshot_dir}")
    return snapshot_dir
