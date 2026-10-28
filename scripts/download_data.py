"""Download the SIA training data from the Hugging Face Hub into the folder
layout the training scripts read (``data_root`` in the configs)::

    <out>/<category>/<setting>/<item>/
        input.png, combined<kept edits>.png, prompt.json, refrence_edit_object<i>.png

Images are written byte-for-byte as stored on the Hub.

Usage:
    python scripts/download_data.py --out data
"""
import argparse
import copy
import json
from pathlib import Path

from datasets import Image, load_dataset

REPO = "Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy"


def _no_decode(feature):
    """Copy of ``feature`` with every Image (at any nesting depth) set to decode=False."""
    if isinstance(feature, Image):
        return Image(decode=False)
    if isinstance(feature, dict):
        return type(feature)({k: _no_decode(v) for k, v in feature.items()})
    if isinstance(feature, list):
        return [_no_decode(f) for f in feature]
    if hasattr(feature, "feature"):  # List / Sequence / LargeList
        clone = copy.copy(feature)
        clone.feature = _no_decode(feature.feature)
        return clone
    return feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()

    ds = load_dataset(args.repo, "subjects", split="train")
    # Keep the stored PNG bytes instead of decoding and re-encoding them.
    ds = ds.cast(_no_decode(ds.features))

    out = Path(args.out)
    for row in ds:
        item = out / row["subject_id"]
        item.mkdir(parents=True, exist_ok=True)
        (item / "input.png").write_bytes(row["source"]["bytes"])
        full = "".join(str(i) for i in range(1, row["num_edits"] + 1))
        (item / f"combined{full}.png").write_bytes(row["full_edit"]["bytes"])
        for partial in row["partial_edits"]:
            kept = "".join(str(i) for i in partial["kept_edits"])
            (item / f"combined{kept}.png").write_bytes(partial["image"]["bytes"])
        for i, ref in enumerate(row["edit_references"], 1):
            (item / f"refrence_edit_object{i}.png").write_bytes(ref["bytes"])
        prompt = {f"edit{i}": e for i, e in enumerate(row["edits"], 1)}
        (item / "prompt.json").write_text(json.dumps(prompt, indent=4))
    print(f"Wrote {len(ds)} subjects to {out}/")


if __name__ == "__main__":
    main()
