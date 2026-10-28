#!/usr/bin/env python3
"""
_check_token_alignment.py

Regression test for the token-position bug:
  - ExLoRA's mask must align against the *chat-templated* Qwen prompt
    (the same sequence Flux2KleinPipeline._get_qwen3_prompt_embeds encodes).
  - The CSV (steering vector) must use the same position space, so that
    masking edit ``e_i`` with ExLoRA and steering edit ``e_i`` with the CSV
    operate on EXACTLY the same token indices.

This script needs only the Qwen tokenizer (CPU, no GPU, no model load).
It exits non-zero on any mismatch so it can be wired into CI / pre-commit.

Coverage:
  - 2-edit, 3-edit, and 4-edit prompts (matching the SIA multi-edit dataset).
  - For every edit:
      * positions decode back to the edit text,
      * positions are a subset of the templated-prompt positions,
      * ExLoRA's ``build_token_mask`` and the steering-vector
        ``templated_substring_indices`` agree on the position set.
  - Templated vs. raw alignment differ (canary against the pre-fix
    silent-empty-mask bug).
  - ``build_token_mask`` raises (not silently zeroes) on missing edits.

Usage:
  python training/_check_token_alignment.py
  python training/_check_token_alignment.py --pretrained black-forest-labs/FLUX.2-klein-base-9B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from transformers import Qwen2TokenizerFast  # noqa: E402

from visual_analogy.utils.hf_utils import resolve_hf_snapshot_path  # noqa: E402
from visual_analogy.utils.selective_lora import (  # noqa: E402
    _apply_klein_chat_template,
    klein_find_substring_token_indices,
    klein_templated_prompt_token_positions,
)
from training.train_exlora_flux2_klein import (  # noqa: E402
    build_analogy_prompt,
    build_token_mask,
)
from training.infer_single import templated_substring_indices  # noqa: E402


MAX_LEN = 512

# One representative prompt per supported edit count (N=2, 3, 4).
# The strings are chosen to match the kind of text actually emitted by
# prompt.json in the SIA dataset (short imperative edit phrases).
EDITS_BY_N: dict[int, list[str]] = {
    2: [
        "turned-on front lights",
        "open the front doors",
    ],
    3: [
        "add open-top crown",
        "add gold seamless necklace",
        "add angry expression",
    ],
    4: [
        "add open-top crown without jewelries",
        "add green color hair",
        "add old cobblestone alley background",
        "add Van gogh style",
    ],
}


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _check(cond: bool, msg: str) -> bool:
    print((_green("[ok]   ") if cond else _red("[FAIL] ")) + msg)
    return bool(cond)


def _run_checks_for_n(n: int, edits: list[str], tokenizer) -> bool:
    print()
    print(f"=== N={n} edits ===")
    prompt = build_analogy_prompt(edits, edits)
    print(f"prompt: {prompt!r}")
    templated = _apply_klein_chat_template(prompt, tokenizer)
    print(f"templated len(chars)={len(templated)}; raw len(chars)={len(prompt)}")

    enc = tokenizer(
        templated,
        max_length=MAX_LEN, padding="max_length", truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"][0].tolist()

    all_ok = True

    # 1. Each edit aligns and decodes back to itself
    for i, edit in enumerate(edits):
        positions = klein_find_substring_token_indices(
            prompt, edit, tokenizer, max_length=MAX_LEN, use_chat_template=True)
        decoded = tokenizer.decode([ids[p] for p in positions],
                                   skip_special_tokens=False)
        ok = decoded.strip() == edit.strip()
        all_ok &= _check(
            ok,
            f"  edit {i+1} aligns: positions={positions[0]}..{positions[-1]} "
            f"decoded={decoded!r}"
        )

    # 2. Per-edit positions are a strict subset of the all-prompt positions
    all_positions = set(klein_templated_prompt_token_positions(
        prompt, tokenizer, max_length=MAX_LEN))
    for i, edit in enumerate(edits):
        edit_positions = set(klein_find_substring_token_indices(
            prompt, edit, tokenizer, max_length=MAX_LEN, use_chat_template=True))
        all_ok &= _check(
            edit_positions.issubset(all_positions),
            f"  edit {i+1} positions are a subset of templated-prompt positions"
        )

    # 3. ExLoRA mask ↔ CSV/steering-vector positions agree per edit.
    # This is the explicit invariant the user asked us to verify: the
    # token index used by ExLoRA to gate the LoRA residual MUST equal the
    # token index used by the CSV to steer the prompt embedding.
    for i, edit in enumerate(edits):
        exlora_mask = build_token_mask(
            prompt, [edit], tokenizer, device=torch.device("cpu"),
            max_seq_len=MAX_LEN, use_chat_template=True,
        )
        exlora_positions = set(torch.nonzero(exlora_mask[0]).flatten().tolist())
        csv_positions = set(templated_substring_indices(
            prompt, edit, tokenizer, max_length=MAX_LEN))
        all_ok &= _check(
            exlora_positions == csv_positions and len(exlora_positions) > 0,
            f"  edit {i+1} ExLoRA mask positions == CSV positions "
            f"({sorted(exlora_positions)[:4]}..)"
        )

    # 4. build_token_mask with the full edit list covers >= sum-of-per-edit
    full_mask = build_token_mask(
        prompt, edits, tokenizer, device=torch.device("cpu"),
        max_seq_len=MAX_LEN, use_chat_template=True,
    )
    n_true = int(full_mask.sum().item())
    expected_min = sum(len(klein_find_substring_token_indices(
        prompt, e, tokenizer, max_length=MAX_LEN, use_chat_template=True))
        for e in edits)
    all_ok &= _check(
        n_true == expected_min,
        f"  build_token_mask True-count == sum of per-edit positions "
        f"({n_true} vs {expected_min})"
    )

    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str,
                        default="black-forest-labs/FLUX.2-klein-base-9B")
    args = parser.parse_args()

    local_path = resolve_hf_snapshot_path(args.pretrained)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        local_path, subfolder="tokenizer", local_files_only=True)

    all_ok = True
    for n in sorted(EDITS_BY_N):
        all_ok &= _run_checks_for_n(n, EDITS_BY_N[n], tokenizer)

    print()
    print("=== Regression canaries (run once, N=2) ===")
    edits = EDITS_BY_N[2]
    prompt = build_analogy_prompt(edits, edits)

    # Templated alignment differs from raw alignment
    raw_positions = klein_find_substring_token_indices(
        prompt, edits[0], tokenizer, max_length=MAX_LEN, use_chat_template=False)
    tmpl_positions = klein_find_substring_token_indices(
        prompt, edits[0], tokenizer, max_length=MAX_LEN, use_chat_template=True)
    all_ok &= _check(
        raw_positions != tmpl_positions,
        f"raw vs templated positions differ (chat template adds a prefix) — "
        f"raw={raw_positions[:3]}..  tmpl={tmpl_positions[:3]}.."
    )

    # build_token_mask raises (not silently zeroes) when an edit is missing
    bogus_edit = "this exact phrase definitely is not in the prompt 12345"
    raised = False
    try:
        build_token_mask(
            prompt, [bogus_edit], tokenizer, device=torch.device("cpu"),
            max_seq_len=MAX_LEN, use_chat_template=True,
        )
    except RuntimeError:
        raised = True
    all_ok &= _check(
        raised,
        "build_token_mask RAISES on missing edit (no silent empty mask)"
    )

    if all_ok:
        print(_green("\nAll token-alignment checks passed."))
        sys.exit(0)
    print(_red("\nOne or more token-alignment checks FAILED."))
    sys.exit(1)


if __name__ == "__main__":
    main()
