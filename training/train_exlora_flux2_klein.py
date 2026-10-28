"""
train_exlora_flux2_klein.py

ExLoRA (token-gated LoRA from the SIA paper) trainer for
the Selective Image Analogy project, built on top of FLUX.2-Klein.

Architecture:
  - FLUX.2-Klein model stack (Qwen3 text encoder, Flux2 VAE, Flux2 Transformer)
  - ExLoRA injected into context (text-stream) MM-DiT layers
  - Concat-mode analogy: A, A', B as conditions → denoise B'
  - Selective-suppression loss: full analogy prompt + token mask over the
    edit phrases that should be suppressed; target image is the partial
    ``combined<retained>.png`` (paper §4.3, eq. 14).
  - Multi-edit dataset (2/3/4 edits) read from ``<root>/<category>/<setting>
    /<item>/`` with ``combinedN.png`` partial-target images and a
    ``prompt.json`` listing every edit.

Usage:
  python training/train_exlora_flux2_klein.py \
      --config training/configs/train_exlora_flux2_klein.yaml
"""

# ── Offline mode by default: models are read from the local HF cache ────────
import argparse
import copy
import inspect as _inspect
import itertools
import json
import math
import os
import pickle
import random
import re
import sys
import textwrap
from pathlib import Path

# Must set offline env vars BEFORE any HuggingFace imports.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# WandB is disabled when WANDB_MODE=disabled or WANDB_DISABLED=true.
# Some GPU nodes have no wandb-core installed, so we skip
# all wandb calls entirely rather than just switching to offline mode.
_WANDB_MODE = os.environ.get("WANDB_MODE", "online").lower()
_USE_WANDB = (
    _WANDB_MODE not in ("disabled", "dryrun")
    and os.environ.get("WANDB_DISABLED", "false").lower() != "true"
)
if _USE_WANDB:
    import wandb

# Ensure the visual_analogy package (parent of this training/ dir) is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# ── Now safe to import HF libraries ───────────────────────────────────────
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from visual_analogy.models.selective_lora import ExLoRALinear, AnaLoRALinear
from visual_analogy.utils import (
    resolve_hf_snapshot_path,
    save_exlora_state_dict,
    save_analora_state_dict,
    load_analora_state_dict,
    exlora_token_mask_ctx,
)
from visual_analogy.utils.selective_lora import (
    inject_exlora_modules,
    inject_analora_modules,
    load_analora_from_peft_checkpoint,
    collect_analora_params,
    collect_exlora_params,
    set_analora_requires_grad,
    set_exlora_requires_grad,
    klein_find_substring_token_indices,
)

# ── Compatibility shim: strip unknown kwargs from set_module_tensor_to_device
try:
    from accelerate.utils import modeling as _accelerate_modeling
    _orig_smttd = _accelerate_modeling.set_module_tensor_to_device
    _known_smttd_params = set(_inspect.signature(_orig_smttd).parameters)
    _VAR_KEYWORD = _inspect.Parameter.VAR_KEYWORD
    _has_var_kw = any(
        p.kind == _VAR_KEYWORD
        for p in _inspect.signature(_orig_smttd).parameters.values()
    )
    if not _has_var_kw:
        def _smttd_compat(*args, **kwargs):
            filtered = {k: v for k, v in kwargs.items()
                        if k in _known_smttd_params}
            return _orig_smttd(*args, **filtered)
        _accelerate_modeling.set_module_tensor_to_device = _smttd_compat
        try:
            import diffusers.models.model_loading_utils as _dml
            _dml.set_module_tensor_to_device = _smttd_compat
        except Exception:
            pass
except Exception:
    pass
# ───────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# ExLoRA target modules: context (text-stream) layers only.
# ---------------------------------------------------------------------------
EXLORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_to_match(img, target_w, target_h):
    """Resize image to (target_w, target_h), both divisible by 16."""
    target_w = (target_w // 16) * 16
    target_h = (target_h // 16) * 16
    return img.resize((target_w, target_h), Image.LANCZOS)


def resize_keep_ratio(img, max_side=512):
    """Resize so the longer side equals max_side, rounding both sides to multiples of 16."""
    w, h = img.size
    scale = max_side / max(w, h)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def harmonize_images(a, a_prime, b, b_prime, max_side=512):
    """Normalise all four images for training.

    * A is resized to max_side (long-side) while preserving its aspect ratio,
      rounded to multiples of 16.
    * A' is snapped to A's exact pixel dimensions (same subject, same crop).
    * B is resized independently with the same max_side / ratio rule so its
      own aspect ratio is respected.
    * B' is snapped to B's exact pixel dimensions so that the denoising target
      always has the same spatial size as the B condition image.
    """
    a       = resize_keep_ratio(a, max_side)
    a_w, a_h = a.size
    a_prime = resize_to_match(a_prime, a_w, a_h)

    b       = resize_keep_ratio(b, max_side)
    b_w, b_h = b.size
    b_prime = resize_to_match(b_prime, b_w, b_h)

    return a, a_prime, b, b_prime


# ---------------------------------------------------------------------------
# Klein-specific helpers
# ---------------------------------------------------------------------------

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None,
                       timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        return scheduler.timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# AnaLoRA target modules (same as analora training)
# ---------------------------------------------------------------------------
ANALORA_TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]


# ---------------------------------------------------------------------------
# Helpers for the multi-edit (2/3/4) dataset layout
# ---------------------------------------------------------------------------
def _parse_combined_filename(filename: str):
    """``combined123.png`` → ``[1, 2, 3]``; returns ``None`` if no match."""
    m = re.match(r'combined(\d+)\.png$', filename)
    if not m:
        return None
    return sorted(int(d) for d in m.group(1))


def _get_combined_files(data_dir: Path):
    """Return list of ``(filename, edit_indices)`` for every ``combined*.png``
    in ``data_dir``."""
    result = []
    for f in data_dir.iterdir():
        indices = _parse_combined_filename(f.name)
        if indices is not None:
            result.append((f.name, indices))
    return result


def _retained_tier_probs(n_tiers: int) -> list:
    """Per-tier sampling probabilities for *retained* subsets in the ExLoRA
    selective-suppression step. Longest (most retained) tier first.

    1 → [1.0]
    2 → [0.65, 0.35]   — small preference for "more kept" but enough
                          mass on stronger suppression
    3 → [0.55, 0.30, 0.15]
    4+ → [0.50, 0.25, 0.25/(n-2), ...]
    """
    if n_tiers <= 1:
        return [1.0] * max(n_tiers, 1)
    if n_tiers == 2:
        return [0.70, 0.20]
    if n_tiers == 3:
        return [0.70, 0.20, 0.10]
    tail = 0.25 / (n_tiers - 2)
    return [0.60, 0.15] + [tail] * (n_tiers - 2)


def _sample_retained_subset(partial_files: list):
    """Pick one ``(filename, indices)`` from a list of available partial
    combined files, biased toward subsets that keep more edits."""
    if not partial_files:
        return None
    tier_dict: dict = {}
    for fname, indices in partial_files:
        tier_dict.setdefault(len(indices), []).append((fname, indices))
    sorted_tiers = sorted(tier_dict.keys(), reverse=True)
    probs = _retained_tier_probs(len(sorted_tiers))
    chosen_tier = random.choices(sorted_tiers, weights=probs)[0]
    return random.choice(tier_dict[chosen_tier])


# ---------------------------------------------------------------------------
# Dataset — multi-edit (2/3/4) layout with combinedN.png + prompt.json
# ---------------------------------------------------------------------------

class ImageAnalogyDataset(Dataset):
    """Selective image analogy dataset for ExLoRA training (Flux2-Klein).

    Directory layout (Selective Image Analogy multi-edit dataset)::

        <data_root>/
            <category>/                 # e.g. 2edits_non_creature | 3_edits |
                                        #      3edits_non_creature | 4edits
                <setting>/              # setting1 | setting2 | …
                    <item>/             # e.g. 1, 2, … (distinct subjects/items)
                        input.png           → original ``I^{∅}`` (A or B)
                        combined1.png       → image with edit 1 only
                        combined12.png      → image with edits 1+2
                        combined123.png     → image with edits 1+2+3
                        combined1234.png    → image with edits 1+2+3+4 (when N=4)
                        combinedXY.png      → any other available subset
                        prompt.json         → ``{"edit1": …, "edit2": …, …, "editN": …}``

    Each item is stored as a dict with:

    * ``input``    — path to ``input.png``
    * ``edits``    — ``{1: "text1", 2: "text2", …}`` (1-based int keys)
    * ``combined`` — ``{frozenset({1,2}): path, frozenset({1}): path, …}``
    * ``dir``      — item directory path string

    Sampling produces an analogy pair ``(item_a, item_b)`` from the same
    ``<category>/<setting>`` and selects a ``(demo_set, kept_set)`` pair
    where both combined images exist in both items and ``kept_set`` is a
    strict proper non-empty subset of ``demo_set``.  The suppression set is
    ``suppress_set = demo_set − kept_set``.

    * ``A``, ``B``          — input images
    * ``A_prime``           — ``item_a``'s ``combined[demo_set]`` (the demo A')
    * ``A_prime_total``     — same as ``A_prime``
    * ``A_prime_partial``   — ``item_a``'s ``combined[kept_set]`` if available,
                              else ``None`` (trainer falls back to ``A_prime``)
    * ``B_prime``           — ``item_b``'s ``combined[kept_set]`` (partial target
                              B'_R); equals ``B`` when ``kept_set`` is empty
                              (all-suppress fallback)
    * ``B_total``           — ``item_b``'s ``combined[demo_set]`` (full demo B')
    * ``pair_a_edits``      — list of edit strings for the *demo* edits, in
                              ``demo_set`` index order; length = ``|demo_set|``
    * ``suppress_indices``  — 0-based indices into ``pair_a_edits`` of edits
                              that are suppressed (``suppress_set`` positions)
    * ``suppress_edit_texts`` — the actual edit-text strings for every
                              suppressed edit (``[pair_a_edits[i] for i in
                              suppress_indices]``); always a list, never a
                              single string

    Demo sets are weighted by ``|demo_set|³`` so the model sees larger (more
    complete) demonstrations most of the time, matching §4.3 of the SIA paper.
    When no valid ``(demo, kept)`` pair exists across both items, the largest
    common demo is used with ``kept_set = ∅`` (all-suppress fallback).

    Notes
    -----
    * Items with fewer than 2 edits or no ``combined*.png`` on disk are
      skipped at construction time.
    * The ``_load_item`` static method handles per-item parsing and can be
      reused by subclasses or external tooling.
    """

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.categories: dict = {}
        n_total = 0
        n_skipped = 0

        for cat_dir in sorted(self.data_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat = cat_dir.name
            self.categories[cat] = {}

            for setting_dir in sorted(cat_dir.iterdir()):
                if not setting_dir.is_dir():
                    continue
                setting = setting_dir.name
                items = []

                for item_dir in sorted(setting_dir.iterdir()):
                    if not item_dir.is_dir():
                        continue
                    info = self._load_item(item_dir)
                    if info is None:
                        n_skipped += 1
                        continue
                    items.append(info)
                    n_total += 1

                if len(items) >= 2:
                    self.categories[cat][setting] = items

        self.cat_names = sorted(self.categories.keys())
        print(f"ExLoRA ImageAnalogyDataset: {len(self.cat_names)} categories, "
              f"{n_total} items kept ({n_skipped} skipped).")
        for cat in self.cat_names:
            n = sum(len(items) for items in self.categories[cat].values())
            print(f"  {cat}: {len(self.categories[cat])} settings, {n} items")

    # ------------------------------------------------------------------
    @staticmethod
    def _load_item(item_dir: Path):
        """Parse one item directory.  Returns None when the item is invalid."""
        input_path  = item_dir / 'input.png'
        prompt_file = item_dir / 'prompt.json'
        if not (input_path.exists() and prompt_file.exists()):
            return None
        try:
            prompt_dict = json.loads(prompt_file.read_text())
        except Exception:
            return None

        # Build 1-based edits dict from 'edit1', 'edit2', ... keys.
        edits: dict = {}
        for k, v in prompt_dict.items():
            m = re.fullmatch(r'edit(\d+)', k)
            if m and v and (v_stripped := v.strip()):
                edits[int(m.group(1))] = v_stripped

        if len(edits) < 2:
            return None

        # Build combined_map: frozenset(indices) → absolute path.
        # Only include combined files whose indices are all valid edit indices.
        valid_indices = set(edits.keys())
        combined_map: dict = {}
        for f in sorted(item_dir.iterdir()):
            indices_list = _parse_combined_filename(f.name)
            if indices_list is None:
                continue
            fs = frozenset(indices_list)
            if fs and fs.issubset(valid_indices):
                combined_map[fs] = str(f)

        if not combined_map:
            return None

        return {
            'input':    str(input_path),
            'edits':    edits,        # {1: "text1", 2: "text2", …}
            'combined': combined_map, # {frozenset(indices): path}
            'dir':      str(item_dir),
        }

    # ------------------------------------------------------------------
    def __len__(self):
        return sum(len(items) for c in self.categories.values()
                   for items in c.values())

    # ------------------------------------------------------------------
    def _sample_two_items(self):
        """Pick a category/setting and two distinct items from that setting.

        Categories are picked uniformly at random; settings are picked
        uniformly within a category; items are picked without replacement
        within a setting. Settings with < 2 items are excluded.
        """
        cat = random.choice(self.cat_names)
        settings_with_pairs = [
            s for s, items in self.categories[cat].items() if len(items) >= 2
        ]
        if not settings_with_pairs:
            # Fall back to any category that has at least one usable setting.
            valid_cats = [
                c for c in self.cat_names
                if any(len(items) >= 2 for items in self.categories[c].values())
            ]
            cat = random.choice(valid_cats)
            settings_with_pairs = [
                s for s, items in self.categories[cat].items() if len(items) >= 2
            ]
        setting = random.choice(settings_with_pairs)
        items = self.categories[cat][setting]
        item_a, item_b = random.sample(items, 2)
        return cat, setting, item_a, item_b

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_demo_kept(common_sets):
        """Sample a ``(demo_set, kept_set)`` pair from combined sets present
        in both items.

        All valid pairs ``(demo, kept)`` where ``kept`` is a non-empty proper
        subset of ``demo`` are enumerated.  Pairs are weighted by
        ``|demo|³`` to strongly prefer larger (more complete) demos.
        Falls back to ``(largest_demo, frozenset())`` when no strict-subset
        pair exists (all-suppress branch).

        Returns ``(demo_set, kept_set)`` — both frozensets.
        """
        valid_pairs = [
            (demo, kept)
            for demo in common_sets
            for kept in common_sets
            if len(kept) > 0 and kept < demo   # strict proper subset
        ]
        if valid_pairs:
            weights = [len(demo) ** 3 for demo, _ in valid_pairs]
            return random.choices(valid_pairs, weights=weights, k=1)[0]
        # No valid partial pair — use the largest common combined as the
        # demo, leave kept empty (all-suppress fallback).
        demo_set = max(common_sets, key=lambda s: (len(s), tuple(sorted(s))))
        return demo_set, frozenset()

    # ------------------------------------------------------------------
    def sample_val_pair(self):
        """Return a dict of paths + metadata for use in validation."""
        cat, setting, item_a, item_b = self._sample_two_items()

        a_edits    = item_a['edits']
        a_combined = item_a['combined']
        b_combined = item_b['combined']

        common_sets = set(a_combined) & set(b_combined)
        demo_set, kept_set = self._pick_demo_kept(common_sets)
        suppress_set = demo_set - kept_set

        demo_indices     = sorted(demo_set)
        kept_indices     = sorted(kept_set)
        suppress_indices = sorted(suppress_set)

        demo_edit_texts = [a_edits[i] for i in demo_indices if i in a_edits]

        # Positional 1-based indices within demo_edit_texts (used by the
        # validation code which does ``i - 1`` to get 0-based positions).
        demo_idx_to_pos1 = {idx: pos + 1 for pos, idx in enumerate(demo_indices)}
        retained_indices_1b = [demo_idx_to_pos1[i] for i in kept_indices
                                if i in demo_idx_to_pos1]
        suppress_indices_1b = [demo_idx_to_pos1[i] for i in suppress_indices
                                if i in demo_idx_to_pos1]

        a_demo_path  = a_combined[demo_set]
        b_demo_path  = b_combined[demo_set]
        b_kept_path  = b_combined.get(kept_set) if kept_set else None
        a_kept_path  = a_combined.get(kept_set) if kept_set else None

        return {
            'a_path':               item_a['input'],
            'a_prime_total_path':   a_demo_path,
            'a_prime_partial_path': a_kept_path,
            'b_path':               item_b['input'],
            'b_prime_total_path':   b_demo_path,
            # Fall back to demo when no kept image exists (all-suppress case).
            'b_prime_partial_path': b_kept_path if b_kept_path else b_demo_path,
            'all_indices_1b':       list(range(1, len(demo_indices) + 1)),
            'retained_indices_1b':  retained_indices_1b,
            'suppress_indices_1b':  suppress_indices_1b,
            'pair_a_edits':         demo_edit_texts,
            'pair_b_edits':         demo_edit_texts,
            'edit_name':            f"{cat}/{setting}",
        }

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        for _ in range(8):
            sample = self._try_sample()
            if sample is not None:
                return sample
        raise RuntimeError(
            "Failed to sample a valid analogy pair after 8 attempts.")

    # ------------------------------------------------------------------
    def _try_sample(self):
        cat, setting, item_a, item_b = self._sample_two_items()

        a_edits    = item_a['edits']
        a_combined = item_a['combined']
        b_combined = item_b['combined']

        common_sets = set(a_combined) & set(b_combined)
        if not common_sets:
            return None

        demo_set, kept_set = self._pick_demo_kept(common_sets)
        suppress_set = demo_set - kept_set

        # ---- Load images ----
        a_img      = Image.open(item_a['input']).convert('RGB')
        b_img      = Image.open(item_b['input']).convert('RGB')
        a_demo_img = Image.open(a_combined[demo_set]).convert('RGB')
        b_demo_img = Image.open(b_combined[demo_set]).convert('RGB')

        if kept_set:
            b_kept_img = Image.open(b_combined[kept_set]).convert('RGB')
            # A's matching partial for the AnaLoRA-partial step (may be absent).
            a_kept_img = (Image.open(a_combined[kept_set]).convert('RGB')
                          if kept_set in a_combined else None)
        else:
            b_kept_img = None
            a_kept_img = None

        # ---- Harmonize sizes ----
        a_img, a_demo_img, b_img, b_demo_img = harmonize_images(
            a_img, a_demo_img, b_img, b_demo_img, max_side=512)
        if b_kept_img is not None:
            b_kept_img = resize_to_match(b_kept_img, *b_img.size)
        if a_kept_img is not None:
            a_kept_img = resize_to_match(a_kept_img, *a_img.size)

        # ---- Build edit text lists ----
        demo_indices     = sorted(demo_set)
        suppress_indices = sorted(suppress_set)

        # demo_edit_texts: one entry per demo edit, in index order.
        # suppress_edit_texts: sub-list for edits being suppressed — every
        # edit in suppress_set gets its own entry (never collapsed to one).
        demo_edit_texts     = [a_edits[i] for i in demo_indices if i in a_edits]
        suppress_edit_texts = [a_edits[i] for i in suppress_indices if i in a_edits]

        if not demo_edit_texts:
            return None

        # 0-based positions within demo_edit_texts for the suppressed edits.
        demo_idx_to_pos = {idx: pos for pos, idx in enumerate(demo_indices)}
        suppress_indices_0b = [demo_idx_to_pos[i] for i in suppress_indices
                               if i in demo_idx_to_pos]

        # B_prime: partial target (kept edits only); B itself when all-suppress.
        b_prime = b_kept_img if b_kept_img is not None else b_img

        combined_prompt = build_analogy_prompt(demo_edit_texts, demo_edit_texts)

        return {
            'A':                   a_img,
            'A_prime':             a_demo_img,  # demo A' (same as A_prime_total)
            'A_prime_total':       a_demo_img,
            'A_prime_partial':     a_kept_img,  # matched partial A', or None
            'B':                   b_img,
            'B_prime':             b_prime,     # partial target B'_R
            'B_total':             b_demo_img,  # full demo B' (AnaLoRA total step)
            'edit_name':           f"{cat}/{setting}",
            'prompt':              combined_prompt,
            'pair_a_edits':        demo_edit_texts,
            'pair_b_edits':        demo_edit_texts,
            # 0-based positions in pair_a_edits for the edits being suppressed.
            'suppress_indices':    suppress_indices_0b,
            # Actual text strings for ALL suppressed edits (one per edit, not
            # collapsed) — used directly by build_token_mask without re-indexing.
            'suppress_edit_texts': suppress_edit_texts,
        }


def collate_fn(batch):
    return {
        'A':                   [item['A']                   for item in batch],
        'A_prime':             [item['A_prime']             for item in batch],
        'A_prime_total':       [item['A_prime_total']       for item in batch],
        'A_prime_partial':     [item['A_prime_partial']     for item in batch],
        'B':                   [item['B']                   for item in batch],
        'B_prime':             [item['B_prime']             for item in batch],
        'B_total':             [item['B_total']             for item in batch],
        'prompts':             [item['prompt']              for item in batch],
        'pair_a_edits':        [item['pair_a_edits']        for item in batch],
        'pair_b_edits':        [item['pair_b_edits']        for item in batch],
        # 0-based positions in pair_a_edits; each entry is a list (multi-edit).
        'suppress_indices':    [item['suppress_indices']    for item in batch],
        # Actual suppressed edit strings per sample — list[list[str]].
        'suppress_edit_texts': [item['suppress_edit_texts'] for item in batch],
    }


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_analogy_prompt(pair_a_edits, pair_b_edits, suppress_indices=None):
    """Build the concat-mode analogy prompt for an arbitrary number of edits.

    Supports N∈{2,3,4} (and beyond) by simply iterating over ``pair_a_edits``.
    Follows the same format as train_analora_flux2_klein.py: only the A
    pair edits are listed (the model infers B→B' by analogy).

    Args:
        pair_a_edits: list of edit strings for A→A' (length = number of edits).
        pair_b_edits: list of edit strings for B→B' (unused in prompt, kept
            for API compatibility).
        suppress_indices: iterable of 0-based indices into ``pair_a_edits`` to
            omit from the listing (``None`` = keep all).

    Returns:
        Prompt string. The listed edit numbers in the output are renumbered
        contiguously after suppression (Edit 1, Edit 2, …) so the visible
        prompt never references an omitted edit.
    """
    suppress = set(suppress_indices or [])
    kept = [e for i, e in enumerate(pair_a_edits) if i not in suppress and e]
    edits_part = " ".join(f"Edit {n + 1}: {e}." for n, e in enumerate(kept))
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )


# ---------------------------------------------------------------------------
# Token mask helpers (Qwen/Flux2 tokenizer)
# ---------------------------------------------------------------------------

def build_token_mask(full_prompt, edit_texts, tokenizer, device,
                     max_seq_len=512, use_chat_template: bool = True):
    """Build a (1, max_seq_len) bool mask covering tokens for each edit text.

    The mask is aligned against the **chat-templated** Qwen prompt — the
    exact sequence ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` feeds
    into the text encoder. Mask positions therefore line up with rows of
    ``prompt_embeds`` and with the ``encoder_hidden_states`` consumed by
    ``ExLoRALinear``.

    Args:
        full_prompt:  full prompt string (raw, pre-template)
        edit_texts:   list of edit strings whose tokens should be masked
        tokenizer:    Qwen2TokenizerFast
        device:       target device
        max_seq_len:  must match max_length used in prompt encoding
        use_chat_template: kept as an escape hatch for forensic runs only.
            ``True`` (default) is the only correct setting for Flux2-Klein.

    Returns:
        torch.BoolTensor of shape (1, max_seq_len). If a particular edit
        cannot be located the function raises — silent zero-masks were the
        original ExLoRA bug and must never recur.
    """
    mask = torch.zeros((1, max_seq_len), device=device, dtype=torch.bool)
    for text in edit_texts:
        if not text:
            continue
        try:
            indices = klein_find_substring_token_indices(
                full_prompt, text, tokenizer,
                max_length=max_seq_len,
                use_chat_template=use_chat_template,
            )
        except AssertionError as exc:
            # Substring not found in the prompt — almost always an edit-text
            # vs. prompt-builder mismatch. Loud failure is mandatory because
            # an empty mask silently disables ExLoRA.
            raise RuntimeError(
                f"[TokenMask] Could not locate edit text in prompt:\n"
                f"  edit:   {text!r}\n"
                f"  prompt: {full_prompt!r}\n"
                f"  underlying: {exc}"
            ) from exc
        for idx in indices:
            if idx < max_seq_len:
                mask[0, idx] = True

    if edit_texts and not bool(mask.any()):
        raise RuntimeError(
            "[TokenMask] All edit texts produced empty positions — ExLoRA "
            "would be a no-op. Check tokenizer / prompt format."
        )
    return mask


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    local_path = resolve_hf_snapshot_path(args.pretrained_model_name_or_path)
    # The frozen models are cast to bf16 in main() anyway; loading them in bf16
    # avoids a ~70 GB fp32 copy in host RAM (the LoRA layers stay fp32).
    bf16 = dict(local_files_only=True, torch_dtype=torch.bfloat16)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        local_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        local_path, subfolder="text_encoder", **bf16)
    vae = AutoencoderKLFlux2.from_pretrained(
        local_path, subfolder="vae", **bf16)
    transformer = Flux2Transformer2DModel.from_pretrained(
        local_path, subfolder="transformer", **bf16)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        local_path, subfolder="scheduler", local_files_only=True)
    return tokenizer, text_encoder, vae, transformer, noise_scheduler


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_text_embeddings(prompts, tokenizer, text_encoder):
    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompts,
    )
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds)
    text_ids = text_ids.to(prompt_embeds.device)
    return prompt_embeds, text_ids


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ---------------------------------------------------------------------------
# Latent helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_single_image(pipe, img, cell_size=512):
    """Resize, preprocess, and VAE-encode a single PIL image. Returns (1, C, h, w) latent."""
    img = resize_to_match(img, cell_size, cell_size)
    tensor = pipe.image_processor.preprocess(img, cell_size, cell_size)
    tensor = tensor.to(device=pipe._execution_device, dtype=torch.bfloat16)
    return pipe._encode_vae_image(tensor, generator=None)


@torch.no_grad()
def prepare_analogy_latents(pipe, batch, resolution=512):
    """Encode A, A', B, B' through VAE and prepare concat-mode latents + IDs.

    Returns:
        target_latents:    (B, N, C) packed B' latents
        condition_latents: (B, 3*N, C) packed [A, A', B] condition latents
        target_ids:        position IDs for target
        condition_ids:     position IDs for conditions
    """
    target_arr, cond_arr = [], []

    for a, a_prime, b, b_prime in zip(
        batch['A'], batch['A_prime'], batch['B'], batch['B_prime']
    ):
        lat_a       = encode_single_image(pipe, a,       resolution)
        lat_a_prime = encode_single_image(pipe, a_prime, resolution)
        lat_b       = encode_single_image(pipe, b,       resolution)
        lat_b_prime = encode_single_image(pipe, b_prime, resolution)

        target_arr.append(Flux2KleinPipeline._pack_latents(lat_b_prime))
        cond_a  = Flux2KleinPipeline._pack_latents(lat_a).squeeze(0)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime).squeeze(0)
        cond_b  = Flux2KleinPipeline._pack_latents(lat_b).squeeze(0)
        cond_arr.append(
            torch.cat([cond_a, cond_ap, cond_b], dim=0).unsqueeze(0))

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_b_prime)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b])
        
    return (
        torch.cat(target_arr, dim=0),
        torch.cat(cond_arr,   dim=0),
        target_ids,
        condition_ids,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_peft_ckpt_into_exlora(transformer, ckpt_dir, device):
    """Initialize ExLoRA A/B weights from a PEFT safetensors checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(ckpt_path):
        print(f"[PEFT Init] No file at {ckpt_path}, skipping checkpoint init.")
        return

    print(f"[PEFT Init] Loading PEFT weights from {ckpt_path}")
    peft_sd = load_file(ckpt_path)
    print(f"[PEFT Init] Found {len(peft_sd)} keys in PEFT checkpoint.")

    exlora_map = {
        name: mod
        for name, mod in transformer.named_modules()
        if isinstance(mod, ExLoRALinear)
    }
    print(f"[PEFT Init] ExLoRA has {len(exlora_map)} injected modules.")

    matched, skipped, mismatched = [], [], []

    for key, weight in peft_sd.items():
        norm = key
        for pfx in ("transformer.", "base_model.model."):
            if norm.startswith(pfx):
                norm = norm[len(pfx):]

        norm = re.sub(r'\.lora_([AB])\.[\w\-]+\.weight$', r'.lora_\1.weight', norm)

        if ".lora_A.weight" in norm:
            mod_name = norm.replace(".lora_A.weight", "")
            ab = "lora_A"
        elif ".lora_B.weight" in norm:
            mod_name = norm.replace(".lora_B.weight", "")
            ab = "lora_B"
        else:
            skipped.append(key)
            continue

        if mod_name not in exlora_map:
            skipped.append(key)
            continue

        target = getattr(exlora_map[mod_name], ab)
        if target.weight.shape != weight.shape:
            mismatched.append(f"{key}: peft={weight.shape} exlora={target.weight.shape}")
            continue

        with torch.no_grad():
            target.weight.copy_(weight.to(device))
        matched.append(key)

    print(
        f"[PEFT Init] matched={len(matched)}, "
        f"skipped={len(skipped)}, shape_mismatch={len(mismatched)}"
    )
    if mismatched:
        print("[PEFT Init] Shape mismatches:")
        for m in mismatched:
            print(f"  {m}")


def restore_training_state(optimizer, lr_scheduler, ckpt_dir, device):
    """Restore optimizer, lr_scheduler, and RNG state from an Accelerate checkpoint dir."""
    opt_path = os.path.join(ckpt_dir, "optimizer.bin")
    sch_path = os.path.join(ckpt_dir, "scheduler.bin")
    rng_path = os.path.join(ckpt_dir, "random_states_0.pkl")

    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"[Ckpt] Restored optimizer from {opt_path}")
    if os.path.exists(sch_path):
        lr_scheduler.load_state_dict(torch.load(sch_path, map_location=device))
        print(f"[Ckpt] Restored lr_scheduler from {sch_path}")
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng_state = pickle.load(f)
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        print(f"[Ckpt] Restored RNG state from {rng_path}")


# ---------------------------------------------------------------------------
# Accelerate save / load hooks
# ---------------------------------------------------------------------------

def get_save_model_hook(accelerator):
    def hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, Flux2Transformer2DModel):
                    save_exlora_state_dict(
                        unwrapped,
                        os.path.join(output_dir, "selective_lora.pt"),
                    )
                    save_analora_state_dict(
                        unwrapped,
                        os.path.join(output_dir, "base_lora.pt"),
                    )
                    weights.pop()
    return hook


def get_load_model_hook(accelerator):
    def hook(models, input_dir):
        assert len(models) == 1, "Only transformer should be in the models list."
        transformer = models.pop()

        # Load ExLoRA weights
        exlora_path = os.path.join(input_dir, "selective_lora.pt")
        state = torch.load(exlora_path, map_location=accelerator.device)
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys when loading ExLoRA: {unexpected}")
        print(f"[Resume] Loaded ExLoRA weights ({len(missing)} missing keys).")

        # Load AnaLoRA weights
        analora_path = os.path.join(input_dir, "base_lora.pt")
        if os.path.exists(analora_path):
            load_analora_state_dict(transformer, analora_path, accelerator.device)

        cast_training_params([transformer], dtype=torch.float32)
    return hook


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_analogy_validation(pipeline, train_dataset, tokenizer, text_encoder,
                           args, accelerator, global_step):
    """Run analogy inference with ExLoRA and log a comparison strip.

    Layout:
      - ONE shared header row at the very top:  A, A', B, B' GT(all_edits)
      - Then one row PER kept_set combination (2^n total, n = num edits
        in the full demo), each row showing:
            supp=[...],
            B' GT(keep=[...], supp=[...]),
            B' (AnaLoRA total prompt),
            B' (AnaLoRA partial prompt),
            B' (ExLoRA)

    The "all suppressed -> expect B" sanity-check column has been removed
    entirely (not computed, not displayed).

    Only this function was modified; ImageAnalogyDataset is used purely
    via its public attributes (no changes there).
    """
    print(f"Running ExLoRA analogy validation at step {global_step}...")

    pipe   = pipeline
    device = accelerator.device
    dtype  = torch.bfloat16

    # ------------------------------------------------------------------
    # 1) Sample one (item_a, item_b) pair using the dataset's own sampler.
    # ------------------------------------------------------------------
    cat, setting, item_a, item_b = train_dataset._sample_two_items()
    edit_name = f"{cat}/{setting}"

    a_edits    = item_a['edits']
    a_combined = item_a['combined']
    b_combined = item_b['combined']

    common_sets = set(a_combined) & set(b_combined)
    if not common_sets:
        print(f"  No common combined sets for pair in {edit_name}; skipping validation.")
        return

    # ------------------------------------------------------------------
    # 2) Fix demo_set = the largest common set (the full n-edit demo).
    # ------------------------------------------------------------------
    demo_set = max(common_sets, key=lambda s: (len(s), tuple(sorted(s))))
    demo_indices = sorted(demo_set)
    n_edits = len(demo_indices)

    pair_a_edits = [a_edits[i] for i in demo_indices if i in a_edits]
    if not pair_a_edits or len(pair_a_edits) != n_edits:
        print(f"  Missing edit text for demo_set {demo_set} in {edit_name}; skipping validation.")
        return
    pair_b_edits = pair_a_edits  # same convention as before (A and B share edit texts)

    # Map demo-set edit index (1-based, original numbering) -> 0-based
    # position within pair_a_edits.
    demo_idx_to_pos = {idx: pos for pos, idx in enumerate(demo_indices)}

    a_demo_path = a_combined[demo_set]
    b_demo_path = b_combined[demo_set]

    # ------------------------------------------------------------------
    # 3) Enumerate ALL 2^n subsets of demo_indices as kept_set candidates.
    #    Ordered full set first, empty set last.
    # ------------------------------------------------------------------
    all_kept_sets = []
    for r in range(n_edits, -1, -1):
        for combo in itertools.combinations(demo_indices, r):
            all_kept_sets.append(frozenset(combo))
    # all_kept_sets now has exactly 2^n entries.

    prompt = build_analogy_prompt(pair_a_edits, pair_b_edits)

    val_sample = {
        'A':       Image.open(item_a['input']).convert('RGB'),
        'A_prime': Image.open(a_demo_path).convert('RGB'),
        'B':       Image.open(item_b['input']).convert('RGB'),
        'B_total': Image.open(b_demo_path).convert('RGB'),  # B' GT (all edits)
        'edit_name': edit_name,
        'prompt':    prompt,
    }

    resolution = args.get("resolution", 512)

    # Encode images through VAE (shared across all kept_set combinations).
    lat_a       = encode_single_image(pipe, val_sample['A'],       resolution)
    lat_a_prime = encode_single_image(pipe, val_sample['A_prime'], resolution)
    lat_b       = encode_single_image(pipe, val_sample['B'],       resolution)

    # Text embeddings for the full prompt (shared across all combinations).
    prompt_embeds, text_ids = compute_text_embeddings(
        [prompt], tokenizer, text_encoder
    )
    guidance = None  # Klein: no guidance embeds

    patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

    def decode_latents(packed, h, w):
        unpacked = packed.permute(0, 2, 1).reshape(1, -1, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(
            1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(
            1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    def run_denoising(exlora_scale=1.0, suppress_indices=None,
                      prompt_embeds_override=None, text_ids_override=None,
                      mask_prompt_override=None):
        """Run denoising loop with ExLoRA at the given uniform scale.

        suppress_indices: list of 0-based positions (into pair_a_edits) to
            suppress via the token mask. ``None`` -> no suppression.
        When exlora_scale == 0 the mask is unused (ExLoRA is fully disabled).
        """
        emb     = prompt_embeds_override if prompt_embeds_override is not None else prompt_embeds
        tids    = text_ids_override      if text_ids_override      is not None else text_ids
        raw_pt  = mask_prompt_override   if mask_prompt_override   is not None else prompt

        if exlora_scale != 0.0:
            indices = suppress_indices if suppress_indices is not None else []
            texts_to_mask = [
                pair_a_edits[i]
                for i in indices
                if i < len(pair_a_edits) and pair_a_edits[i]
            ]
            max_seq_len = emb.shape[1]
            exlora_mask = build_token_mask(
                raw_pt, texts_to_mask, tokenizer, device,
                max_seq_len=max_seq_len,
            )  # (1, max_seq_len)
        else:
            exlora_mask = None

        for m in pipe.transformer.modules():
            if isinstance(m, ExLoRALinear):
                m.set_scaling(exlora_scale)

        cond_a  = Flux2KleinPipeline._pack_latents(lat_a)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
        cond_b  = Flux2KleinPipeline._pack_latents(lat_b)
        condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
        gen_latents = Flux2KleinPipeline._pack_latents(gen_latents)

        target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
        condition_ids = Flux2KleinPipeline._prepare_image_ids(
            [lat_a, lat_a_prime, lat_b]).to(device)
        combined_ids = torch.cat([target_ids, condition_ids], dim=1)

        image_seq_len = gen_latents.shape[1]
        num_steps = args.num_val_inference_steps
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        mu = compute_empirical_mu(image_seq_len, num_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)

        with exlora_token_mask_ctx(pipe.transformer, exlora_mask, disable_mask_after=False):
            for t in timesteps:
                latent_model_input = torch.cat(
                    [gen_latents, condition_latents], dim=1)
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t.expand(1).to(dtype) / 1000,
                    guidance=guidance,
                    encoder_hidden_states=emb,
                    txt_ids=tids,
                    img_ids=combined_ids,
                    return_dict=False,
                )[0][:, :gen_latents.shape[1]]
                gen_latents = pipe.scheduler.step(
                    noise_pred, t, gen_latents, return_dict=False)[0]

        for m in pipe.transformer.modules():
            if isinstance(m, ExLoRALinear):
                m.reset_scaling()

        return decode_latents(gen_latents, patch_h, patch_w)

    # ------------------------------------------------------------------
    # Shared header-row images (computed once, shown only at the very top).
    # ------------------------------------------------------------------
    target_w = 512
    target_h = 512

    def resize_img(img):
        return resize_to_match(img, target_w, target_h)

    a_img_full         = resize_img(val_sample['A'])
    a_prime_img_full   = resize_img(val_sample['A_prime'])
    b_img_full         = resize_img(val_sample['B'])
    b_gt_all_edits_img = resize_img(val_sample['B_total'])

    cell_w, cell_h = target_w, target_h

    def load_combined(combined_map, fset):
        """Return PIL image for combined_map[fset] if present, else None."""
        path = combined_map.get(fset)
        if path is None:
            return None
        return Image.open(path).convert('RGB')

    # ------------------------------------------------------------------
    # Pre-compute per-kept_set metadata (GT image, suppressed prompt, etc.)
    # ------------------------------------------------------------------
    row_groups = []

    for kept_set in all_kept_sets:
        suppress_set = demo_set - kept_set
        kept_indices_sorted    = sorted(kept_set)
        suppress_indices_sorted = sorted(suppress_set)

        suppress_pos_0b = [demo_idx_to_pos[i] for i in suppress_indices_sorted]
        kept_pos_1b     = [demo_idx_to_pos[i] + 1 for i in kept_indices_sorted]
        suppress_pos_1b  = [demo_idx_to_pos[i] + 1 for i in suppress_indices_sorted]

        b_prime_type = f"keep={kept_pos_1b}/supp={suppress_pos_1b}"

        # B' ground truth for this kept_set.
        if kept_set == demo_set:
            b_prime_gt_img = Image.open(b_demo_path).convert('RGB')
        elif not kept_set:
            b_kept_img = load_combined(b_combined, kept_set)
            b_prime_gt_img = b_kept_img if b_kept_img is not None else val_sample['B']
        else:
            b_kept_img = load_combined(b_combined, kept_set)
            b_prime_gt_img = b_kept_img if b_kept_img is not None else val_sample['B_total']

        suppress_edit_texts = [pair_a_edits[p] for p in suppress_pos_0b if pair_a_edits[p]]
        suppressed_prompt = build_analogy_prompt(
            pair_a_edits, pair_b_edits, suppress_indices=suppress_pos_0b,
        )

        row_groups.append({
            'kept_set':            kept_set,
            'suppress_set':        suppress_set,
            'kept_pos_1b':         kept_pos_1b,
            'suppress_pos_1b':     suppress_pos_1b,
            'b_prime_type':        b_prime_type,
            'b_prime_gt_img':      b_prime_gt_img,
            'suppress_pos_0b':     suppress_pos_0b,
            'suppress_edit_texts': suppress_edit_texts,
            'suppressed_prompt':   suppressed_prompt,
        })

    # ------------------------------------------------------------------
    # Run inference for every row-group's 3 prediction columns:
    #   1. AnaLoRA total prompt    — full prompt, ExLoRA off
    #   2. AnaLoRA partial prompt  — suppressed edit(s) stripped, ExLoRA off
    #   3. ExLoRA                    — full prompt + token mask for this kept_set
    # Column 1 doesn't depend on kept_set, so it's computed once and reused.
    # (The previous "all suppressed -> expect B" column has been removed.)
    # ------------------------------------------------------------------
    pred_base_total_shared = run_denoising(exlora_scale=0.0)

    for rg in row_groups:
        suppressed_prompt = rg['suppressed_prompt']
        suppress_pos_0b   = rg['suppress_pos_0b']

        if not suppress_pos_0b:
            # Nothing suppressed: base-partial == base-total, ExLoRA == no-mask.
            pred_base_partial = pred_base_total_shared
            pred_with_lora    = run_denoising(exlora_scale=1.0, suppress_indices=[])
        else:
            suppressed_prompt_embeds, suppressed_text_ids = compute_text_embeddings(
                [suppressed_prompt], tokenizer, text_encoder
            )
            pred_base_partial = run_denoising(
                exlora_scale=0.0,
                prompt_embeds_override=suppressed_prompt_embeds,
                text_ids_override=suppressed_text_ids,
                mask_prompt_override=suppressed_prompt,
            )
            pred_with_lora = run_denoising(
                exlora_scale=1.0, suppress_indices=suppress_pos_0b)

        rg['pred_base_total']   = pred_base_total_shared
        rg['pred_base_partial'] = pred_base_partial
        rg['pred_with_lora']    = pred_with_lora

    # ------------------------------------------------------------------
    # Render: ONE shared header row, then one row per kept_set combination.
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        font_size        = 20
        label_font_size  = 22
        _font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        ]
        font = None
        label_font = None
        for _fp in _font_candidates:
            if os.path.exists(_fp):
                font       = ImageFont.truetype(_fp, font_size)
                label_font = ImageFont.truetype(_fp, label_font_size)
                break

        n_cols = 4
        supp_col_w = 240
        strip_w = supp_col_w + cell_w * n_cols
        label_height = 28

        approx_char_w = (font_size * 0.55) if font else 6
        chars_per_line = max(60, int((strip_w - 8) / approx_char_w))

        def wrap(text):
            return "\n".join(textwrap.wrap(text, chars_per_line))

        row_block_h = label_height + cell_h

        # Top prompt text, shown once.
        top_prompt_lines = wrap(f"Full prompt ({n_edits} edits): {prompt}")
        line_h = font_size + 4
        top_text_h = (top_prompt_lines.count("\n") + 1) * line_h + 16

        # Total height: top text + 1 shared header row + 2^n per-combination rows.
        strip_h = top_text_h + row_block_h + len(row_groups) * row_block_h
        strip   = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
        draw    = ImageDraw.Draw(strip)

        draw.multiline_text(
            (8, 8), top_prompt_lines, fill=(0, 0, 160),
            font=font if font else None, spacing=4,
        )

        def _paste_row(row_imgs, row_labels, y0, supp_text=""):
            # Left suppression-text column.
            if supp_text:
                supp_lines = textwrap.wrap(str(supp_text), width=22)
                draw.multiline_text(
                    (10, y0 + 4),
                    "\n".join(supp_lines),
                    fill=(0, 0, 0),
                    font=label_font if label_font else None,
                    spacing=2,
                )

            # Optional separator to make the text column visually distinct.
            sep_x = supp_col_w - 1
            draw.line((sep_x, y0, sep_x, y0 + row_block_h), fill=(180, 180, 180), width=2)

            # Four image columns to the right.
            for i, (img, label) in enumerate(zip(row_imgs, row_labels)):
                cell_img = resize_img(img)
                cw, ch = cell_img.size
                x0 = supp_col_w + i * cell_w
                x_off = x0 + (cell_w - cw) // 2
                y_off = y0 + label_height + (cell_h - ch) // 2
                strip.paste(cell_img, (x_off, y_off))
                draw.text(
                    (x0 + 6, y0 + 4),
                    label,
                    fill=(0, 0, 0),
                    font=label_font if label_font else None,
                )

        y_cursor = top_text_h

        # ---- Shared header row (shown ONCE) ----
        header_row_imgs = [
            a_img_full, a_prime_img_full, b_img_full, b_gt_all_edits_img,
        ]
        header_row_labels = ["A", "A'", "B", "B' GT\n(all edits)"]
        _paste_row(header_row_imgs, header_row_labels, y0=y_cursor, supp_text="")
        y_cursor += row_block_h

        # ---- One row per kept_set combination ----
        for rg in row_groups:
            b_prime_type = rg['b_prime_type']

            row_imgs = [
                rg['b_prime_gt_img'],
                rg['pred_base_total'],
                rg['pred_base_partial'],
                rg['pred_with_lora'],
            ]
            row_labels = [
                f"B' GT\n({b_prime_type})",
                "B' (AnaLoRA\ntotal prompt)",
                "B' (AnaLoRA\npartial prompt)",
                "B' (ExLoRA)",
            ]
            supp_text = f"supp={rg['suppress_pos_1b']}"
            _paste_row(row_imgs, row_labels, y0=y_cursor, supp_text=supp_text)
            y_cursor += row_block_h

        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip ({len(row_groups)} combinations, "
              f"{n_edits} edits) to {strip_path}")

        # Save prompts/metadata to text file alongside the strip.
        prompt_path = vis_dir / f"step_{global_step:06d}_prompt.txt"
        lines = [f"edit: {edit_name}", "", f"full prompt:\n{prompt}", ""]
        for rg in row_groups:
            lines.append(
                f"keep={rg['kept_pos_1b']} supp={rg['suppress_pos_1b']} "
                f"-> partial prompt: {rg['suppressed_prompt']}"
            )
        prompt_path.write_text("\n".join(lines) + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(print_args=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = OmegaConf.load(parser.parse_args().config)
    if print_args:
        print("=" * 40, "Arguments", "=" * 40)
        for k in args:
            print(f"  {k}: {args[k]}")
        print("=" * 91)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args(print_args=True)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=dataloader_config,
    )

    if accelerator.is_main_process and _USE_WANDB:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(
            project=args.tracker_name,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*",   step_metric="train/step")

    set_seed(args.seed)

    # ---- Load models ----
    tokenizer, text_encoder, vae, transformer, noise_scheduler = load_models(args)

    for m in [text_encoder, vae, transformer]:
        m.requires_grad_(False)

    weight_dtype = torch.bfloat16
    for m in [text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=weight_dtype)

    # ---- Inject AnaLoRALinear on ALL AnaLoRA target modules ----
    base_replaced = inject_analora_modules(
        transformer, ANALORA_TARGET_MODULES,
        r=args.lora_rank, alpha=args.lora_rank, dropout=0.0,
    )
    print(f"[AnaLoRA] Injected {len(base_replaced)} AnaLoRALinear modules")

    # ---- Load pre-trained AnaLoRA weights from PEFT checkpoint ----
    load_analora_from_peft_checkpoint(
        transformer, args.analora_path, device="cpu")

    # ---- Inject ExLoRA on context targets (wraps AnaLoRALinear where applicable) ----
    replaced = inject_exlora_modules(
        transformer, EXLORA_TARGET_MODULES,
        r=args.lora_rank, alpha=args.lora_rank, dropout=args.lora_dropout,
    )
    print(f"[ExLoRA] Injected {len(replaced)} ExLoRALinear modules:")
    for name in replaced:
        print(f"  {name}")

    # ---- Optionally initialise ExLoRA from a PEFT checkpoint ----
    init_ckpt = getattr(args, "init_from_lora_ckpt", None)
    if init_ckpt:
        load_peft_ckpt_into_exlora(transformer, init_ckpt, accelerator.device)

    transformer.enable_gradient_checkpointing()

    # ---- Register Accelerate save/load hooks ----
    accelerator.register_save_state_pre_hook(get_save_model_hook(accelerator))
    accelerator.register_load_state_pre_hook(get_load_model_hook(accelerator))

    # ---- Keep LoRA params in float32 ----
    cast_training_params([transformer], dtype=torch.float32)

    # ---- Snapshot AnaLoRA weights as L2 anchor (prevents drift) ----
    analora_anchor = {}
    for mod_name, mod in transformer.named_modules():
        if isinstance(mod, AnaLoRALinear):
            analora_anchor[(mod_name, 'A')] = mod.lora_A.weight.detach().clone()
            analora_anchor[(mod_name, 'B')] = mod.lora_B.weight.detach().clone()
    analora_anchor_weight = getattr(args, "analora_anchor_weight", 1e-4)
    print(f"[L2 Anchor] Saved {len(analora_anchor)} AnaLoRA param tensors, "
          f"λ = {analora_anchor_weight}")

    # ---- Two optimizer param groups: ExLoRA (normal LR) + AnaLoRA (low LR) ----
    analora_lr = getattr(args, "analora_learning_rate", 5e-6)
    exlora_params = collect_exlora_params(transformer)
    analora_params = collect_analora_params(transformer)
    print(f"[Optimizer] ExLoRA params: {len(exlora_params)}, "
          f"AnaLoRA params: {len(analora_params)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": exlora_params, "lr": args.learning_rate},
            {"params": analora_params, "lr": analora_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # ---- Dataset & dataloader ----
    train_dataset = ImageAnalogyDataset(args.data_root)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=True,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # ---- Optionally restore optimizer/scheduler/RNG from init checkpoint ----
    if init_ckpt and getattr(args, "init_optimizer_from_ckpt", False):
        restore_training_state(optimizer, lr_scheduler, init_ckpt, accelerator.device)

    len_train_dataset          = len(train_dataset)
    len_train_dataloader       = math.ceil(len_train_dataset / args.train_batch_size)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    num_train_epochs           = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    print("***** Running ExLoRA training (Flux2-Klein) *****")
    print(f"  Num examples             = {len_train_dataset}")
    print(f"  Num epochs               = {num_train_epochs}")
    print(f"  Batch size per device    = {args.train_batch_size}")
    print(f"  Gradient accumulation    = {args.gradient_accumulation_steps}")
    print(f"  Total optimisation steps = {args.max_train_steps}")

    initial_global_step = 0
    global_step  = 0
    first_epoch  = 0

    # ---- Resume from ExLoRA checkpoint ----
    if args.resume_from_checkpoint:
        # A fresh run has no output_dir yet; treat that as "nothing to resume".
        existing = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
        dirs = [d for d in existing if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if dirs else None
        if path is None:
            print(f"No checkpoint found in {args.output_dir}. Starting fresh.")
            args.resume_from_checkpoint = False
        else:
            print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step         = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch         = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ---- Build pipeline for VAE encoding & validation ----
    pipeline = Flux2KleinPipeline(
        # Own copy: validation calls retrieve_timesteps() on pipe.scheduler, which would
        # otherwise shrink noise_scheduler.timesteps to the 28 inference steps and
        # collapse training timestep sampling onto the lowest-noise one.
        scheduler=copy.deepcopy(noise_scheduler),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer, keep_fp32_wrapper=False),
    )
    unwrapped_transformer = accelerator.unwrap_model(transformer)

    transformer.train()

    analora_step_ratio = getattr(args, "analora_step_ratio", 0.333)
    print(f"[Training] analora_step_ratio = {analora_step_ratio}")

    # ====================================================================
    # Training loop
    # ====================================================================
    for epoch in range(first_epoch, num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                device = accelerator.device
                pair_a_edits = batch['pair_a_edits']  # List[List[str]]
                pair_b_edits = batch['pair_b_edits']

                # ---- Decide step type ----
                # Distribution (single draw):
                #   [0.00, 0.25) → AnaLoRA partial  (25 %)
                #   [0.25, 0.55) → AnaLoRA total    (30 %)
                #   [0.55, 0.90) → ExLoRA partial     (35 %)
                #   [0.90, 1.00) → ExLoRA all-supp    (10 %)
                _r = random.random()
                is_all_suppressed_step = _r >= 0.90
                is_analora_step      = _r < 0.55           # covers both AnaLoRA cases
                use_total_change       = 0.25 <= _r < 0.55   # AnaLoRA total only
                is_analora_partial   = is_analora_step and not use_total_change

                # ---- Toggle trainability ----
                if is_analora_step:
                    set_exlora_requires_grad(unwrapped_transformer, False)
                    set_analora_requires_grad(unwrapped_transformer, True)
                else:
                    # Both ExLoRA steps (partial + all-suppressed) train ExLoRA.
                    set_exlora_requires_grad(unwrapped_transformer, True)
                    set_analora_requires_grad(unwrapped_transformer, False)

                # ---- Choose target image ----
                # AnaLoRA-partial also swaps A_prime to the matched partial
                # (same "kept edits" as B_prime).  Falls back to A_prime_total
                # where the matched partial is unavailable for a given sample.
                # ExLoRA branches keep A_prime = A_prime_total (the full-edit
                # demonstration), since the paper §4.3 selective-suppression
                # step trains the model to suppress edits that REMAIN visible
                # in A'.
                if use_total_change:
                    target_batch = {**batch, 'B_prime': batch['B_total']}
                elif is_all_suppressed_step:
                    # All edits suppressed → model should reproduce B unchanged.
                    target_batch = {**batch, 'B_prime': batch['B']}
                elif is_analora_partial:
                    a_prime_matched = [
                        ap if ap is not None else at
                        for ap, at in zip(
                            batch['A_prime_partial'], batch['A_prime_total']
                        )
                    ]
                    target_batch = {**batch, 'A_prime': a_prime_matched}
                else:
                    target_batch = batch

                # ---- Encode images (concat mode) ----
                target_latents, condition_latents, target_ids, condition_ids = \
                    prepare_analogy_latents(pipeline, target_batch,
                                           resolution=args.get("resolution", 512))

                target_ids    = target_ids.to(device)
                condition_ids = condition_ids.to(device)
                bsz = target_latents.shape[0]

                target_ids    = target_ids.expand(bsz, -1, -1)
                condition_ids = condition_ids.expand(bsz, -1, -1)
                combined_ids  = torch.cat([target_ids, condition_ids], dim=1)

                # ---- Sample timestep and add noise ----
                u = compute_density_for_timestep_sampling("none", batch_size=bsz)
                indices = (
                    u * (args.max_train_timesteps - args.min_train_timesteps)
                    + args.min_train_timesteps
                ).long().clamp(0, len(noise_scheduler.timesteps) - 1)
                timesteps = noise_scheduler.timesteps[indices].to(device)
                sigmas = get_sigmas(
                    noise_scheduler, timesteps, device,
                    n_dim=target_latents.ndim, dtype=target_latents.dtype,
                )
                noise = torch.randn_like(target_latents)
                noisy_model_input = (1.0 - sigmas) * target_latents + sigmas * noise

                # Klein: no guidance embeds
                guidance = None

                orig_shape = noisy_model_input.shape
                latent_model_input = torch.cat(
                    [noisy_model_input, condition_latents], dim=1)

                if is_analora_step:
                    # ==== AnaLoRA step: standard flow matching loss ====
                    # AnaLoRA has no selective suppression mechanism, so the
                    # suppressed edit is removed from the prompt text entirely
                    # (instead of masking tokens inside a full prompt).  ExLoRA
                    # is fully disabled (mask=None) for the whole forward.
                    if use_total_change:
                        # GT=B_prime_total → keep every edit in the prompt.
                        prompts_to_encode = [
                            build_analogy_prompt(ae, be)
                            for ae, be in zip(pair_a_edits, pair_b_edits)
                        ]
                    else:
                        # GT=partial combined → strip the suppressed edits
                        # from the prompt so only the retained edits remain.
                        # ``suppress_indices`` is a *list* of 0-based indices
                        # (multi-edit suppression for N=3,4 datasets).
                        prompts_to_encode = [
                            build_analogy_prompt(
                                ae, be,
                                suppress_indices=idxs,
                            )
                            for ae, be, idxs in zip(
                                pair_a_edits, pair_b_edits,
                                batch['suppress_indices'],
                            )
                        ]
                    prompt_embeds, text_ids = compute_text_embeddings(
                        prompts_to_encode, tokenizer, text_encoder,
                    )

                    with exlora_token_mask_ctx(transformer, None):
                        model_pred = transformer(
                            hidden_states=latent_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=combined_ids,
                            return_dict=False,
                        )[0][:, :orig_shape[1]]

                    flow_target = noise - target_latents
                    weighting = compute_loss_weighting_for_sd3("none", sigmas=sigmas)
                    loss = torch.mean(
                        (weighting.float()
                         * (model_pred.float() - flow_target.float()) ** 2
                         ).reshape(bsz, -1),
                        dim=1,
                    ).mean()

                else:
                    # ==== ExLoRA step: flow matching loss with token mask ====
                    # GT = combined<retained>.png (partial edit) OR B (all suppressed)
                    if is_all_suppressed_step:
                        # Mask every edit token — model must output B unchanged.
                        suppress_per_sample = [
                            list(range(len(ae))) for ae in pair_a_edits
                        ]
                    else:
                        # ``batch['suppress_indices']`` is a list of lists
                        # (0-based edit indices to suppress per sample).
                        suppress_per_sample = batch['suppress_indices']

                    full_prompts = [
                        build_analogy_prompt(ae, be)
                        for ae, be in zip(pair_a_edits, pair_b_edits)
                    ]

                    prompt_embeds, text_ids = compute_text_embeddings(
                        full_prompts, tokenizer, text_encoder,
                    )

                    # ---- Build (bsz, max_seq_len) token mask for suppressed edits ----
                    max_seq_len = prompt_embeds.shape[1]
                    tokens_mask = torch.zeros((bsz, max_seq_len), device=device, dtype=torch.bool)
                    for i in range(bsz):
                        texts_to_mask = []
                        for j in suppress_per_sample[i]:
                            if j < len(pair_a_edits[i]) and pair_a_edits[i][j]:
                                texts_to_mask.append(pair_a_edits[i][j])
                        single_mask = build_token_mask(
                            full_prompts[i], texts_to_mask, tokenizer, device,
                            max_seq_len=max_seq_len,
                        )
                        tokens_mask[i] = single_mask[0]

                    # ---- Forward with ExLoRA active + token mask ----
                    with exlora_token_mask_ctx(transformer, tokens_mask, disable_mask_after=False):
                        model_pred = transformer(
                            hidden_states=latent_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=combined_ids,
                            return_dict=False,
                        )[0][:, :orig_shape[1]]

                    flow_target = noise - target_latents
                    weighting = compute_loss_weighting_for_sd3("none", sigmas=sigmas)
                    loss = torch.mean(
                        (weighting.float()
                         * (model_pred.float() - flow_target.float()) ** 2
                         ).reshape(bsz, -1),
                        dim=1,
                    ).mean()

                # ---- L2 anchor: prevent AnaLoRA drift on analora steps ----
                if is_analora_step and analora_anchor_weight > 0:
                    anchor_loss = torch.tensor(0.0, device=device)
                    for mod_name, mod in unwrapped_transformer.named_modules():
                        if isinstance(mod, AnaLoRALinear):
                            for ab, param in [('A', mod.lora_A.weight),
                                              ('B', mod.lora_B.weight)]:
                                anchor = analora_anchor[(mod_name, ab)]
                                anchor_loss = anchor_loss + torch.sum(
                                    (param.float() - anchor.to(device).float()) ** 2)
                    print(f"[L2 Anchor] AnaLoRA anchor loss: {anchor_loss.item():.4f}")
                    print("loss before anchor:", loss.item())
                    loss = loss + analora_anchor_weight * anchor_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ---- After gradient sync ----
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and _USE_WANDB:
                    wandb.log({
                        "train/loss":  loss.detach().item(),
                        "train/lr":    lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step":  global_step,
                    })

                if accelerator.is_main_process:
                    if (global_step % args.checkpointing_steps == 0
                            or global_step == args.max_train_steps):
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if (global_step % args.validation_steps == 0
                            or global_step == args.max_train_steps):
                        pipeline.transformer = accelerator.unwrap_model(
                            transformer, keep_fp32_wrapper=False)
                        log_analogy_validation(
                            pipeline, train_dataset,
                            tokenizer, text_encoder,
                            args, accelerator, global_step,
                        )
                        transformer.train()

            if is_all_suppressed_step:
                step_type = "exlora_all_supp"
            elif use_total_change:
                step_type = "base_total"
            elif is_analora_step:
                step_type = "base_partial"
            else:
                step_type = "exlora"
            logs = {
                "loss": loss.detach().item(),
                "lr":   lr_scheduler.get_last_lr()[0],
                "step_type": step_type,
            }
            if is_analora_step and analora_anchor_weight > 0:
                logs["anchor_loss"] = (analora_anchor_weight * anchor_loss).detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process and _USE_WANDB:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
