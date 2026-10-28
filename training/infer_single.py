#!/usr/bin/env python3
"""
infer_single.py

Single-image analogy inference with all suppression modes.

Given A, A', B and a list of edits [e1, e2, ..., eN], runs:
  - analora          : AnaLoRA only, ExLoRA off
  - exlora (no supp)   : ExLoRA on, no mask
  - exlora (supp e1)   : ExLoRA on, mask e1 tokens
  - exlora (supp e2)   : ExLoRA on, mask e2 tokens
  ...
  - exlora (supp all)  : ExLoRA on, mask all edit tokens

Saves a strip:  A | A' | B | analora | exlora_none | exlora_e1 | ... | exlora_all

Usage:
  python training/infer_single.py \\
      --config   training/configs/train_exlora_flux2_klein.yaml \\
      --image_a       /path/to/A.png \\
      --image_a_prime /path/to/A_prime.png \\
      --image_b       /path/to/B.png \\
      --edits "X" "Y" "Z" \\
      --output /tmp/result.png
"""

import argparse
import itertools
import os
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from contextlib import contextmanager
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from visual_analogy.models.selective_lora import ExLoRALinear, AnaLoRALinear
from visual_analogy.utils import (
    resolve_hf_snapshot_path,
    load_analora_state_dict,
    exlora_token_mask_ctx,
)
from visual_analogy.utils.selective_lora import (
    inject_analora_modules,
    inject_exlora_modules,
    klein_find_substring_token_indices,
    klein_templated_prompt_token_positions,
    load_analora_from_peft_checkpoint,
)
from training.train_exlora_flux2_klein import (
    ANALORA_TARGET_MODULES,
    EXLORA_TARGET_MODULES,
    build_analogy_prompt,
    build_token_mask,
    compute_empirical_mu,
    compute_text_embeddings,
    encode_single_image,
    retrieve_timesteps,
)
from training.steering_dom import get_or_build_steering_vector


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]

def _load_font(size):
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return None


def _fit_thumbnail(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale ``img`` to fit inside (max_w, max_h) while preserving aspect ratio.
    The result is centred on a white background of exactly (max_w, max_h)."""
    src_w, src_h = img.size
    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
    off_x = (max_w - new_w) // 2
    off_y = (max_h - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas


def _cell_dims(ref_img: Image.Image, cell_size: int) -> tuple:
    """Derive (cell_w, cell_h) for grid cells from a reference image's aspect
    ratio, bounded by ``cell_size``. The larger side is always exactly
    ``cell_size``; the smaller side scales down proportionally."""
    src_w, src_h = ref_img.size
    if src_w >= src_h:
        cell_w = cell_size
        cell_h = max(1, int(cell_size * src_h / src_w))
    else:
        cell_h = cell_size
        cell_w = max(1, int(cell_size * src_w / src_h))
    return cell_w, cell_h


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if not dirs:
        return None
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, dirs[-1])


def load_pipeline(args, device):
    local_path = resolve_hf_snapshot_path(args.pretrained_model_name_or_path)
    tokenizer    = Qwen2TokenizerFast.from_pretrained(local_path, subfolder="tokenizer",    local_files_only=True)
    # Load straight in bf16 (the pipeline is cast to bf16 below anyway); an fp32
    # load needs ~70 GB of host RAM for the 9B transformer + 8B text encoder.
    bf16 = dict(local_files_only=True, torch_dtype=torch.bfloat16)
    text_encoder = Qwen3ForCausalLM.from_pretrained( local_path, subfolder="text_encoder", **bf16)
    vae          = AutoencoderKLFlux2.from_pretrained(local_path, subfolder="vae",          **bf16)
    transformer  = Flux2Transformer2DModel.from_pretrained(local_path, subfolder="transformer", **bf16)
    scheduler    = FlowMatchEulerDiscreteScheduler.from_pretrained(local_path, subfolder="scheduler", local_files_only=True)

    explicit_ckpt = getattr(args, "checkpoint_dir", None)
    if explicit_ckpt:
        # Fail loud: a user-supplied path that doesn't exist used to be
        # silently swallowed (ckpt_dir = None), which left ExLoRA at its
        # zero-initialised state — ExLoRA had *no effect* and st=0 vs st=1
        # produced visually identical outputs. Now we raise so typos /
        # wrong experiment names are caught immediately.
        if not os.path.isdir(explicit_ckpt):
            raise FileNotFoundError(
                f"--checkpoint_dir not found: {explicit_ckpt}\n"
                f"  Check that the experiment name and step are correct, e.g.\n"
                f"  experiments/<exp>/checkpoint-<N>")
        ckpt_dir = explicit_ckpt
    else:
        ckpt_dir = _find_latest_checkpoint(args.output_dir)

    # A Stage-2 checkpoint carries its own AnaLoRA, which replaces the Stage-1
    # weights; only fall back to analora_path when it doesn't. Take the rank
    # from the weights so a config/checkpoint rank mismatch can't break loading.
    bl_path = os.path.join(ckpt_dir, "base_lora.pt") if ckpt_dir else None
    has_ckpt_base = bl_path is not None and os.path.exists(bl_path)
    if has_ckpt_base:
        bl_state = torch.load(bl_path, map_location="cpu", weights_only=True)
        rank = next(w.shape[0] for k, w in bl_state.items() if k.endswith("lora_A.weight"))
        if rank != args.lora_rank:
            print(f"[Checkpoint] lora_rank {args.lora_rank} in config -> {rank} from checkpoint")
            args.lora_rank = rank

    n_base = inject_analora_modules(transformer, ANALORA_TARGET_MODULES,
                                      r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[AnaLoRA] Injected {len(n_base)} modules")
    if not has_ckpt_base:
        load_analora_from_peft_checkpoint(transformer, args.analora_path, device="cpu")

    n_st = inject_exlora_modules(transformer, EXLORA_TARGET_MODULES,
                                         r=args.lora_rank, alpha=args.lora_rank, dropout=0.0)
    print(f"[ExLoRA] Injected {len(n_st)} modules")

    if ckpt_dir:
        print(f"[Checkpoint] Loading from {ckpt_dir}")
        st_path = os.path.join(ckpt_dir, "selective_lora.pt")
        if os.path.exists(st_path):
            state = torch.load(st_path, map_location="cpu", weights_only=True)
            _, unexpected = transformer.load_state_dict(state, strict=False)
            if unexpected:
                raise ValueError(f"Unexpected keys: {unexpected}")
            st_params = {
                f"{name}.{ab}.weight"
                for name, m in transformer.named_modules()
                if isinstance(m, ExLoRALinear)
                for ab in ("lora_A", "lora_B")
            }
            matched = sum(1 for k in state if k in st_params)
            print(f"[Checkpoint] ExLoRA loaded: {matched}/{len(state)} keys matched "
                  f"({len(st_params)} ExLoRA params in model)")
            if matched == 0:
                raise RuntimeError("No ExLoRA keys matched — check module naming.")
            for m in transformer.modules():
                if isinstance(m, ExLoRALinear) and m.lora_B.weight.abs().sum() > 0:
                    print(f"[Checkpoint] ExLoRA B norm OK: {m.lora_B.weight.norm().item():.4f}")
                    break
            else:
                print("[Checkpoint] WARNING — all ExLoRA lora_B weights are zero")
        elif explicit_ckpt:
            # User explicitly pointed at a checkpoint dir but it has no
            # selective_lora.pt — that almost certainly means the user
            # picked an AnaLoRA-only run by mistake. Refuse to silently
            # run with zero-init ExLoRA.
            raise FileNotFoundError(
                f"selective_lora.pt missing from {ckpt_dir}\n"
                f"  Files present: {sorted(os.listdir(ckpt_dir))}\n"
                f"  Without this file ExLoRA is zero-initialised and has "
                f"no effect (st=0 == st=1).")
        if has_ckpt_base:
            load_analora_state_dict(transformer, bl_path, "cpu")
            print("[Checkpoint] AnaLoRA loaded from checkpoint")
    else:
        print("[Checkpoint] No checkpoint found — using freshly initialised weights")

    pipe = Flux2KleinPipeline(
        transformer=transformer, vae=vae,
        text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
    )
    pipe = pipe.to(device=device, dtype=torch.bfloat16)
    pipe.transformer.eval()
    return pipe, tokenizer, text_encoder


# ---------------------------------------------------------------------------
# LoRA mode context manager
# ---------------------------------------------------------------------------

SCALE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # overridden by --scale_values


@contextmanager
def lora_mode_ctx(transformer, analora_scale: float = 1.0, exlora_scale: float = 1.0):
    """Scale both LoRAs by the given multipliers (0.0 = off, 1.0 = full strength)."""
    base_mods = [m for m in transformer.modules() if type(m) is AnaLoRALinear]
    st_mods   = [m for m in transformer.modules() if isinstance(m, ExLoRALinear)]
    orig_base = [m.scaling for m in base_mods]
    for m, s in zip(base_mods, orig_base):
        m.scaling = analora_scale * s
    for m in st_mods:
        m.set_scaling(exlora_scale * m.alpha / m.r)
    try:
        yield
    finally:
        for m, s in zip(base_mods, orig_base):
            m.scaling = s
        for m in st_mods:
            m.reset_scaling()


# ---------------------------------------------------------------------------
# Text-embedding steering (Diffusion Sliders — Ekin & Gandelsman, arXiv:2603.17998)
# ---------------------------------------------------------------------------
# Paper's idea: construct a unit-norm steering direction in text-embedding space
# from a large contrastive dataset via difference-of-means (DoM), then add
# ``alpha * v`` to the prompt embeddings at concept-token positions.
#
# Here we replicate the paper's pipeline: per edit, Qwen3 generates N contrastive
# {pos, neg, pos_style, neg_style} sentence pairs, each pair is encoded via the
# same Qwen3 text encoder that the diffusion pipeline uses, pooled at the style-
# identifier token positions, and averaged into a unit-norm direction.
# See training/steering_dom.py for the generator/pool/DoM helpers.
#
# Multiplying by ``alpha`` and adding to the inference-prompt edit-token
# positions gives:
#     alpha > 0  : amplify edit_i
#     alpha < 0  : suppress edit_i  (complements ExLoRA-based suppression)
#     alpha = 0  : no-op (falls back to pure LoRA mode)
# ---------------------------------------------------------------------------


def templated_substring_indices(prompt: str, substr: str, tokenizer,
                                max_length: int = 512):
    """Thin wrapper around the canonical chat-template-aware substring
    locator. Returns ``[]`` when the substring is missing instead of
    raising — kept for backward compatibility with old callers in this
    file that treat "not found" as a soft signal.
    """
    try:
        return klein_find_substring_token_indices(
            prompt, substr, tokenizer,
            max_length=max_length,
            use_chat_template=True,
        )
    except AssertionError:
        return []


STEERING_SCOPES = ("tokens", "prompt")
STEERING_MODES  = ("add", "ablate")


def apply_csv(
    prompt_embeds: torch.Tensor,
    edit_positions: list,
    steering_vecs: list,
    alpha_per_edit: list,
    scope: str = "tokens",
    all_prompt_positions: list = None,
    mode: str = "add",
    max_push_norm_ratio: float = 0.0,
) -> torch.Tensor:
    """Return a steered copy of ``prompt_embeds`` using the chosen steering
    mode.

    scope
    -----
      - ``"tokens"`` : apply only at the suppressed edit's own token positions
        (``edit_positions[i]``). Same footprint as the ExLoRA mask.
      - ``"prompt"`` : apply at every non-padding token position in the prompt
        (``all_prompt_positions``).

    mode
    ----
      - ``"add"`` (default, original behaviour):
            x' = x + alpha * v̂
        Pure additive push. Requires a large ``alpha`` to overcome whatever
        component of the concept is already encoded in x.
      - ``"ablate"`` (directional ablation + optional push):
            x' = x - (x · v̂) v̂ + alpha * v̂
        First projects out the component of x along v̂ — i.e. *deletes* the
        concept from that 1-D subspace regardless of its magnitude — then
        adds a controlled anti-push of magnitude ``alpha``. With alpha = 0
        this is pure concept removal (Arditi et al.-style ablation). Usually
        dramatically stronger than ``"add"`` for the same ``|alpha|``.

    alpha_per_edit semantics
    ------------------------
    ``alpha_per_edit[i]`` controls edit ``i``:
      - ``None``  : skip this edit entirely (no push, no ablation). Use this
        for edits that are NOT being suppressed in the current sweep row.
      - ``0.0``   : no additive push. In ``mode="ablate"`` this still
        performs pure concept removal; in ``mode="add"`` it is a no-op.
      - any other float: scaled push / push-after-ablation magnitude.

    Important: passing ``alpha=0`` for a non-suppressed edit in ``"ablate"``
    mode would *still* project its concept out of the prompt. Use ``None``
    for edits you do not want to touch.

    max_push_norm_ratio
    -------------------
    When > 0, clamps the push magnitude so it never exceeds
    ``max_push_norm_ratio * ‖x_t‖`` (the norm of the embedding at each
    token position *before* any modification). This prevents extreme
    out-of-distribution embeddings that arise when ``|alpha|`` far exceeds
    the embedding norm — a common cause of paradoxical concept
    amplification (see degenerate-cache note in ``steering_dom.py``).
    Set to 0 (default) to disable clamping.

    v̂ must be unit-norm (which is what ``steering_dom.compute_dom_vector``
    already produces). ``alpha`` here is the pre-multiplied value
    ``alpha_i * embedding_scale_factor`` coming from the caller.

    When multiple edits are non-None (e.g. scope=="prompt" and multiple
    suppressed edits), each (ablate + push) is applied in sequence. If you
    want the edit vectors to be independent, Gram-Schmidt them upstream —
    this function treats each ``v_i`` as-is.
    """
    if scope not in STEERING_SCOPES:
        raise ValueError(f"Unknown steering scope {scope!r}; "
                         f"expected one of {STEERING_SCOPES}.")
    if mode not in STEERING_MODES:
        raise ValueError(f"Unknown steering mode {mode!r}; "
                         f"expected one of {STEERING_MODES}.")

    effective = [a for a in alpha_per_edit if a is not None]
    if not effective:
        return prompt_embeds
    # Additive mode with all zero alphas is a no-op. Ablation mode is
    # meaningful even at alpha==0 (pure concept removal), so don't skip.
    if mode == "add" and not any(abs(a) > 0 for a in effective):
        return prompt_embeds
    if not any(v is not None for v in steering_vecs):
        return prompt_embeds

    steered = prompt_embeds.clone()
    for positions, v, alpha in zip(edit_positions, steering_vecs, alpha_per_edit):
        if v is None or alpha is None:
            continue
        if mode == "add" and alpha == 0:
            continue  # nothing to add

        if scope == "prompt":
            target_positions = all_prompt_positions if all_prompt_positions else positions
        else:
            target_positions = positions
        if not target_positions:
            continue

        v_hat = v.to(dtype=steered.dtype)
        raw_push = (float(alpha) * v_hat) if alpha != 0 else None
        for t in target_positions:
            if t >= steered.shape[1]:
                continue
            x_t = steered[0, t, :]
            if max_push_norm_ratio > 0 and raw_push is not None:
                x_norm = x_t.norm().item()
                max_mag = max_push_norm_ratio * max(x_norm, 1e-6)
                push_mag = abs(float(alpha))
                if push_mag > max_mag:
                    push = (float(alpha) * max_mag / push_mag) * v_hat
                else:
                    push = raw_push
            else:
                push = raw_push
            if mode == "ablate":
                proj_scalar = torch.dot(x_t, v_hat)
                x_t = x_t - proj_scalar * v_hat
            if push is not None:
                x_t = x_t + push
            steered[0, t, :] = x_t
    return steered


# ``templated_prompt_token_positions`` lives in visual_analogy.utils.selective_lora
# under the canonical name ``klein_templated_prompt_token_positions``. We
# re-export it here as the historical name to keep call sites compact.
templated_prompt_token_positions = klein_templated_prompt_token_positions


def parse_embedding_schedule(raw):
    """Normalize an ``embedding_schedule`` spec into a sorted list of
    ``(step_boundary, alpha)`` tuples.

    Accepts any of:
      - ``None`` / empty -> returns ``None``
      - list of ``"boundary:alpha"`` strings  (CLI form)
      - list of ``[boundary, alpha]`` pairs    (YAML form)
      - list of ``{"until_step": b, "alpha": a}`` dicts (YAML form)

    Semantics: for denoising step index ``t`` (0-indexed), the first entry with
    ``t < boundary`` wins — i.e. piecewise-constant, left-closed / right-open.
    """
    if raw is None:
        return None
    try:
        items = list(raw)
    except TypeError:
        return None
    if not items:
        return None
    parsed = []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if not s:
                continue
            if ":" not in s:
                raise ValueError(
                    f"embedding_schedule entry {it!r} must be 'step_boundary:alpha'")
            a, b = s.split(":", 1)
            parsed.append((int(a), float(b)))
        elif isinstance(it, dict) or hasattr(it, "keys"):
            boundary = it.get("until_step", it.get("step", it.get("boundary")))
            alpha = it.get("alpha", it.get("value", it.get("weight")))
            if boundary is None or alpha is None:
                raise ValueError(
                    f"embedding_schedule entry {dict(it)!r} must have "
                    "keys 'until_step' and 'alpha'")
            parsed.append((int(boundary), float(alpha)))
        else:
            seq = list(it)
            if len(seq) != 2:
                raise ValueError(
                    f"embedding_schedule entry {it!r} must be a pair "
                    "[step_boundary, alpha]")
            parsed.append((int(seq[0]), float(seq[1])))
    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])
    return parsed


def alpha_at_step(schedule, step_idx: int) -> float:
    """Piecewise-constant lookup for the schedule."""
    for boundary, alpha in schedule:
        if step_idx < boundary:
            return alpha
    return schedule[-1][1]


def format_schedule_tag(schedule) -> str:
    """Short filename-safe descriptor for a schedule."""
    parts = []
    for boundary, alpha in schedule:
        tag_a = f"{alpha:+.1f}".replace(".", "p").replace("+", "")
        parts.append(f"{boundary}a{tag_a}")
    return "sched_" + "_".join(parts)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Max side of the longer dimension used during training.
# Keep in sync with configs/train_exlora_flux2_klein.yaml  resolution: 512
TRAIN_SIZE = 512


def _encode_image(pipe, img: Image.Image, resolution: int = TRAIN_SIZE):
    """VAE-encode ``img`` exactly as training does (``encode_single_image``):
    resized to ``resolution`` x ``resolution`` without keeping aspect ratio.
    The checkpoints only ever saw square, stretched inputs, so encoding at the
    native aspect ratio here would put them off-distribution."""
    return encode_single_image(pipe, img, resolution)


@torch.no_grad()
def run_inference(pipe, tokenizer, text_encoder,
                  image_a, image_a_prime, image_b,
                  prompt, texts_to_suppress, num_steps, device,
                  analora_scale: float = 1.0, exlora_scale: float = 1.0,
                  prompt_embeds_override: torch.Tensor = None,
                  prompt_embeds_schedule: list = None,
                  generator: "torch.Generator | None" = None):
    """
    texts_to_suppress: list of edit strings whose tokens are masked for ExLoRA.
                       Pass [] for no suppression.
    analora_scale / exlora_scale: multiplier in [0, 1] for each LoRA's strength.
    prompt_embeds_override: optional precomputed (possibly steered) prompt embeds
                            to use instead of re-encoding from `prompt`. When
                            provided, `prompt` is still used for token-mask
                            substring alignment.
    prompt_embeds_schedule: optional list of prompt_embed tensors (one per
                            denoising step). When given, the embedding fed to
                            the transformer is swapped per step — used to
                            implement a time-varying steering-alpha schedule.
    generator: optional ``torch.Generator`` used to sample the initial noise.
               Pass a freshly seeded generator per sweep cell to keep the
               initial noise fixed across the ablation axis — otherwise each
               call consumes the global RNG and every cell sees different
               noise, confounding the ablation.

    Images are encoded at TRAIN_SIZE x TRAIN_SIZE, matching training. The
    square output is resized back to B's size before it is returned.
    """
    dtype = torch.bfloat16

    lat_a       = _encode_image(pipe, image_a,       TRAIN_SIZE)
    lat_a_prime = _encode_image(pipe, image_a_prime, TRAIN_SIZE)
    lat_b       = _encode_image(pipe, image_b,       TRAIN_SIZE)

    if prompt_embeds_schedule is not None and len(prompt_embeds_schedule) > 0:
        if len(prompt_embeds_schedule) != num_steps:
            raise ValueError(
                f"prompt_embeds_schedule has {len(prompt_embeds_schedule)} entries "
                f"but num_steps={num_steps} — must be one embed per step.")
        prompt_embeds = prompt_embeds_schedule[0]
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(prompt_embeds.device)
    elif prompt_embeds_override is not None:
        prompt_embeds = prompt_embeds_override
        text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(prompt_embeds.device)
    else:
        prompt_embeds, text_ids = compute_text_embeddings([prompt], tokenizer, text_encoder)
    patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

    cond_a  = Flux2KleinPipeline._pack_latents(lat_a)
    cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
    cond_b  = Flux2KleinPipeline._pack_latents(lat_b)
    condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

    if generator is not None:
        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype,
                                  generator=generator)
    else:
        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
    gen_latents   = Flux2KleinPipeline._pack_latents(gen_latents)
    target_ids    = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
    condition_ids = Flux2KleinPipeline._prepare_image_ids([lat_a, lat_a_prime, lat_b]).to(device)
    combined_ids  = torch.cat([target_ids, condition_ids], dim=1)

    sigmas       = np.linspace(1.0, 1 / num_steps, num_steps)
    mu           = compute_empirical_mu(gen_latents.shape[1], num_steps)
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)

    # Always build the edit-token mask whenever we *would* use ExLoRA
    # suppression, even when ``exlora_scale == 0``.  Rely on ``lora_mode_ctx`` to
    # zero the ExLoRA delta.  (Previously we only set the mask when
    # ``exlora_scale > 0``, which left ``_token_mask is None`` at st=0; that
    # short-circuits ``ExLoRALinear`` to ``return self.base(x)`` without
    # ever going through the adapter path.  For scale=0 the output is the same
    # either way, but keeping the mask set makes the st=0 vs st=1 comparison
    # a single "adapter on with zero gain" vs "adapter on with full gain" path
    # and matches training when you zero scale with a non-None mask.)
    if texts_to_suppress:
        token_mask = build_token_mask(
            prompt, texts_to_suppress, tokenizer, device,
            max_seq_len=prompt_embeds.shape[1],
        )
        if exlora_scale > 0.0:
            print(f"[ExLoRA] token_mask covers {int(token_mask.sum().item())} "
                  f"templated-prompt position(s) for {len(texts_to_suppress)} "
                  f"suppressed edit text(s).")
    else:
        token_mask = None

    with lora_mode_ctx(pipe.transformer, analora_scale, exlora_scale), \
         exlora_token_mask_ctx(pipe.transformer, token_mask, disable_mask_after=True):
        for step_idx, t in enumerate(timesteps):
            step_embeds = (prompt_embeds_schedule[step_idx]
                           if prompt_embeds_schedule is not None
                           else prompt_embeds)
            latent_model_input = torch.cat([gen_latents, condition_latents], dim=1)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(1).to(dtype) / 1000,
                guidance=None,
                encoder_hidden_states=step_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0][:, :gen_latents.shape[1]]
            gen_latents = pipe.scheduler.step(noise_pred, t, gen_latents, return_dict=False)[0]

    unpacked = gen_latents.permute(0, 2, 1).reshape(1, -1, patch_h, patch_w)
    bn_mean  = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
    bn_std   = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1)
                          + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
    unpacked = unpacked * bn_std + bn_mean
    unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
    decoded  = pipe.vae.decode(unpacked, return_dict=False)[0]
    out = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return out.resize(image_b.size, Image.LANCZOS)


# ---------------------------------------------------------------------------
# Image annotation helper
# ---------------------------------------------------------------------------

def annotate_image(img, label, suppressed_texts, font_size=16):
    """Return a new image with a white banner below containing label + suppressed edits."""
    font       = _load_font(font_size)
    line_h     = font_size + 4
    col_w      = img.width
    chars_per_line = max(30, int(col_w / (font_size * 0.55)))

    lines = [label.replace("\n", " ")]
    if suppressed_texts:
        supp_str = "Suppressed: " + "; ".join(suppressed_texts)
        lines += textwrap.wrap(supp_str, chars_per_line)
    else:
        lines.append("Suppressed: (none)")

    banner_h = len(lines) * line_h + 8
    out = Image.new("RGB", (col_w, img.height + banner_h), (255, 255, 255))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        color = (160, 0, 0) if i > 0 and suppressed_texts else (0, 0, 0)
        draw.text((4, img.height + 4 + i * line_h), line,
                  fill=color, font=font)
    return out


# ---------------------------------------------------------------------------
# Combined 2-D grid builder
# ---------------------------------------------------------------------------

def build_combined_grid(image_a, image_a_prime, image_b,
                        row_descs,    # [(label, suppress_list), ...]
                        all_preds,    # dict: (row_label, st_s) -> PIL.Image
                        scale_values, # list of exlora scale floats
                        cell_size=256,
                        analora_image=None):
    """
    Build a single 2-D grid:
      Rows    : suppression modes
      Columns : exlora_scale values (left → right: 0 → 1)
    Plus a reference strip (A | A' | B | analora) and column headers above the grid.
    """
    n_rows = len(row_descs)
    n_cols = len(scale_values)
    font_size = 14
    font      = _load_font(font_size)
    line_h    = font_size + 4

    # Cell dims follow B's aspect ratio so outputs and references share shape.
    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32               # title banner
    col_hdr_h   = 24               # column header row
    row_hdr_w   = 140              # left margin for row labels
    ref_strip_h = cell_h + 20

    ref_images = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if analora_image is not None:
        ref_images.append((analora_image, "analora"))
    n_ref_cols = len(ref_images)

    total_w = row_hdr_w + max(n_cols, n_ref_cols) * cell_w
    total_h = header_h + ref_strip_h + col_hdr_h + n_rows * cell_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # --- Title banner ---
    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    draw.text((6, 8), "ExLoRA ablation:  suppress mode (rows) × scale (cols)",
              fill=(0, 0, 100), font=font)

    # --- Reference thumbnails: A | A' | B | analora (same cell size) ---
    ref_y = header_h
    for ci, (ref_img, lbl) in enumerate(ref_images):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    # --- Column headers: exlora scale ---
    col_hdr_y = header_h + ref_strip_h
    draw.rectangle([0, col_hdr_y, row_hdr_w - 1, col_hdr_y + col_hdr_h - 1],
                   fill=(230, 230, 230))
    draw.text((4, col_hdr_y + 4), "supp \\ scale", fill=(80, 80, 80), font=font)
    for ci, sv in enumerate(scale_values):
        x = row_hdr_w + ci * cell_w
        draw.rectangle([x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                       fill=(220, 240, 255))
        draw.text((x + 4, col_hdr_y + 4), f"st={sv:.2g}", fill=(0, 0, 0), font=font)

    # --- Grid cells ---
    grid_y0 = col_hdr_y + col_hdr_h
    for ri, (row_label, suppress) in enumerate(row_descs):
        y  = grid_y0 + ri * cell_h
        bg = (245, 245, 255) if ri % 2 == 0 else (235, 235, 250)
        draw.rectangle([0, y, row_hdr_w - 1, y + cell_h - 1], fill=bg)
        draw.text((4, y + 4), row_label, fill=(0, 0, 100), font=font)
        if suppress:
            supp_txt = "supp: " + "; ".join(s[:30] for s in suppress)
            for li, part in enumerate(textwrap.wrap(supp_txt, 18)):
                draw.text((4, y + 4 + (li + 1) * line_h), part,
                          fill=(160, 0, 0), font=font)

        for ci, sv in enumerate(scale_values):
            x    = row_hdr_w + ci * cell_w
            pred = all_preds.get((row_label, sv))
            if pred is not None:
                img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
            draw.text((x + 3, y + 3), f"st={sv:.2g}", fill=(255, 255, 200), font=font)

    return img


# ---------------------------------------------------------------------------
# 3-axis grid: suppression row → block of (exlora × embedding) sub-grids
# ---------------------------------------------------------------------------

def build_joint_grid(image_a, image_a_prime, image_b,
                     row_descs,     # [(label, suppress_list, suppress_idx), ...]
                     all_preds,     # dict: (row_label, st_s, emb_s) -> PIL.Image
                     st_scales,     # list of exlora floats (inner rows)
                     emb_scales,    # list of embedding alphas (inner cols)
                     cell_size=256,
                     analora_image=None):
    """
    Outer rows   : suppression mode
    Inner grid   : exlora_scale (rows, top→bottom: first→last) × embedding α (cols)
    Reference strip A | A' | B | analora on top.
    """
    n_sup = len(row_descs)
    n_st  = len(st_scales)
    n_emb = len(emb_scales)
    font_size = 14
    font      = _load_font(font_size)
    line_h    = font_size + 4

    # Cells follow B's aspect ratio so references and outputs share pixel size.
    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32
    col_hdr_h   = 24
    row_hdr_w   = 170               # outer row label
    inner_lbl_w = 54                # inner row label (exlora scale)
    ref_strip_h = cell_h + 20

    ref_images = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if analora_image is not None:
        ref_images.append((analora_image, "analora"))
    n_ref_cols = len(ref_images)

    inner_w = inner_lbl_w + n_emb * cell_w
    inner_h = col_hdr_h + n_st  * cell_h
    total_w = max(row_hdr_w + inner_w, row_hdr_w + n_ref_cols * cell_w)
    total_h = header_h + ref_strip_h + n_sup * inner_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    draw.text(
        (6, 8),
        "Joint ablation — outer rows: suppress mode | inner rows: ExLoRA scale "
        "| inner cols: text-embedding alpha",
        fill=(0, 0, 100), font=font,
    )

    # Reference thumbnails A | A' | B | analora (same cell size as outputs)
    ref_y = header_h
    for ci, (ref_img, lbl) in enumerate(ref_images):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    inner_y0 = header_h + ref_strip_h
    for ri, (row_label, suppress, _idx) in enumerate(row_descs):
        outer_y = inner_y0 + ri * inner_h

        # Outer row label cell
        bg = (245, 245, 255) if ri % 2 == 0 else (235, 235, 250)
        draw.rectangle(
            [0, outer_y, row_hdr_w - 1, outer_y + inner_h - 1], fill=bg)
        draw.text((4, outer_y + 4), row_label, fill=(0, 0, 100), font=font)
        if suppress:
            supp_txt = "supp: " + "; ".join(s[:24] for s in suppress)
            for li, part in enumerate(textwrap.wrap(supp_txt, 22)):
                draw.text(
                    (4, outer_y + 4 + (li + 1) * line_h),
                    part, fill=(160, 0, 0), font=font)

        # Inner column header (embedding alphas)
        col_hdr_y = outer_y
        draw.rectangle(
            [row_hdr_w, col_hdr_y,
             row_hdr_w + inner_lbl_w - 1, col_hdr_y + col_hdr_h - 1],
            fill=(230, 230, 230))
        draw.text((row_hdr_w + 3, col_hdr_y + 4),
                  "st\\emb", fill=(80, 80, 80), font=font)
        for ci, emb_s in enumerate(emb_scales):
            x = row_hdr_w + inner_lbl_w + ci * cell_w
            draw.rectangle(
                [x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                fill=(220, 240, 255))
            draw.text((x + 4, col_hdr_y + 4),
                      f"emb={emb_s:.2g}", fill=(0, 0, 0), font=font)

        # Inner grid cells
        cells_y0 = col_hdr_y + col_hdr_h
        for ii, st_s in enumerate(st_scales):
            y = cells_y0 + ii * cell_h

            # Inner row label (exlora scale)
            draw.rectangle(
                [row_hdr_w, y, row_hdr_w + inner_lbl_w - 1, y + cell_h - 1],
                fill=(245, 245, 245))
            draw.text((row_hdr_w + 3, y + 4),
                      f"st={st_s:.2g}", fill=(0, 0, 0), font=font)

            for ci, emb_s in enumerate(emb_scales):
                x = row_hdr_w + inner_lbl_w + ci * cell_w
                pred = all_preds.get((row_label, st_s, emb_s))
                if pred is not None:
                    img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
                draw.text(
                    (x + 3, y + 3),
                    f"st={st_s:.2g},emb={emb_s:.2g}",
                    fill=(255, 255, 200), font=font)

    return img


# ---------------------------------------------------------------------------
# Per-row grid: one suppression mode — rows=ExLoRA scale, cols=embedding alpha
# ---------------------------------------------------------------------------

def build_row_grid(image_a, image_a_prime, image_b,
                   row_label, suppress,
                   row_preds,    # dict: (st_s, emb_s) -> PIL.Image
                   st_scales, emb_scales,
                   cell_size=256,
                   analora_image=None):
    """Save a self-contained grid for a single suppression row.
    Layout: rows = ExLoRA scale, cols = embedding alpha.
    Same visual style as ``build_joint_grid``'s inner block, plus a reference
    strip (A | A' | B | analora) at the top.
    """
    n_st  = len(st_scales)
    n_emb = len(emb_scales)
    font_size = 14
    font      = _load_font(font_size)

    # Cells follow B's aspect ratio so references and outputs share pixel size.
    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32
    col_hdr_h   = 24
    row_hdr_w   = 70               # inner row label (st scale)
    ref_strip_h = cell_h + 20

    ref_images = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if analora_image is not None:
        ref_images.append((analora_image, "analora"))
    n_ref_cols = len(ref_images)

    total_w = row_hdr_w + max(n_emb, n_ref_cols) * cell_w
    total_h = header_h + ref_strip_h + col_hdr_h + n_st * cell_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    title = f"{row_label}  |  rows: ExLoRA scale  |  cols: embedding alpha"
    draw.text((6, 8), title, fill=(0, 0, 100), font=font)

    # Reference thumbnails A | A' | B | analora (same pixel size as output cells)
    ref_y = header_h
    for ci, (ref_img, lbl) in enumerate(ref_images):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    # Column headers (embedding alpha)
    col_hdr_y = header_h + ref_strip_h
    draw.rectangle([0, col_hdr_y, row_hdr_w - 1, col_hdr_y + col_hdr_h - 1],
                   fill=(230, 230, 230))
    draw.text((4, col_hdr_y + 4), "st\\emb", fill=(80, 80, 80), font=font)
    for ci, emb_s in enumerate(emb_scales):
        x = row_hdr_w + ci * cell_w
        draw.rectangle([x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                       fill=(220, 240, 255))
        draw.text((x + 4, col_hdr_y + 4),
                  f"emb={emb_s:.2g}", fill=(0, 0, 0), font=font)

    # Grid cells
    grid_y0 = col_hdr_y + col_hdr_h
    for ii, st_s in enumerate(st_scales):
        y = grid_y0 + ii * cell_h
        draw.rectangle([0, y, row_hdr_w - 1, y + cell_h - 1],
                       fill=(245, 245, 245))
        draw.text((4, y + 4), f"st={st_s:.2g}", fill=(0, 0, 0), font=font)

        for ci, emb_s in enumerate(emb_scales):
            x = row_hdr_w + ci * cell_w
            pred = row_preds.get((st_s, emb_s))
            if pred is not None:
                img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
            draw.text((x + 3, y + 3),
                      f"st={st_s:.2g},emb={emb_s:.2g}",
                      fill=(255, 255, 200), font=font)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--image_a",       type=str, required=True)
    parser.add_argument("--image_a_prime", type=str, required=True)
    parser.add_argument("--image_b",       type=str, required=True)
    parser.add_argument("--edits", type=str, nargs="+", required=True,
                        help="Raw edit texts in order (e.g. 'change expression to laugh'). "
                             "The prompt is auto-built from these using the same format as training.")
    parser.add_argument("--scale_values", type=float, nargs="+",
                        default=None,
                        help="LoRA scale sweep values (default: 0 0.25 0.5 0.75 1).")
    parser.add_argument("--embedding_scales", type=float, nargs="+",
                        default=None,
                        help="Text-embedding steering alpha values, applied at suppressed "
                             "edits' token positions (Diffusion-Sliders-style). "
                             "alpha>0 amplifies the edit; alpha<0 suppresses. "
                             "Default [0.0] = pure LoRA interpolation (no embedding steering).")
    parser.add_argument("--embedding_schedule", type=str, nargs="+", default=None,
                        help="Piecewise-constant schedule for the embedding alpha "
                             "across denoising steps. Each entry is "
                             "'step_boundary:alpha'; for step index t the first "
                             "entry with t<boundary wins. Example: "
                             "'10:-40 20:-30 28:-20' uses alpha=-40 for steps 0-9, "
                             "-30 for 10-19, -20 for 20-27. "
                             "When set, overrides --embedding_scales and runs one "
                             "inference per (suppress-mode × exlora-scale). "
                             "The schedule can also be set in the YAML config as "
                             "'embedding_schedule: [[10, -40], [20, -30], [28, -20]]'.")
    parser.add_argument("--embedding_scale_factor", type=float, default=1.0,
                        help="Absolute magnitude multiplier on the unit-norm steering "
                             "direction. Since edit tokens in prompt_embeds can have norm "
                             "~10-50, alpha=1 corresponds to a unit-length push. Raise this "
                             "to amplify steering effect (typical: 5-30).")
    parser.add_argument("--csv_scope", type=str, nargs="+",
                        default=None, choices=list(STEERING_SCOPES),
                        help="Where to add the steering vector in the prompt "
                             "embedding: 'tokens' (only the suppressed edit's "
                             "own token positions — same footprint as ExLoRA), "
                             "or 'prompt' (every non-padding prompt token). "
                             "Pass multiple values to sweep both as an "
                             "ablation — each scope gets its own output "
                             "subdirectory. Default: 'tokens'. "
                             "Can also be set in YAML as "
                             "'csv_scope: \"tokens\"' or "
                             "'csv_scope: [\"tokens\", \"prompt\"]'.")
    parser.add_argument("--csv_mode", type=str, default=None,
                        choices=list(STEERING_MODES),
                        help="How the steering vector modifies the prompt "
                             "embedding at target positions. "
                             "'add' (default): x' = x + alpha*v_hat — pure "
                             "additive push (original behaviour). "
                             "'ablate': x' = x - (x.v_hat)*v_hat + alpha*v_hat "
                             "— directional ablation (project the concept "
                             "component OUT of x) then optional push. Usually "
                             "much stronger for suppression at the same |alpha|. "
                             "With alpha=0, 'ablate' performs pure concept "
                             "removal. Can also be set in YAML as "
                             "'csv_mode: \"ablate\"'.")
    parser.add_argument("--num_steering_pairs", type=int, default=32,
                        help="Number of contrastive pos/neg sentence pairs per edit that "
                             "Qwen3 generates (Diffusion-Sliders paper uses 100 via GPT-4o; "
                             "we default to 32 with a local Qwen to keep runtime modest).")
    parser.add_argument("--steering_cache_dir", type=str, default="./steering_cache",
                        help="Directory in which to cache generated pairs.jsonl and "
                             "vector.pt per edit (keyed by sha1(edit)).")
    parser.add_argument("--regen_steering", action="store_true",
                        help="Regenerate the contrastive pairs + DoM vector even if a cached "
                             "vector.pt exists for the edit.")
    parser.add_argument("--max_push_norm_ratio", type=float, default=0.0,
                        help="Clamp the steering push magnitude to this multiple of the "
                             "per-token embedding norm (before modification). Prevents "
                             "extreme out-of-distribution embeddings. 0 = no clamp (default). "
                             "Typical safe value: 1.0-3.0.")
    parser.add_argument("--incontext_steering", action="store_true",
                        help="Compute the DoM steering direction inside the actual inference "
                             "prompt (in-context) instead of from isolated short sentences. "
                             "Fixes the causal-LM context mismatch: the direction is extracted "
                             "in the same representation space where it is applied. Requires "
                             "1+8 text-encoder forward passes per edit (fast) and no Qwen "
                             "generation.")
    parser.add_argument("--output",         type=str, default="result.png")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Explicit checkpoint directory to load ExLoRA/AnaLoRA from. "
                             "Overrides the auto-discovery from output_dir in the config.")
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--cell_size", type=int, default=512)
    parser.add_argument("--seed",      type=int, default=0)
    cli = parser.parse_args()

    cfg = OmegaConf.load(cli.config)
    cfg.num_steps  = cli.num_steps
    cfg.cell_size  = cli.cell_size
    cfg.seed       = cli.seed
    cfg.image_a         = cli.image_a
    cfg.image_a_prime   = cli.image_a_prime
    cfg.image_b         = cli.image_b
    cfg.edits           = cli.edits
    # Build the prompt the same way training does — guarantees edits are exact substrings
    cfg.prompt          = build_analogy_prompt(cli.edits, cli.edits)
    cfg.output          = cli.output
    cfg.checkpoint_dir  = cli.checkpoint_dir
    cfg.scale_values    = cli.scale_values if cli.scale_values is not None else SCALE_VALUES
    cfg.embedding_scales = (
        cli.embedding_scales if cli.embedding_scales is not None else [0.0])
    cfg.embedding_scale_factor = cli.embedding_scale_factor
    cfg.num_steering_pairs     = cli.num_steering_pairs
    cfg.steering_cache_dir     = cli.steering_cache_dir
    cfg.regen_steering         = cli.regen_steering
    cfg.incontext_steering     = cli.incontext_steering
    cfg.max_push_norm_ratio    = cli.max_push_norm_ratio

    # Schedule: CLI overrides YAML. Store as a plain list of (boundary, alpha)
    # tuples on cfg.embedding_schedule_parsed; leave the raw form on
    # cfg.embedding_schedule for traceability.
    raw_sched = cli.embedding_schedule
    if raw_sched is None:
        raw_sched = OmegaConf.to_container(cfg.get("embedding_schedule"), resolve=True) \
            if "embedding_schedule" in cfg else None
    cfg.embedding_schedule_parsed = parse_embedding_schedule(raw_sched)

    # Steering scope: CLI overrides YAML. Normalize to a list of valid choices.
    if cli.csv_scope is not None:
        raw_scopes = cli.csv_scope
    elif "csv_scope" in cfg:
        raw_scopes = OmegaConf.to_container(cfg.csv_scope, resolve=True)
    else:
        raw_scopes = ["tokens"]
    if isinstance(raw_scopes, str):
        raw_scopes = [raw_scopes]
    scopes = []
    for s in raw_scopes:
        s = str(s).strip().lower()
        if s not in STEERING_SCOPES:
            raise ValueError(
                f"csv_scope entry {s!r} invalid; "
                f"expected one of {STEERING_SCOPES}.")
        if s not in scopes:   # dedupe, preserve order
            scopes.append(s)
    cfg.csv_scopes = scopes

    # Steering mode (single value): CLI overrides YAML, default "add".
    if cli.csv_mode is not None:
        mode = cli.csv_mode
    elif "csv_mode" in cfg:
        mode = str(cfg.csv_mode)
    else:
        mode = "add"
    mode = str(mode).strip().lower()
    if mode not in STEERING_MODES:
        raise ValueError(f"csv_mode={mode!r} invalid; "
                         f"expected one of {STEERING_MODES}.")
    cfg.csv_mode = mode
    return cfg


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    image_a       = Image.open(args.image_a).convert("RGB")
    image_a_prime = Image.open(args.image_a_prime).convert("RGB")
    image_b       = Image.open(args.image_b).convert("RGB")

    print("Loading models …")
    pipe, tokenizer, text_encoder = load_pipeline(args, device)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Precompute base prompt embeds ----
    with torch.no_grad():
        base_prompt_embeds, _ = compute_text_embeddings(
            [args.prompt], tokenizer, text_encoder)
    max_seq_len = base_prompt_embeds.shape[1]

    # ---- AnaLoRA-only reference image (ExLoRA off) ----
    print("[analora] Running AnaLoRA-only inference for the reference strip …")
    analora_gen = torch.Generator(device=device).manual_seed(int(args.seed))
    analora_image = run_inference(
        pipe, tokenizer, text_encoder,
        image_a, image_a_prime, image_b,
        args.prompt, [],
        args.num_steps, device,
        analora_scale=1.0,
        exlora_scale=0.0,
        prompt_embeds_override=base_prompt_embeds,
        generator=analora_gen,
    )
    analora_ref_path = out_dir / "analora.png"
    analora_image.save(analora_ref_path)
    print(f"  Saved analora reference → {analora_ref_path}")

    # ---- Resolve embedding schedule (piecewise-constant alpha over steps) ----
    schedule = args.get("embedding_schedule_parsed", None)
    if schedule:
        print(f"[Steering schedule] Using time-varying alpha schedule: "
              f"{[(b, a) for b, a in schedule]} "
              f"(overrides --embedding_scales)")

    # ---- Per-edit: templated token positions in the inference prompt ----
    # These positions index into ``base_prompt_embeds`` (which is computed on
    # the Qwen chat-templated sequence), so we align the edit substring against
    # the templated text to get the correct positions for the steering add.
    edit_positions = []
    # We need DoM vectors whenever any steering will happen. For additive
    # mode this only matters when at least one alpha is non-zero. For ablate
    # mode it matters always — pure concept removal (alpha=0) is still
    # meaningful.
    steering_mode_for_gate = str(args.get("csv_mode", "add"))
    any_alpha = (
        steering_mode_for_gate == "ablate"
        or any(abs(float(a)) > 0 for a in args.embedding_scales)
        or (schedule is not None and any(abs(a) > 0 for _, a in schedule))
    )
    for i, edit_text in enumerate(args.edits):
        pos = templated_substring_indices(
            args.prompt, edit_text, tokenizer, max_length=max_seq_len)
        edit_positions.append(pos)
        # Sanity check: decode tokens at those positions and confirm they match
        # the edit text (they should; mismatch means the templated alignment
        # silently failed).
        if pos:
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            templated_ids = tokenizer(
                templated_prompt, max_length=max_seq_len,
                padding="max_length", truncation=True,
                return_tensors="pt")["input_ids"][0].tolist()
            decoded = tokenizer.decode([templated_ids[p] for p in pos],
                                       skip_special_tokens=False)
            print(f"[Steering] edit_{i+1} positions[{len(pos)}]: "
                  f"{pos[0]}..{pos[-1]}  decoded={decoded!r}")
        else:
            print(f"[Steering] edit_{i+1} ({args.edits[i][:50]}): "
                  "NO tokens aligned in templated prompt — steering disabled for this edit.")

    # ---- Per-edit: DoM steering vector via Qwen3 + caching ----
    steering_vecs = [None] * len(args.edits)
    use_incontext = bool(args.get("incontext_steering", False))
    if use_incontext:
        print("[Steering DoM] In-context mode enabled — DoM will be computed "
              "inside the full inference prompt (causal-LM-aware).")
    if any_alpha:
        cache_root = Path(args.steering_cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        hidden_dim = base_prompt_embeds.shape[-1]
        for i, edit_text in enumerate(args.edits):
            if not edit_positions[i]:
                print(f"[Steering DoM] Skipping edit_{i+1} — no positions in prompt.")
                continue
            incontext_kwargs = {}
            if use_incontext:
                incontext_kwargs = dict(
                    incontext_prompt=args.prompt,
                    all_edits=list(args.edits),
                    edit_index=i,
                    build_prompt_fn=build_analogy_prompt,
                )
            v, meta = get_or_build_steering_vector(
                edit=edit_text,
                n_pairs=int(args.num_steering_pairs),
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                device=device,
                cache_root=cache_root,
                regen=bool(args.regen_steering),
                max_sequence_length=max_seq_len,
                **incontext_kwargs,
            )
            if v.shape[-1] != hidden_dim:
                raise RuntimeError(
                    f"Steering vector dim {v.shape[-1]} != prompt_embeds dim "
                    f"{hidden_dim} for edit_{i+1}. Cache from a different "
                    f"encoder config? Pass --regen_steering.")
            steering_vecs[i] = v.to(device=device, dtype=base_prompt_embeds.dtype)
            n_pooled = meta.get("n_usable_after_pooling", "?")
            pre = meta.get("pre_normalization_norm", None)
            src = meta.get("source", "?")
            pre_str = f"{pre:.3f}" if isinstance(pre, (int, float)) else str(pre)
            print(f"[Steering DoM] edit_{i+1} ({edit_text[:50]}): "
                  f"source={src} |v|_pre={pre_str} pooled_pairs={n_pooled} "
                  f"dim={v.shape[-1]}")
    else:
        print("[Steering DoM] All embedding_scales == 0 and mode='add' — "
              "skipping DoM vector computation (pure LoRA mode).")

    # Row descriptors: (row_label, suppress_list, suppress_indices)
    # "no supp" row is excluded — it doesn't demonstrate selective suppression.
    # Ablation is only on ExLoRA suppression; analora is always on at scale=1.
    row_descs = []
    n = len(args.edits)
    for r in range(1, n + 1):          # start at 1 to skip "no supp"
        for indices in itertools.combinations(range(n), r):
            subset = [args.edits[i] for i in indices]
            if r == n and n > 1:
                label = "exlora (supp all)"
            else:
                label = "exlora (supp " + "+".join(f"e{i+1}" for i in indices) + ")"
            row_descs.append((label, subset, list(indices)))

    # When a schedule is active, the per-step alpha comes from the schedule;
    # emb_scales collapses to a single placeholder key so per-row bookkeeping
    # and grid layout (which expects a scalar emb axis) still work.
    if schedule:
        emb_scales = [0.0]
        sched_tag = format_schedule_tag(schedule)
    else:
        emb_scales = list(args.embedding_scales)
        sched_tag = None
    st_scales  = list(args.scale_values)

    # Steering scope ablation axis. "tokens" touches only the suppressed edit's
    # own token positions (same footprint as ExLoRA); "prompt" touches every
    # non-padding prompt token (to push against concept residue that leaked
    # into neighbouring tokens through the encoder's self-attention).
    scopes = list(args.csv_scopes)
    all_prompt_positions = None
    if "prompt" in scopes:
        all_prompt_positions = templated_prompt_token_positions(
            args.prompt, tokenizer, max_length=max_seq_len)
        print(f"[Steering] scope='prompt' will add the vector at "
              f"{len(all_prompt_positions)} non-padding token positions "
              f"(positions {all_prompt_positions[0]}..{all_prompt_positions[-1]}).")
    print(f"[Steering] Scope ablation: {scopes}")
    mode = str(args.csv_mode)
    print(f"[Steering] Mode: {mode}  "
          + ("(additive push: x' = x + alpha * v_hat)"
             if mode == "add"
             else "(directional ablation + push: "
                  "x' = x - (x . v_hat) v_hat + alpha * v_hat)"))

    total_runs = len(scopes) * len(row_descs) * len(st_scales) * len(emb_scales)
    run_idx    = 0
    output_path = Path(args.output)
    combined_paths = []

    for scope in scopes:
        # Per-scope output subdir only when we are actually running multiple
        # scopes; otherwise keep the original flat layout for backward compat.
        if len(scopes) > 1:
            scope_out_dir = out_dir / f"scope_{scope}"
            scope_out_dir.mkdir(parents=True, exist_ok=True)
            scope_output  = scope_out_dir / output_path.name
        else:
            scope_out_dir = out_dir
            scope_output  = output_path
        scope_tag_str = f"scope={scope}"

        all_preds = {}   # (row_label, st_s, emb_s) -> PIL.Image  (per-scope)

        for row_label, suppress, suppress_idx in row_descs:
            row_dir = scope_out_dir / row_label.replace(" ", "_")\
                .replace("/", "-").replace("(", "").replace(")", "")
            row_dir.mkdir(parents=True, exist_ok=True)

            base_s = 1.0       # analora is always at full strength
            row_preds = {}
            for emb_s in emb_scales:
                if schedule:
                    embeds_schedule = []
                    for step_idx in range(int(args.num_steps)):
                        a_t = alpha_at_step(schedule, step_idx)
                        # None for non-suppressed edits → leave their concept
                        # untouched even in ``mode="ablate"`` (where alpha=0
                        # would otherwise mean "pure concept removal").
                        alpha_per_edit = [
                            (a_t * args.embedding_scale_factor) if i in suppress_idx else None
                            for i in range(n)
                        ]
                        embeds_schedule.append(apply_csv(
                            base_prompt_embeds, edit_positions, steering_vecs,
                            alpha_per_edit,
                            scope=scope,
                            all_prompt_positions=all_prompt_positions,
                            mode=mode,
                            max_push_norm_ratio=float(args.max_push_norm_ratio)))
                    steered_embeds = None
                    step_tag = sched_tag
                else:
                    # None for non-suppressed edits → see note above.
                    alpha_per_edit = [
                        (emb_s * args.embedding_scale_factor) if i in suppress_idx else None
                        for i in range(n)
                    ]
                    steered_embeds = apply_csv(
                        base_prompt_embeds, edit_positions, steering_vecs,
                        alpha_per_edit,
                        scope=scope,
                        all_prompt_positions=all_prompt_positions,
                        mode=mode,
                        max_push_norm_ratio=float(args.max_push_norm_ratio))
                    embeds_schedule = None
                    step_tag = f"emb={emb_s:.2g}"

                for st_s in st_scales:
                    run_idx += 1
                    print(f"[{run_idx}/{total_runs}] {scope_tag_str}  "
                          f"{row_label}  suppress={suppress}  "
                          f"base={base_s}  st={st_s}  {step_tag}")
                    # Freshly-seeded generator per cell so the INITIAL NOISE
                    # is identical across the ablation axis. Without this the
                    # global RNG advances with every torch.randn call and each
                    # cell sees different noise — you cannot tell whether an
                    # image changed because of alpha or because of noise.
                    cell_gen = torch.Generator(device=device).manual_seed(
                        int(args.seed))
                    pred = run_inference(
                        pipe, tokenizer, text_encoder,
                        image_a, image_a_prime, image_b,
                        args.prompt, suppress,
                        args.num_steps, device,
                        analora_scale=base_s,
                        exlora_scale=st_s,
                        prompt_embeds_override=steered_embeds,
                        prompt_embeds_schedule=embeds_schedule,
                        generator=cell_gen,
                    )
                    all_preds[(row_label, st_s, emb_s)] = pred
                    row_preds[(st_s, emb_s)] = pred

                    annotated = annotate_image(
                        pred,
                        f"{row_label}  st={st_s:.2g}  {step_tag}  {scope_tag_str}",
                        suppress,
                    )
                    if schedule:
                        fname = (f"st{st_s:.2f}".replace(".", "p")
                                 + f"_{sched_tag}.png")
                    else:
                        fname = (f"st{st_s:.2f}".replace(".", "p")
                                 + f"_emb{emb_s:.2f}".replace(".", "p") + ".png")
                    annotated.save(row_dir / fname)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Per-row grid
            row_grid = build_row_grid(
                image_a, image_a_prime, image_b,
                f"{row_label}  [{scope_tag_str}]", suppress,
                row_preds, st_scales, emb_scales,
                cell_size=args.cell_size,
                analora_image=analora_image,
            )
            row_grid.save(row_dir / "grid.png")
            print(f"  Saved row grid → {row_dir / 'grid.png'}")

        # Combined grid for this scope
        if len(emb_scales) == 1:
            legacy_preds = {(rl, s): all_preds[(rl, s, emb_scales[0])]
                            for rl, _, _ in row_descs for s in st_scales}
            combined = build_combined_grid(
                image_a, image_a_prime, image_b,
                [(rl, sup) for rl, sup, _ in row_descs],
                legacy_preds, st_scales, cell_size=args.cell_size,
                analora_image=analora_image,
            )
        else:
            combined = build_joint_grid(
                image_a, image_a_prime, image_b,
                row_descs, all_preds, st_scales, emb_scales,
                cell_size=args.cell_size,
                analora_image=analora_image,
            )
        combined.save(scope_output)
        combined_paths.append(scope_output)
        print(f"[{scope_tag_str}] Saved combined grid → {scope_output}")

    print("Individual images saved under:")
    for p in combined_paths:
        print(f"  {p.parent}/")


if __name__ == "__main__":
    main()
