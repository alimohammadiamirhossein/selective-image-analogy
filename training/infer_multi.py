#!/usr/bin/env python3
"""
infer_multi.py

Multi-seed analogy inference with all suppression modes.

Extends infer_single.py with a multi-seed AnaLoRA preview:
  1. Flux generates N candidates (one per seed in --seeds) using a cheap
     partial-denoise schedule (--partial_denoise_steps, default 4 steps).
  2. All latents are concatenated and decoded in ONE batched VAE.decode().
  3. A Qwen-VL judge picks the best seed by pairwise analogy comparison:
     shows (A, A', B, cand_i, cand_j) and asks which candidate best applies
     the A→A' edit to B.  The winning seed is then used for ALL subsequent
     suppression/ExLoRA/steering sweeps (apples-to-apples ablation).

Given A, A', B and a list of edits [e1, e2, ..., eN], runs:
  - analora          : best seed from multi-seed preview, full denoise
  - exlora (supp e1)   : ExLoRA on, mask e1 tokens
  - exlora (supp e2)   : ExLoRA on, mask e2 tokens
  ...
  - exlora (supp all)  : ExLoRA on, mask all edit tokens

Saves a strip:  A | A' | B | analora | exlora_e1 | ... | exlora_all

Usage:
  python training/infer_multi.py \\
      --config   training/configs/train_exlora_flux2_klein.yaml \\
      --image_a       /path/to/A.png \\
      --image_a_prime /path/to/A_prime.png \\
      --image_b       /path/to/B.png \\
      --edits "X" "Y" "Z" \\
      --seeds 7 30 42 73 \\
      --partial_denoise_steps 4 \\
      --output /tmp/result.png
"""

import argparse
import base64
import io
import itertools
import json
import os
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from diffusers import Flux2KleinPipeline
from transformers import AutoModelForCausalLM, AutoProcessor

from visual_analogy.utils import (
    exlora_token_mask_ctx,
)
from visual_analogy.utils.selective_lora import (
    klein_templated_prompt_token_positions,
)
from training.train_exlora_flux2_klein import (
    build_analogy_prompt,
    build_token_mask,
    compute_empirical_mu,
    compute_text_embeddings,
    retrieve_timesteps,
)
from training.infer_single import (
    _cell_dims,
    _encode_image,
    _fit_thumbnail,
    _load_font,
    alpha_at_step,
    annotate_image,
    format_schedule_tag,
    load_pipeline,
    lora_mode_ctx,
    parse_embedding_schedule,
    templated_substring_indices,
)
from training.steering_dom import get_or_build_steering_vector


# ---------------------------------------------------------------------------
# Scale sweep values
# ---------------------------------------------------------------------------

SCALE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Text-embedding steering
# ---------------------------------------------------------------------------

STEERING_SCOPES = ("tokens", "prompt")
STEERING_MODES  = ("add", "ablate")


templated_prompt_token_positions = klein_templated_prompt_token_positions


def apply_csv(
    prompt_embeds: torch.Tensor,
    edit_positions: list,
    steering_vecs: list,
    alpha_per_edit: list,
    scope: str = "tokens",
    all_prompt_positions: list = None,
    mode: str = "add",
) -> torch.Tensor:
    if scope not in STEERING_SCOPES:
        raise ValueError(f"Unknown steering scope {scope!r}; expected one of {STEERING_SCOPES}.")
    if mode not in STEERING_MODES:
        raise ValueError(f"Unknown steering mode {mode!r}; expected one of {STEERING_MODES}.")

    effective = [a for a in alpha_per_edit if a is not None]
    if not effective:
        return prompt_embeds
    if mode == "add" and not any(abs(a) > 0 for a in effective):
        return prompt_embeds
    if not any(v is not None for v in steering_vecs):
        return prompt_embeds

    steered = prompt_embeds.clone()
    for positions, v, alpha in zip(edit_positions, steering_vecs, alpha_per_edit):
        if v is None or alpha is None:
            continue
        if mode == "add" and alpha == 0:
            continue
        if scope == "prompt":
            target_positions = all_prompt_positions if all_prompt_positions else positions
        else:
            target_positions = positions
        if not target_positions:
            continue
        v_hat = v.to(dtype=steered.dtype)
        push = (float(alpha) * v_hat) if alpha != 0 else None
        for t in target_positions:
            if t >= steered.shape[1]:
                continue
            x_t = steered[0, t, :]
            if mode == "ablate":
                proj_scalar = torch.dot(x_t, v_hat)
                x_t = x_t - proj_scalar * v_hat
            if push is not None:
                x_t = x_t + push
            steered[0, t, :] = x_t
    return steered


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

TRAIN_SIZE = 512


@torch.no_grad()
def run_inference(pipe, tokenizer, text_encoder,
                  image_a, image_a_prime, image_b,
                  prompt, texts_to_suppress, num_steps, device,
                  analora_scale: float = 1.0, exlora_scale: float = 1.0,
                  prompt_embeds_override: torch.Tensor = None,
                  prompt_embeds_schedule: list = None,
                  generator: "torch.Generator | None" = None,
                  partial_denoise_steps: "int | None" = None,
                  return_latents: bool = False):
    """
    texts_to_suppress  : edit strings whose tokens are masked for ExLoRA.
    analora_scale / exlora_scale : LoRA multipliers (0=off, 1=full).
    prompt_embeds_override : optional precomputed (steered) prompt embeds.
    prompt_embeds_schedule : one embed per denoising step (time-varying alpha).
    generator          : torch.Generator for reproducible initial noise.
    partial_denoise_steps : when set to K (0 < K < num_steps), only the first K
        sigmas of the full schedule are used; the K-th step becomes a single jump
        to sigma=0.  Used for cheap multi-seed previews.  The mu shift is still
        computed with the full num_steps so shapes remain consistent.
    return_latents     : when True, skip VAE decode and return
        (packed_latent, patch_h, patch_w).  Use this to batch multiple seeds
        into a single decode_packed_latents_batch() call.
    """
    dtype = torch.bfloat16

    lat_a       = _encode_image(pipe, image_a,       TRAIN_SIZE)
    lat_a_prime = _encode_image(pipe, image_a_prime, TRAIN_SIZE)
    lat_b       = _encode_image(pipe, image_b,       TRAIN_SIZE)

    # -- figure out active step count for schedule validation --
    schedule_len_required = (
        int(partial_denoise_steps)
        if (partial_denoise_steps is not None
            and 0 < int(partial_denoise_steps) < num_steps)
        else num_steps
    )

    if prompt_embeds_schedule is not None and len(prompt_embeds_schedule) > 0:
        if len(prompt_embeds_schedule) != schedule_len_required:
            raise ValueError(
                f"prompt_embeds_schedule has {len(prompt_embeds_schedule)} entries "
                f"but the active step schedule has {schedule_len_required} "
                f"(num_steps={num_steps}, partial_denoise_steps={partial_denoise_steps})")
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
        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype, generator=generator)
    else:
        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
    gen_latents   = Flux2KleinPipeline._pack_latents(gen_latents)
    target_ids    = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
    condition_ids = Flux2KleinPipeline._prepare_image_ids([lat_a, lat_a_prime, lat_b]).to(device)
    combined_ids  = torch.cat([target_ids, condition_ids], dim=1)

    sigmas_full  = np.linspace(1.0, 1 / num_steps, num_steps)
    mu           = compute_empirical_mu(gen_latents.shape[1], num_steps)
    if (partial_denoise_steps is not None
            and 0 < int(partial_denoise_steps) < num_steps):
        sigmas = sigmas_full[: int(partial_denoise_steps)]
    else:
        sigmas = sigmas_full
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)

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

    if return_latents:
        return gen_latents.detach(), patch_h, patch_w

    decoded_list = decode_packed_latents_batch(pipe, gen_latents, patch_h, patch_w,
                                               out_size=image_b.size)
    return decoded_list[0]


def decode_packed_latents_batch(pipe, packed_latents: torch.Tensor,
                                patch_h: int, patch_w: int, out_size=None):
    """Decode a batch of packed denoised latents in a SINGLE VAE.decode() call.

    Parameters
    ----------
    packed_latents : (B, seq_len, hidden_dim) — stacked gen_latents from
        run_inference(..., return_latents=True).
    patch_h, patch_w : int — spatial grid dims returned by run_inference.
    out_size : optional (w, h) to resize each image to, e.g. ``image_b.size``
        (generation runs at the square training resolution).

    Returns
    -------
    list[PIL.Image.Image]  — one image per batch entry.
    """
    B = packed_latents.shape[0]
    unpacked = packed_latents.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
    bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
    bn_std  = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1)
                         + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
    unpacked = unpacked * bn_std + bn_mean
    unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
    decoded  = pipe.vae.decode(unpacked, return_dict=False)[0].detach()
    out = pipe.image_processor.postprocess(decoded, output_type="pil")
    if not isinstance(out, list):
        out = [out]
    if out_size is not None:
        out = [img.resize(out_size, Image.LANCZOS) for img in out]
    return out


# ---------------------------------------------------------------------------
# Qwen-VL judge (inline, no external package dependency)
# ---------------------------------------------------------------------------

def _pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


class _QwenVLJudge:
    """Lazy-loaded Qwen3-VL judge for pairwise seed selection.

    Loads on first call to compare(), then can be unloaded explicitly
    via unload() to free GPU memory before the main ablation sweep.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
                 device: str = "cuda"):
        self.model_id = model_id
        self.device   = device
        self._model   = None
        self._proc    = None

    def _load(self):
        if self._model is not None:
            return
        print(f"[QwenVL Judge] Loading {self.model_id} ...")
        self._proc  = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map=self.device)
        self._model.eval()
        print("[QwenVL Judge] Loaded.")

    def unload(self):
        if self._model is not None:
            del self._model, self._proc
            self._model = None
            self._proc  = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[QwenVL Judge] Unloaded.")

    def _compare_pair(self, img_a_ref: Image.Image, img_ap_ref: Image.Image,
                      img_b: Image.Image,
                      cand_left: Image.Image, cand_right: Image.Image,
                      apply_edits: list) -> str:
        """Ask the VL model which of LEFT/RIGHT better applies A→A' to B.

        Returns "A" (left wins), "B" (right wins), or "tie".
        """
        self._load()
        edits_str = "; ".join(apply_edits)
        question = (
            "You are comparing two image editing results.\n"
            "The first image shows the SOURCE (A), the second shows the EDITED SOURCE (A'), "
            "and the third shows the TARGET (B) to be edited.\n"
            "Images A→A' define the visual edit: '" + edits_str + "'.\n"
            "The fourth image is candidate A and the fifth is candidate B — "
            "each is an attempt to apply the same A→A' edit to B.\n\n"
            "Which candidate (A or B) better applies the edit from A→A' to B?\n"
            "Answer with a single letter: A or B."
        )

        # Build a simple text-only prompt if image input fails, as fallback.
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_a_ref},
                    {"type": "image", "image": img_ap_ref},
                    {"type": "image", "image": img_b},
                    {"type": "image", "image": cand_left},
                    {"type": "image", "image": cand_right},
                    {"type": "text",  "text": question},
                ]
            }]
            text = self._proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self._proc.process_vision_info(messages)
            inputs = self._proc(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(self._model.device)
        except Exception:
            # Fallback: try without vision-info helper (different API version).
            try:
                text_only = (
                    f"Image editing task.\n"
                    f"Edit: {edits_str}.\n"
                    f"Candidate A or B is better? Answer with a single letter: A or B."
                )
                inputs = self._proc(
                    text=[text_only], return_tensors="pt"
                ).to(self._model.device)
            except Exception as exc:
                print(f"[QwenVL Judge] Encoding failed: {exc} — returning tie.")
                return "tie"

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        # Strip prompt tokens from output.
        input_len = inputs["input_ids"].shape[1]
        gen_ids   = out_ids[:, input_len:]
        answer    = self._proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        first = answer.upper()[:1]
        if first == "A":
            return "A"
        if first == "B":
            return "B"
        return "tie"

    def pick_best_by_pairwise(
        self,
        candidate_images: list,
        base_image: Image.Image,
        apply_edits: list,
        analogy_pair: "tuple[Image.Image, Image.Image] | None" = None,
        both_orders: bool = True,
    ) -> "tuple[int, list[int], list[dict], list]":
        """Pairwise-voting selection over ``candidate_images``.

        Each unordered pair (i, j) is compared once (or twice when
        ``both_orders=True``, to mitigate position bias).  The candidate
        with the most wins is returned.

        Returns
        -------
        best_idx    : int
        votes       : list[int]  — cumulative wins per candidate
        pairwise_log: list[dict] — per-comparison record
        fallback    : list[None] — placeholder for API compat
        """
        self._load()
        n = len(candidate_images)
        votes = [0] * n
        pairwise_log = []
        img_a_ref  = analogy_pair[0] if analogy_pair else base_image
        img_ap_ref = analogy_pair[1] if analogy_pair else base_image

        for i in range(n):
            for j in range(i + 1, n):
                # Order 1: left=i, right=j
                ans_ab = self._compare_pair(
                    img_a_ref, img_ap_ref, base_image,
                    candidate_images[i], candidate_images[j],
                    apply_edits)
                if ans_ab == "A":
                    votes[i] += 1
                elif ans_ab == "B":
                    votes[j] += 1

                ans_ba = None
                if both_orders:
                    # Order 2: left=j, right=i  (swap to catch position bias)
                    ans_ba_raw = self._compare_pair(
                        img_a_ref, img_ap_ref, base_image,
                        candidate_images[j], candidate_images[i],
                        apply_edits)
                    ans_ba = ans_ba_raw
                    if ans_ba_raw == "A":
                        votes[j] += 1
                    elif ans_ba_raw == "B":
                        votes[i] += 1

                pairwise_log.append({
                    "left_idx": i, "right_idx": j,
                    "ans_ab": ans_ab, "ans_ba": ans_ba,
                })

        best_idx = max(range(n), key=lambda k: (votes[k], -k))
        return best_idx, votes, pairwise_log, [None] * n


# ---------------------------------------------------------------------------
# Multi-seed AnaLoRA preview + Qwen judge
# ---------------------------------------------------------------------------

def _run_analora_multi_seed(pipe, tokenizer, text_encoder,
                               image_a, image_a_prime, image_b,
                               args, out_dir: Path, device):
    """Multi-seed partial-denoise baseline with batched decode + Qwen-VL judge.

    Pipeline
    --------
    1. For each seed in args.seeds, run run_inference() with ExLoRA off,
       no embedding steering, partial_denoise_steps=args.partial_denoise_steps,
       and return_latents=True.
    2. Concatenate all latents and call decode_packed_latents_batch() ONCE
       (amortises the most expensive step across seeds).
    3. Save every per-seed image to out_dir/analora_seeds/seed_NN.png.
    4. Optionally load the Qwen-VL judge and pick the best seed by pairwise
       analogy comparison (A:A'::B:? with the candidate pairs).
    5. Re-run the winning seed with the FULL num_steps schedule and save as
       out_dir/analora.png.

    Returns
    -------
    (PIL.Image, int)  — (chosen_img, chosen_seed)
    """
    seeds    = list(args.seeds)
    partial_K = args.get("partial_denoise_steps", None)
    judge_enabled = bool(args.get("qwen_judge_enabled", True))
    judge_model   = str(args.get("qwen_judge_model", "Qwen/Qwen3-VL-8B-Instruct"))

    if partial_K:
        sched_msg = (f"partial_denoise_steps={partial_K} "
                     f"(first {partial_K} sigmas of {args.num_steps}-step schedule)")
    else:
        sched_msg = f"full {args.num_steps}-step schedule"
    print(f"[AnaLoRA] Multi-seed baseline (ExLoRA off, no steering): "
          f"seeds={seeds}  {sched_msg}")

    seeds_dir = out_dir / "analora_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1+2: cheap per-seed inference → batched decode --
    packed_latents_per_seed = []
    patch_hw = None
    for s in seeds:
        gen = torch.Generator(device=device).manual_seed(int(s))
        packed, ph, pw = run_inference(
            pipe, tokenizer, text_encoder,
            image_a, image_a_prime, image_b,
            args.prompt, [],
            args.num_steps, device,
            analora_scale=1.0,
            exlora_scale=0.0,
            generator=gen,
            partial_denoise_steps=partial_K,
            return_latents=True,
        )
        packed_latents_per_seed.append(packed)
        if patch_hw is None:
            patch_hw = (ph, pw)
        elif patch_hw != (ph, pw):
            raise RuntimeError(
                f"Inconsistent patch dims across seeds: {patch_hw} vs {(ph, pw)}")

    batched_latents = torch.cat(packed_latents_per_seed, dim=0)
    print(f"[AnaLoRA] Batched decode: {batched_latents.shape[0]} seeds "
          f"in a single VAE.decode() call.")
    decoded = decode_packed_latents_batch(pipe, batched_latents, *patch_hw,
                                          out_size=image_b.size)
    if len(decoded) != len(seeds):
        raise RuntimeError(
            f"Batched decode produced {len(decoded)} images for {len(seeds)} seeds.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Step 3: save per-seed previews --
    seed_paths = []
    for s, img in zip(seeds, decoded):
        p = seeds_dir / f"seed_{int(s):02d}.png"
        img.save(p)
        seed_paths.append((int(s), img, p))
        print(f"[AnaLoRA]   seed={s:>3d} -> {p}")

    # -- Step 4: Qwen-VL pairwise judge --
    best_idx     = 0
    chosen_seed  = seeds[0]
    chosen_img   = decoded[0]
    votes        = None
    pairwise_log = None
    fallback_scores = None

    if judge_enabled and len(seeds) > 1:
        apply_edits = list(args.edits)
        judge = _QwenVLJudge(model_id=judge_model, device="cuda")
        try:
            n_pairs = len(seeds) * (len(seeds) - 1) // 2
            print(f"[AnaLoRA] Picking best of {len(seeds)} candidates by "
                  f"pairwise analogy comparison ({n_pairs} pairs × 2 orders) ...")
            best_idx, votes, pairwise_log, fallback_scores = judge.pick_best_by_pairwise(
                candidate_images=[img for _, img, _ in seed_paths],
                base_image=image_b,
                apply_edits=apply_edits,
                analogy_pair=(image_a, image_a_prime),
                both_orders=True,
            )
            chosen_seed = seed_paths[best_idx][0]
            chosen_img  = seed_paths[best_idx][1]
        finally:
            judge.unload()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        max_votes = (len(seeds) - 1) * 2
        for (s, _, _), v in zip(seed_paths, votes):
            mark = "  <- best" if s == chosen_seed else ""
            print(f"[AnaLoRA]   seed={s:>3d}  votes={v}/{max_votes}{mark}")
        for entry in pairwise_log:
            i, j = entry["left_idx"], entry["right_idx"]
            si, sj = seed_paths[i][0], seed_paths[j][0]
            print(f"[AnaLoRA]   pair seed={si} vs seed={sj}: "
                  f"A→B={entry.get('ans_ab')}  B→A={entry.get('ans_ba')}")
        print(f"[AnaLoRA] Best seed={chosen_seed} (votes={votes[best_idx]}/{max_votes}).")
    elif not judge_enabled:
        print(f"[AnaLoRA] Qwen judge disabled — using first seed ({chosen_seed}).")
    else:
        print("[AnaLoRA] Only one seed — skipping judge.")

    # -- Step 5: full-denoise re-run of the winning seed --
    print(f"[AnaLoRA] Full denoising of chosen seed={chosen_seed} "
          f"({args.num_steps} steps, ExLoRA off) ...")
    full_gen = torch.Generator(device=device).manual_seed(int(chosen_seed))
    chosen_img = run_inference(
        pipe, tokenizer, text_encoder,
        image_a, image_a_prime, image_b,
        args.prompt, [],
        args.num_steps, device,
        analora_scale=1.0,
        exlora_scale=0.0,
        generator=full_gen,
        partial_denoise_steps=None,
        return_latents=False,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chosen_path = out_dir / "analora.png"
    chosen_img.save(chosen_path)
    print(f"[AnaLoRA] Saved best (fully denoised) -> {chosen_path}")

    # Persist a record of the pairwise comparison for downstream analysis.
    pairwise_record = {
        "seeds": [int(s) for s in seeds],
        "votes": ([int(v) for v in votes] if votes is not None else None),
        "max_votes": ((len(seeds) - 1) * 2 if judge_enabled and len(seeds) > 1 else 0),
        "chosen_seed": int(chosen_seed),
        "chosen_index": int(best_idx),
        "judge_enabled": bool(judge_enabled),
        "judge_model": judge_model if judge_enabled else None,
        "pairwise_log": pairwise_log,
        "apply_edits": list(args.edits),
        "partial_denoise_steps": (None if partial_K is None else int(partial_K)),
        "num_steps_reference": int(args.num_steps),
    }
    (out_dir / "analora_scores.json").write_text(json.dumps(pairwise_record, indent=2))

    return chosen_img, int(chosen_seed)


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def build_combined_grid(image_a, image_a_prime, image_b,
                        row_descs, all_preds, scale_values,
                        cell_size=256, base_pred=None):
    n_rows = len(row_descs)
    n_cols = len(scale_values)
    font_size = 14
    font      = _load_font(font_size)
    line_h    = font_size + 4

    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32
    col_hdr_h   = 24
    row_hdr_w   = 140
    ref_strip_h = cell_h + 20

    n_ref_cols = 4 if base_pred is not None else 3
    total_w = row_hdr_w + max(n_ref_cols, n_cols) * cell_w
    total_h = header_h + ref_strip_h + col_hdr_h + n_rows * cell_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    draw.text((6, 8), "ExLoRA ablation:  suppress mode (rows) × scale (cols)",
              fill=(0, 0, 100), font=font)

    ref_y = header_h
    ref_items = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if base_pred is not None:
        ref_items.append((base_pred, "analora"))
    for ci, (ref_img, lbl) in enumerate(ref_items):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    col_hdr_y = header_h + ref_strip_h
    draw.rectangle([0, col_hdr_y, row_hdr_w - 1, col_hdr_y + col_hdr_h - 1],
                   fill=(230, 230, 230))
    draw.text((4, col_hdr_y + 4), "supp \\ scale", fill=(80, 80, 80), font=font)
    for ci, sv in enumerate(scale_values):
        x = row_hdr_w + ci * cell_w
        draw.rectangle([x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                       fill=(220, 240, 255))
        draw.text((x + 4, col_hdr_y + 4), f"st={sv:.2g}", fill=(0, 0, 0), font=font)

    grid_y0 = col_hdr_y + col_hdr_h
    for ri, (row_label, suppress) in enumerate(row_descs):
        y  = grid_y0 + ri * cell_h
        bg = (245, 245, 255) if ri % 2 == 0 else (235, 235, 250)
        draw.rectangle([0, y, row_hdr_w - 1, y + cell_h - 1], fill=bg)
        draw.text((4, y + 4), row_label, fill=(0, 0, 100), font=font)
        if suppress:
            supp_txt = "supp: " + "; ".join(s[:30] for s in suppress)
            for li, part in enumerate(textwrap.wrap(supp_txt, 18)):
                draw.text((4, y + 4 + (li + 1) * line_h), part, fill=(160, 0, 0), font=font)
        for ci, sv in enumerate(scale_values):
            x    = row_hdr_w + ci * cell_w
            pred = all_preds.get((row_label, sv))
            if pred is not None:
                img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
            draw.text((x + 3, y + 3), f"st={sv:.2g}", fill=(255, 255, 200), font=font)

    return img


def build_joint_grid(image_a, image_a_prime, image_b,
                     row_descs, all_preds, st_scales, emb_scales,
                     cell_size=256, base_pred=None):
    n_sup = len(row_descs)
    n_st  = len(st_scales)
    n_emb = len(emb_scales)
    font_size = 14
    font      = _load_font(font_size)
    line_h    = font_size + 4

    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32
    col_hdr_h   = 24
    row_hdr_w   = 170
    inner_lbl_w = 54
    ref_strip_h = cell_h + 20

    n_ref_cols = 4 if base_pred is not None else 3

    inner_w = inner_lbl_w + n_emb * cell_w
    inner_h = col_hdr_h + n_st  * cell_h
    total_w = max(row_hdr_w + inner_w, row_hdr_w + n_ref_cols * cell_w)
    total_h = header_h + ref_strip_h + n_sup * inner_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    draw.text((6, 8),
              "Joint ablation — outer rows: suppress | inner rows: ExLoRA scale "
              "| inner cols: text-embedding alpha",
              fill=(0, 0, 100), font=font)

    ref_y = header_h
    ref_items = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if base_pred is not None:
        ref_items.append((base_pred, "analora"))
    for ci, (ref_img, lbl) in enumerate(ref_items):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    inner_y0 = header_h + ref_strip_h
    for ri, (row_label, suppress, _idx) in enumerate(row_descs):
        outer_y = inner_y0 + ri * inner_h
        bg = (245, 245, 255) if ri % 2 == 0 else (235, 235, 250)
        draw.rectangle([0, outer_y, row_hdr_w - 1, outer_y + inner_h - 1], fill=bg)
        draw.text((4, outer_y + 4), row_label, fill=(0, 0, 100), font=font)
        if suppress:
            supp_txt = "supp: " + "; ".join(s[:24] for s in suppress)
            for li, part in enumerate(textwrap.wrap(supp_txt, 22)):
                draw.text((4, outer_y + 4 + (li + 1) * line_h), part,
                          fill=(160, 0, 0), font=font)

        col_hdr_y = outer_y
        draw.rectangle(
            [row_hdr_w, col_hdr_y,
             row_hdr_w + inner_lbl_w - 1, col_hdr_y + col_hdr_h - 1],
            fill=(230, 230, 230))
        draw.text((row_hdr_w + 3, col_hdr_y + 4), "st\\emb", fill=(80, 80, 80), font=font)
        for ci, emb_s in enumerate(emb_scales):
            x = row_hdr_w + inner_lbl_w + ci * cell_w
            draw.rectangle(
                [x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                fill=(220, 240, 255))
            draw.text((x + 4, col_hdr_y + 4), f"emb={emb_s:.2g}", fill=(0, 0, 0), font=font)

        cells_y0 = col_hdr_y + col_hdr_h
        for ii, st_s in enumerate(st_scales):
            y = cells_y0 + ii * cell_h
            draw.rectangle(
                [row_hdr_w, y, row_hdr_w + inner_lbl_w - 1, y + cell_h - 1],
                fill=(245, 245, 245))
            draw.text((row_hdr_w + 3, y + 4), f"st={st_s:.2g}", fill=(0, 0, 0), font=font)
            for ci, emb_s in enumerate(emb_scales):
                x = row_hdr_w + inner_lbl_w + ci * cell_w
                pred = all_preds.get((row_label, st_s, emb_s))
                if pred is not None:
                    img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
                draw.text((x + 3, y + 3), f"st={st_s:.2g},emb={emb_s:.2g}",
                          fill=(255, 255, 200), font=font)

    return img


def build_row_grid(image_a, image_a_prime, image_b,
                   row_label, suppress, row_preds,
                   st_scales, emb_scales, cell_size=256, base_pred=None):
    n_st  = len(st_scales)
    n_emb = len(emb_scales)
    font_size = 14
    font      = _load_font(font_size)

    cell_w, cell_h = _cell_dims(image_b, cell_size)

    header_h    = 32
    col_hdr_h   = 24
    row_hdr_w   = 70
    ref_strip_h = cell_h + 20

    n_ref_cols = 4 if base_pred is not None else 3

    total_w = row_hdr_w + max(n_emb, n_ref_cols) * cell_w
    total_h = header_h + ref_strip_h + col_hdr_h + n_st * cell_h

    img  = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    title = f"{row_label}  |  rows: ExLoRA scale  |  cols: embedding alpha"
    draw.text((6, 8), title, fill=(0, 0, 100), font=font)

    ref_y = header_h
    ref_items = [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]
    if base_pred is not None:
        ref_items.append((base_pred, "analora"))
    for ci, (ref_img, lbl) in enumerate(ref_items):
        x = row_hdr_w + ci * cell_w
        thumb = _fit_thumbnail(ref_img, cell_w, cell_h)
        img.paste(thumb, (x, ref_y + 20))
        draw.text((x + 4, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    col_hdr_y = header_h + ref_strip_h
    draw.rectangle([0, col_hdr_y, row_hdr_w - 1, col_hdr_y + col_hdr_h - 1],
                   fill=(230, 230, 230))
    draw.text((4, col_hdr_y + 4), "st\\emb", fill=(80, 80, 80), font=font)
    for ci, emb_s in enumerate(emb_scales):
        x = row_hdr_w + ci * cell_w
        draw.rectangle([x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                       fill=(220, 240, 255))
        draw.text((x + 4, col_hdr_y + 4), f"emb={emb_s:.2g}", fill=(0, 0, 0), font=font)

    grid_y0 = col_hdr_y + col_hdr_h
    for ii, st_s in enumerate(st_scales):
        y = grid_y0 + ii * cell_h
        draw.rectangle([0, y, row_hdr_w - 1, y + cell_h - 1], fill=(245, 245, 245))
        draw.text((4, y + 4), f"st={st_s:.2g}", fill=(0, 0, 0), font=font)
        for ci, emb_s in enumerate(emb_scales):
            x = row_hdr_w + ci * cell_w
            pred = row_preds.get((st_s, emb_s))
            if pred is not None:
                img.paste(_fit_thumbnail(pred, cell_w, cell_h), (x, y))
            draw.text((x + 3, y + 3), f"st={st_s:.2g},emb={emb_s:.2g}",
                      fill=(255, 255, 200), font=font)

    return img


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, required=True)
    parser.add_argument("--image_a",       type=str, required=True)
    parser.add_argument("--image_a_prime", type=str, required=True)
    parser.add_argument("--image_b",       type=str, required=True)
    parser.add_argument("--edits", type=str, nargs="+", required=True,
                        help="Raw edit texts in order. The prompt is auto-built from these.")
    parser.add_argument("--scale_values", type=float, nargs="+", default=None,
                        help="LoRA scale sweep values (default: 0 0.25 0.5 0.75 1).")
    parser.add_argument("--embedding_scales", type=float, nargs="+", default=None)
    parser.add_argument("--embedding_schedule", type=str, nargs="+", default=None)
    parser.add_argument("--embedding_scale_factor", type=float, default=1.0)
    parser.add_argument("--csv_scope", type=str, nargs="+",
                        default=None, choices=list(STEERING_SCOPES))
    parser.add_argument("--csv_mode", type=str, default=None,
                        choices=list(STEERING_MODES))
    parser.add_argument("--num_steering_pairs", type=int, default=32)
    parser.add_argument("--steering_cache_dir", type=str, default="./steering_cache")
    parser.add_argument("--regen_steering", action="store_true")
    parser.add_argument("--output",         type=str, default="result.png")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--cell_size", type=int, default=512)
    parser.add_argument("--seed",      type=int, default=0,
                        help="Fallback seed (used only when --no_qwen_judge and len(seeds)==1).")
    # ---- Multi-seed arguments ----
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds for the multi-seed AnaLoRA preview. Flux runs a cheap "
                             "partial-denoise for each seed; Qwen-VL picks the best one. "
                             "Default (from YAML or fallback): [7, 30, 42, 73].")
    parser.add_argument("--partial_denoise_steps", type=int, default=None,
                        help="Number of denoising steps for the cheap per-seed preview "
                             "(0 < K < --num_steps). The K-th step jumps straight to sigma=0. "
                             "Default: 4.")
    parser.add_argument("--qwen_judge_model", type=str, default=None,
                        help="HF model id for the Qwen-VL seed judge. "
                             "Default: 'Qwen/Qwen3-VL-8B-Instruct'.")
    parser.add_argument("--no_qwen_judge", action="store_true",
                        help="Disable the Qwen-VL judge. The first seed becomes analora.png.")

    cli = parser.parse_args()

    cfg = OmegaConf.load(cli.config)
    cfg.num_steps  = cli.num_steps
    cfg.cell_size  = cli.cell_size
    cfg.seed       = cli.seed
    cfg.image_a         = cli.image_a
    cfg.image_a_prime   = cli.image_a_prime
    cfg.image_b         = cli.image_b
    cfg.edits           = cli.edits
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

    raw_sched = cli.embedding_schedule
    if raw_sched is None:
        raw_sched = OmegaConf.to_container(cfg.get("embedding_schedule"), resolve=True) \
            if "embedding_schedule" in cfg else None
    cfg.embedding_schedule_parsed = parse_embedding_schedule(raw_sched)

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
            raise ValueError(f"csv_scope entry {s!r} invalid.")
        if s not in scopes:
            scopes.append(s)
    cfg.csv_scopes = scopes

    if cli.csv_mode is not None:
        mode = cli.csv_mode
    elif "csv_mode" in cfg:
        mode = str(cfg.csv_mode)
    else:
        mode = "add"
    mode = str(mode).strip().lower()
    if mode not in STEERING_MODES:
        raise ValueError(f"csv_mode={mode!r} invalid.")
    cfg.csv_mode = mode

    # ---- Multi-seed config ----
    if cli.seeds is not None:
        seeds = list(cli.seeds)
    elif "seeds" in cfg:
        seeds = [int(s) for s in OmegaConf.to_container(cfg.seeds, resolve=True)]
    else:
        seeds = [7, 30, 42, 73]
    if not seeds:
        raise ValueError("`seeds` must be a non-empty list of ints.")
    cfg.seeds = seeds

    if cli.partial_denoise_steps is not None:
        cfg.partial_denoise_steps = int(cli.partial_denoise_steps)
    elif "partial_denoise_steps" in cfg and cfg.partial_denoise_steps is not None:
        cfg.partial_denoise_steps = int(cfg.partial_denoise_steps)
    else:
        cfg.partial_denoise_steps = 4
    if cfg.partial_denoise_steps is not None and cfg.partial_denoise_steps <= 0:
        cfg.partial_denoise_steps = None
    if (cfg.partial_denoise_steps is not None
            and cfg.partial_denoise_steps >= cli.num_steps):
        cfg.partial_denoise_steps = None

    if cli.no_qwen_judge:
        cfg.qwen_judge_enabled = False
    elif "qwen_judge_enabled" in cfg:
        cfg.qwen_judge_enabled = bool(cfg.qwen_judge_enabled)
    else:
        cfg.qwen_judge_enabled = True

    if cli.qwen_judge_model is not None:
        cfg.qwen_judge_model = str(cli.qwen_judge_model)
    elif "qwen_judge_model" in cfg:
        cfg.qwen_judge_model = str(cfg.qwen_judge_model)
    else:
        cfg.qwen_judge_model = "Qwen/Qwen3-VL-8B-Instruct"

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # ---- Multi-seed AnaLoRA baseline ----
    # Runs N cheap partial-denoise previews (one per seed), decodes them in a
    # single batched VAE call, then lets the Qwen-VL judge pick the best seed.
    # The winning seed is then used for ALL subsequent ablation sweep cells
    # so every "with ExLoRA" result is directly comparable to analora.png
    # (same initial noise, only the method changes).
    analora_pred, chosen_seed = _run_analora_multi_seed(
        pipe, tokenizer, text_encoder,
        image_a, image_a_prime, image_b,
        args=args, out_dir=out_dir, device=device,
    )
    sweep_seed = int(chosen_seed)
    print(f"[Sweep] All ablation cells will use seed={sweep_seed} "
          f"(Qwen-VL winner; overrides --seed={int(args.seed)}).")

    # ---- Resolve embedding schedule ----
    schedule = args.get("embedding_schedule_parsed", None)
    if schedule:
        print(f"[Steering schedule] {[(b, a) for b, a in schedule]}")

    # ---- Per-edit: token positions + DoM vectors ----
    edit_positions = []
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
                  "NO tokens aligned — steering disabled for this edit.")

    steering_vecs = [None] * len(args.edits)
    if any_alpha:
        cache_root = Path(args.steering_cache_dir).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        hidden_dim = base_prompt_embeds.shape[-1]
        for i, edit_text in enumerate(args.edits):
            if not edit_positions[i]:
                continue
            v, meta = get_or_build_steering_vector(
                edit=edit_text,
                n_pairs=int(args.num_steering_pairs),
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                device=device,
                cache_root=cache_root,
                regen=bool(args.regen_steering),
                max_sequence_length=max_seq_len,
            )
            if v.shape[-1] != hidden_dim:
                raise RuntimeError(
                    f"Steering vector dim {v.shape[-1]} != prompt_embeds dim {hidden_dim}.")
            steering_vecs[i] = v.to(device=device, dtype=base_prompt_embeds.dtype)
            n_pooled = meta.get("n_usable_after_pooling", "?")
            pre = meta.get("pre_normalization_norm", None)
            src = meta.get("source", "?")
            pre_str = f"{pre:.3f}" if isinstance(pre, (int, float)) else str(pre)
            print(f"[Steering DoM] edit_{i+1} ({edit_text[:50]}): "
                  f"source={src} |v|_pre={pre_str} pooled={n_pooled} dim={v.shape[-1]}")
    else:
        print("[Steering DoM] All embedding_scales == 0 and mode='add' — skipping DoM.")

    # ---- Row descriptors ----
    row_descs = []
    n = len(args.edits)
    for r in range(1, n + 1):
        for indices in itertools.combinations(range(n), r):
            subset = [args.edits[i] for i in indices]
            if r == n and n > 1:
                label = "exlora (supp all)"
            else:
                label = "exlora (supp " + "+".join(f"e{i+1}" for i in indices) + ")"
            row_descs.append((label, subset, list(indices)))

    if schedule:
        emb_scales = [0.0]
        sched_tag  = format_schedule_tag(schedule)
    else:
        emb_scales = list(args.embedding_scales)
        sched_tag  = None
    st_scales = list(args.scale_values)

    scopes = list(args.csv_scopes)
    all_prompt_positions = None
    if "prompt" in scopes:
        all_prompt_positions = templated_prompt_token_positions(
            args.prompt, tokenizer, max_length=max_seq_len)
        print(f"[Steering] scope='prompt' covers "
              f"{len(all_prompt_positions)} non-padding positions.")
    print(f"[Steering] Scope: {scopes}")
    mode = str(args.csv_mode)
    print(f"[Steering] Mode: {mode}")

    total_runs = len(scopes) * len(row_descs) * len(st_scales) * len(emb_scales)
    run_idx    = 0
    output_path = Path(args.output)
    combined_paths = []

    for scope in scopes:
        if len(scopes) > 1:
            scope_out_dir = out_dir / f"scope_{scope}"
            scope_out_dir.mkdir(parents=True, exist_ok=True)
            scope_output  = scope_out_dir / output_path.name
        else:
            scope_out_dir = out_dir
            scope_output  = output_path
        scope_tag_str = f"scope={scope}"

        all_preds = {}

        for row_label, suppress, suppress_idx in row_descs:
            row_dir = scope_out_dir / row_label.replace(" ", "_")\
                .replace("/", "-").replace("(", "").replace(")", "")
            row_dir.mkdir(parents=True, exist_ok=True)

            base_s = 1.0
            row_preds = {}

            for emb_s in emb_scales:
                if schedule:
                    embeds_schedule = []
                    for step_idx in range(int(args.num_steps)):
                        a_t = alpha_at_step(schedule, step_idx)
                        alpha_per_edit = [
                            (a_t * args.embedding_scale_factor) if i in suppress_idx else None
                            for i in range(n)
                        ]
                        embeds_schedule.append(apply_csv(
                            base_prompt_embeds, edit_positions, steering_vecs,
                            alpha_per_edit, scope=scope,
                            all_prompt_positions=all_prompt_positions, mode=mode))
                    steered_embeds = None
                    step_tag = sched_tag
                else:
                    alpha_per_edit = [
                        (emb_s * args.embedding_scale_factor) if i in suppress_idx else None
                        for i in range(n)
                    ]
                    steered_embeds = apply_csv(
                        base_prompt_embeds, edit_positions, steering_vecs,
                        alpha_per_edit, scope=scope,
                        all_prompt_positions=all_prompt_positions, mode=mode)
                    embeds_schedule = None
                    step_tag = f"emb={emb_s:.2g}"

                for st_s in st_scales:
                    run_idx += 1
                    print(f"[{run_idx}/{total_runs}] {scope_tag_str}  "
                          f"{row_label}  suppress={suppress}  "
                          f"base={base_s}  st={st_s}  {step_tag}")
                    # Use the seed Qwen picked for analora.png so this cell
                    # is directly comparable to the analora reference.
                    cell_gen = torch.Generator(device=device).manual_seed(sweep_seed)
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

            row_grid = build_row_grid(
                image_a, image_a_prime, image_b,
                f"{row_label}  [{scope_tag_str}]", suppress,
                row_preds, st_scales, emb_scales,
                cell_size=args.cell_size,
                base_pred=analora_pred,
            )
            row_grid.save(row_dir / "grid.png")
            print(f"  Saved row grid → {row_dir / 'grid.png'}")

        if len(emb_scales) == 1:
            legacy_preds = {(rl, s): all_preds[(rl, s, emb_scales[0])]
                            for rl, _, _ in row_descs for s in st_scales}
            combined = build_combined_grid(
                image_a, image_a_prime, image_b,
                [(rl, sup) for rl, sup, _ in row_descs],
                legacy_preds, st_scales,
                cell_size=args.cell_size,
                base_pred=analora_pred,
            )
        else:
            combined = build_joint_grid(
                image_a, image_a_prime, image_b,
                row_descs, all_preds, st_scales, emb_scales,
                cell_size=args.cell_size,
                base_pred=analora_pred,
            )
        combined.save(scope_output)
        combined_paths.append(scope_output)
        print(f"[{scope_tag_str}] Saved combined grid → {scope_output}")

    print("Individual images saved under:")
    for p in combined_paths:
        print(f"  {p.parent}/")


if __name__ == "__main__":
    main()
