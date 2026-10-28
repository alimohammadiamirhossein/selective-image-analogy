#!/usr/bin/env python3
"""
ablation.py

CSV (Concept Suppression Vector) x ExLoRA 2x2 ablation for Selective Image Analogy.

Given (A, A', B) and a list of edits with one or more `--suppress` indices,
runs four inferences with identical noise seed and identical inputs:

    +-------------------+----------------+----------------+
    |                   |  CSV  off      |  CSV  on       |
    +-------------------+----------------+----------------+
    | ExLoRA off        |  AnaLoRA only|  base + CSV    |
    | ExLoRA on         |  base + ExLoRA |  full method   |
    +-------------------+----------------+----------------+

Outputs:
  - <output_dir>/ablation_grid.png        : 2x2 grid w/ A | A' | B reference strip
  - <output_dir>/csv{0,1}_exlora{0,1}.png : per-cell predictions
  - <output_dir>/summary.txt              : prompt, suppressed edits, edit-token
                                             positions (decoded), seed, vector source
  - <output_dir>/metrics.csv              : optional CLIP/LPIPS/DINO scores
                                             (skipped if evaluators unavailable)

Usage:
  python training/ablation.py \\
      --config        training/configs/train_exlora_flux2_klein.yaml \\
      --image_a       /path/to/A.png \\
      --image_a_prime /path/to/A_prime.png \\
      --image_b       /path/to/B.png \\
      --edits "edit one" "edit two" \\
      --suppress 1 \\
      --csv_alpha 5.0 \\
      --output_dir   /tmp/ablation_run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from training.train_exlora_flux2_klein import (
    build_analogy_prompt,
    compute_text_embeddings,
)
from training.infer_single import (
    _fit_thumbnail,
    _cell_dims,
    _load_font,
    apply_csv,
    load_pipeline,
    run_inference,
    templated_substring_indices,
)
from training.steering_dom import get_or_build_steering_vector
from visual_analogy.utils.selective_lora import klein_templated_prompt_token_positions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="CSV x ExLoRA 2x2 ablation runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, required=True,
                   help="YAML config used by the ExLoRA trainer / infer_single.")
    p.add_argument("--image_a", type=str, required=True)
    p.add_argument("--image_a_prime", type=str, required=True)
    p.add_argument("--image_b", type=str, required=True)
    p.add_argument("--edits", type=str, nargs="+", required=True,
                   help="Edit strings in prompt order (e.g. 'change pose to running').")
    p.add_argument("--suppress", type=int, nargs="+", default=[1],
                   help="Edit indices (0-based) to suppress in the 'ExLoRA on' "
                        "and 'CSV on' cells. Default: [1] (suppress 2nd edit).")
    p.add_argument("--csv_alpha", type=float, default=-5.0,
                   help="CSV steering magnitude (alpha * embedding_scale_factor). "
                        "Negative suppresses the edit; positive amplifies. "
                        "For suppression (the usual case) pass a negative value, e.g. -5.0.")
    p.add_argument("--csv_scope", type=str, default="tokens",
                   choices=["tokens", "prompt"],
                   help="Where to apply the CSV. 'tokens' = suppressed-edit "
                        "positions only (matches ExLoRA's footprint).")
    p.add_argument("--csv_mode", type=str, default="add",
                   choices=["add", "ablate"],
                   help="'add' = additive push; 'ablate' = directional ablation "
                        "+ optional push.")
    p.add_argument("--embedding_scale_factor", type=float, default=1.0)
    p.add_argument("--num_steering_pairs", type=int, default=32)
    p.add_argument("--steering_cache_dir", type=str, default="./steering_cache")
    p.add_argument("--regen_steering", action="store_true")

    p.add_argument("--exlora_scale", type=float, default=1.0,
                   help="ExLoRA scale used in the 'ExLoRA on' cells.")
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cell_size", type=int, default=512)
    p.add_argument("--checkpoint_dir", type=str, default=None)

    p.add_argument("--output_dir", type=str, default="ablation_out")
    p.add_argument("--no_metrics", action="store_true",
                   help="Skip CLIP/LPIPS/DINO scoring even if the evaluators import OK.")

    cli = p.parse_args()
    cfg = OmegaConf.load(cli.config)
    cfg.image_a = cli.image_a
    cfg.image_a_prime = cli.image_a_prime
    cfg.image_b = cli.image_b
    cfg.edits = list(cli.edits)
    cfg.suppress = list(cli.suppress)
    cfg.csv_alpha = float(cli.csv_alpha)
    cfg.csv_scope = cli.csv_scope
    cfg.csv_mode = cli.csv_mode
    cfg.embedding_scale_factor = float(cli.embedding_scale_factor)
    cfg.num_steering_pairs = int(cli.num_steering_pairs)
    cfg.steering_cache_dir = cli.steering_cache_dir
    cfg.regen_steering = bool(cli.regen_steering)
    cfg.exlora_scale = float(cli.exlora_scale)
    cfg.num_steps = int(cli.num_steps)
    cfg.seed = int(cli.seed)
    cfg.cell_size = int(cli.cell_size)
    cfg.checkpoint_dir = cli.checkpoint_dir
    cfg.output_dir_ablation = cli.output_dir
    cfg.no_metrics = bool(cli.no_metrics)
    cfg.prompt = build_analogy_prompt(cli.edits, cli.edits)
    return cfg


# ---------------------------------------------------------------------------
# Steering setup
# ---------------------------------------------------------------------------

def _resolve_steering(args, tokenizer, text_encoder, base_prompt_embeds, device):
    """Build per-edit steering vectors for the suppressed indices.

    Returns
    -------
    edit_positions : list[list[int]]
        Templated-prompt token positions of each edit (length == len(edits)).
        Empty list at index i means we could not align edit i — that index is
        excluded from CSV application.
    steering_vecs : list[Optional[Tensor]]
        Unit-norm DoM vectors per edit (None for non-suppressed or non-aligned).
    suppress_set : set[int]
        Validated set of suppressed edit indices.
    """
    n = len(args.edits)
    suppress_set = set(int(i) for i in args.suppress)
    if not suppress_set or any(i < 0 or i >= n for i in suppress_set):
        raise ValueError(
            f"--suppress {sorted(suppress_set)} is out of range for "
            f"{n} edit(s). Indices are 0-based."
        )

    max_seq_len = base_prompt_embeds.shape[1]
    edit_positions = []
    for i, edit_text in enumerate(args.edits):
        pos = templated_substring_indices(
            args.prompt, edit_text, tokenizer, max_length=max_seq_len)
        edit_positions.append(pos)
        if i in suppress_set and not pos:
            raise RuntimeError(
                f"[ablation] Suppressed edit {i} ({edit_text!r}) does not align "
                f"in templated prompt. Check that the edit string is a literal "
                f"substring of the prompt produced by build_analogy_prompt."
            )

    steering_vecs: list = [None] * n
    cache_root = Path(args.steering_cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    hidden_dim = base_prompt_embeds.shape[-1]
    for i in sorted(suppress_set):
        v, meta = get_or_build_steering_vector(
            edit=args.edits[i],
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
                f"Steering vector dim {v.shape[-1]} != prompt_embeds dim "
                f"{hidden_dim} for edit {i}. Cache from a different encoder "
                f"config? Pass --regen_steering."
            )
        steering_vecs[i] = v.to(device=device, dtype=base_prompt_embeds.dtype)
        print(f"[ablation] CSV ready for edit_{i+1} ({args.edits[i][:60]}): "
              f"source={meta.get('source','?')} "
              f"|v|_pre={meta.get('pre_normalization_norm','?')}")

    return edit_positions, steering_vecs, suppress_set


def _build_steered_embeds(base_prompt_embeds, edit_positions, steering_vecs,
                          suppress_set, args, all_prompt_positions):
    n = len(args.edits)
    alpha_per_edit = [
        (args.csv_alpha * args.embedding_scale_factor) if i in suppress_set else None
        for i in range(n)
    ]
    return apply_csv(
        base_prompt_embeds, edit_positions, steering_vecs,
        alpha_per_edit,
        scope=args.csv_scope,
        all_prompt_positions=all_prompt_positions,
        mode=args.csv_mode,
    )


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def _run_cell(*, pipe, tokenizer, text_encoder, image_a, image_a_prime, image_b,
              prompt, suppressed_edit_texts, num_steps, device, seed,
              analora_scale, exlora_scale, prompt_embeds_override):
    """One ablation cell — fresh-seeded noise generator for fair comparison."""
    cell_gen = torch.Generator(device=device).manual_seed(int(seed))
    return run_inference(
        pipe, tokenizer, text_encoder,
        image_a, image_a_prime, image_b,
        prompt, suppressed_edit_texts,
        num_steps, device,
        analora_scale=analora_scale,
        exlora_scale=exlora_scale,
        prompt_embeds_override=prompt_embeds_override,
        prompt_embeds_schedule=None,
        generator=cell_gen,
    )


# ---------------------------------------------------------------------------
# Grid composition
# ---------------------------------------------------------------------------

def _annotate(cell_img: Image.Image, label: str, font_size: int = 16) -> Image.Image:
    font = _load_font(font_size)
    banner = font_size + 10
    out = Image.new("RGB", (cell_img.width, cell_img.height + banner), (255, 255, 255))
    out.paste(cell_img, (0, 0))
    ImageDraw.Draw(out).text((4, cell_img.height + 4), label, fill=(0, 0, 0), font=font)
    return out


def _build_grid(image_a, image_a_prime, image_b, preds, cell_size: int) -> Image.Image:
    """preds: dict[(csv_on, exlora_on)] -> PIL.Image."""
    cell_w, cell_h = _cell_dims(image_b, cell_size)
    font = _load_font(16)

    header_h = 40
    ref_strip_h = cell_h + 24
    col_hdr_h = 24
    row_hdr_w = 110
    grid_w = row_hdr_w + 2 * cell_w
    total_w = grid_w
    total_h = header_h + ref_strip_h + col_hdr_h + 2 * cell_h

    img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, total_w - 1, header_h - 1], fill=(210, 225, 255))
    draw.text((6, 10), "CSV x ExLoRA ablation  (rows: ExLoRA, cols: CSV)",
              fill=(0, 0, 100), font=font)

    # Reference strip: A | A' | B across the full width, equal thirds.
    ref_y = header_h
    third_w = total_w // 3
    thumb_h = ref_strip_h - 28  # leave room for the column label above
    for ci, (ref_img, lbl) in enumerate(
            [(image_a, "A"), (image_a_prime, "A'"), (image_b, "B")]):
        x = ci * third_w
        thumb = _fit_thumbnail(ref_img, third_w - 6, thumb_h)
        img.paste(thumb, (x + 3, ref_y + 22))
        draw.text((x + 6, ref_y + 2), lbl, fill=(0, 0, 0), font=font)

    col_hdr_y = header_h + ref_strip_h
    draw.rectangle([0, col_hdr_y, row_hdr_w - 1, col_hdr_y + col_hdr_h - 1],
                   fill=(230, 230, 230))
    draw.text((4, col_hdr_y + 4), "ExLoRA \\ CSV", fill=(80, 80, 80), font=font)
    for ci, csv_on in enumerate([0, 1]):
        x = row_hdr_w + ci * cell_w
        draw.rectangle([x, col_hdr_y, x + cell_w - 1, col_hdr_y + col_hdr_h - 1],
                       fill=(220, 240, 255))
        draw.text((x + 6, col_hdr_y + 4),
                  "CSV on" if csv_on else "CSV off", fill=(0, 0, 0), font=font)

    grid_y0 = col_hdr_y + col_hdr_h
    for ri, exlora_on in enumerate([0, 1]):
        y = grid_y0 + ri * cell_h
        draw.rectangle([0, y, row_hdr_w - 1, y + cell_h - 1],
                       fill=(245, 245, 245) if ri == 0 else (235, 235, 250))
        draw.text((4, y + 4),
                  "ExLoRA on" if exlora_on else "ExLoRA off",
                  fill=(0, 0, 100), font=font)
        for ci, csv_on in enumerate([0, 1]):
            x = row_hdr_w + ci * cell_w
            cell = preds.get((csv_on, exlora_on))
            if cell is not None:
                img.paste(_fit_thumbnail(cell, cell_w, cell_h), (x, y))
            tag = f"csv={csv_on} exlora={exlora_on}"
            draw.text((x + 4, y + 4), tag, fill=(255, 255, 200), font=font)
    return img


# ---------------------------------------------------------------------------
# Optional metrics
# ---------------------------------------------------------------------------

def _maybe_score_metrics(preds, image_b, prompt, device):
    """Best-effort CLIP/LPIPS/DINO scoring. Returns list of dict rows or None."""
    try:
        from evaluation.evaluators import (
            CLIPEvaluator, LPIPSFeatureDistanceEvaluator, DiNOFeatureDistanceEvaluator,
        )
    except Exception as exc:
        print(f"[ablation] Metric evaluators unavailable ({exc}); skipping CSV log.")
        return None

    rows = []
    try:
        clip_eval = CLIPEvaluator(device=device)
        lpips_eval = LPIPSFeatureDistanceEvaluator(net="alex", device=device)
        dino_eval = DiNOFeatureDistanceEvaluator(device=device)
    except Exception as exc:
        print(f"[ablation] Could not instantiate evaluators ({exc}); "
              "skipping metric scoring.")
        return None

    for (csv_on, exlora_on), pred in preds.items():
        rows.append({
            "csv": csv_on,
            "exlora": exlora_on,
            "clip_prompt": clip_eval.get_score(pred, prompt),
            "lpips_to_B": lpips_eval.get_distance(pred, image_b),
            "dino_cos_to_B": dino_eval.get_distance(pred, image_b),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    image_a = Image.open(args.image_a).convert("RGB")
    image_a_prime = Image.open(args.image_a_prime).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")

    out_dir = Path(args.output_dir_ablation)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ablation] Loading models …")
    pipe, tokenizer, text_encoder = load_pipeline(args, device)

    with torch.no_grad():
        base_prompt_embeds, _ = compute_text_embeddings(
            [args.prompt], tokenizer, text_encoder)

    edit_positions, steering_vecs, suppress_set = _resolve_steering(
        args, tokenizer, text_encoder, base_prompt_embeds, device)

    # Resolve all_prompt_positions only when needed.
    all_prompt_positions = None
    if args.csv_scope == "prompt":
        all_prompt_positions = klein_templated_prompt_token_positions(
            args.prompt, tokenizer, max_length=base_prompt_embeds.shape[1])

    steered_embeds = _build_steered_embeds(
        base_prompt_embeds, edit_positions, steering_vecs,
        suppress_set, args, all_prompt_positions)

    suppressed_edit_texts = [args.edits[i] for i in sorted(suppress_set)]
    print(f"[ablation] Suppressing edit indices {sorted(suppress_set)}: "
          f"{suppressed_edit_texts}")

    preds = {}

    cell_specs = [
        # (csv_on, exlora_on,  base_scale, exlora_scale,  embeds_override,
        #  texts_to_suppress)
        (0, 0, 1.0, 0.0,                None, []),
        (1, 0, 1.0, 0.0,                steered_embeds, []),
        (0, 1, 1.0, float(args.exlora_scale), None, suppressed_edit_texts),
        (1, 1, 1.0, float(args.exlora_scale), steered_embeds, suppressed_edit_texts),
    ]

    for csv_on, exlora_on, base_s, stl_s, emb, texts in cell_specs:
        tag = f"csv={csv_on} exlora={exlora_on}"
        print(f"[ablation] Running cell {tag} (base={base_s}, st={stl_s})")
        pred = _run_cell(
            pipe=pipe, tokenizer=tokenizer, text_encoder=text_encoder,
            image_a=image_a, image_a_prime=image_a_prime, image_b=image_b,
            prompt=args.prompt, suppressed_edit_texts=texts,
            num_steps=int(args.num_steps), device=device, seed=args.seed,
            analora_scale=base_s, exlora_scale=stl_s,
            prompt_embeds_override=emb,
        )
        preds[(csv_on, exlora_on)] = pred
        cell_path = out_dir / f"csv{csv_on}_exlora{exlora_on}.png"
        _annotate(pred, tag).save(cell_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grid = _build_grid(image_a, image_a_prime, image_b, preds, int(args.cell_size))
    grid_path = out_dir / "ablation_grid.png"
    grid.save(grid_path)
    print(f"[ablation] Saved grid -> {grid_path}")

    # ---- summary.txt ----
    summary_lines = [
        f"prompt: {args.prompt}",
        f"edits: {list(args.edits)}",
        f"suppress (0-based indices): {sorted(suppress_set)}",
        f"suppressed_edit_texts: {suppressed_edit_texts}",
        f"csv_alpha: {args.csv_alpha}",
        f"csv_scope: {args.csv_scope}",
        f"csv_mode: {args.csv_mode}",
        f"embedding_scale_factor: {args.embedding_scale_factor}",
        f"exlora_scale: {args.exlora_scale}",
        f"num_steps: {args.num_steps}",
        f"seed: {args.seed}",
        "",
        "edit-token positions in the templated Qwen prompt (decoded back):",
    ]
    for i, (etext, pos) in enumerate(zip(args.edits, edit_positions)):
        if pos:
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(templated, max_length=base_prompt_embeds.shape[1],
                            padding="max_length", truncation=True,
                            return_tensors="pt")["input_ids"][0].tolist()
            decoded = tokenizer.decode([ids[p] for p in pos],
                                       skip_special_tokens=False)
            summary_lines.append(
                f"  edit_{i+1} (suppressed={i in suppress_set}): "
                f"{len(pos)} tokens, range [{pos[0]}..{pos[-1]}], decoded={decoded!r}"
            )
        else:
            summary_lines.append(f"  edit_{i+1}: NO alignment in templated prompt")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # ---- metrics.csv (best-effort) ----
    if not args.no_metrics:
        rows = _maybe_score_metrics(preds, image_b, args.prompt, device)
        if rows:
            csv_path = out_dir / "metrics.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[ablation] Saved metrics -> {csv_path}")
            for r in rows:
                print(f"  {r}")

    print(f"[ablation] Done. Outputs under {out_dir}/")


if __name__ == "__main__":
    main()
