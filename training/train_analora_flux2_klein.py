# ── Offline mode by default: models are read from the local HF cache ────────
# These env vars MUST be set BEFORE any HuggingFace / transformers / diffusers
# imports, because huggingface_hub reads them at import time.
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# WandB is disabled when WANDB_MODE=disabled or WANDB_DISABLED=true.
# Some GPU nodes have no wandb-core installed, so we must skip
# all wandb calls entirely — not just put the run in offline mode.
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

from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors_file
from peft import LoraConfig
from accelerate.utils import set_seed
from accelerate.utils import DataLoaderConfiguration
from omegaconf import OmegaConf
import random
from tqdm.auto import tqdm
import math
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.models import Flux2Transformer2DModel
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import json
import argparse

# Importing the ExLoRA trainer also applies its accelerate compatibility shim.
from training.train_exlora_flux2_klein import (
    _get_combined_files,
    compute_empirical_mu,
    compute_text_embeddings,
    encode_single_image,
    get_sigmas,
    load_models,
    resize_to_match,
    retrieve_timesteps,
)


def harmonize_images(a, a_prime, b, b_prime):
    """Resize all 4 images to the same size (use A's dimensions as reference).
    Ensures the 2x2 grid has consistent cell sizes."""
    target_w, target_h = a.size
    a_prime = resize_to_match(a_prime, target_w, target_h)
    b = resize_to_match(b,       target_w, target_h)
    b_prime = resize_to_match(b_prime, target_w, target_h)
    return a, a_prime, b, b_prime


def _build_prompt_from_indices(edit_indices: list, prompt_dict: dict) -> str:
    """Build the analogy prompt from selected 1-based edit indices.

    e.g. edit_indices=[1,2], prompt_dict={"edit1":"add hat","edit2":"add glasses"}
    →  "Image 1 is the original … Edit 1: add hat. Edit 2: add glasses. Apply …"
    """
    edits_part = " ".join(
        f"Edit {i}: {prompt_dict[f'edit{i}']}."
        for i in sorted(edit_indices)
        if prompt_dict.get(f'edit{i}', '')
    )
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )


def _tier_probs(n_tiers: int) -> list:
    """Per-tier sampling probabilities, longest tier first.

    AnaLoRA is the "full-transfer" model (paper §4.1, §4.3 eq. 11), so we
    strongly bias toward the largest available subset (= full edit set
    ``I^E``). The tail is kept non-zero so AnaLoRA also sees partial
    subsets occasionally — useful when partial inputs exist and as light
    augmentation against overfitting to a single edit configuration.

    1 → [1.0]
    2 → [0.92, 0.08]
    3 → [0.85, 0.10, 0.05]
    4+ → [0.85, 0.10, 0.05/(n-2), ...]
    """
    if n_tiers <= 1:
        return [1.0] * max(n_tiers, 1)
    if n_tiers == 2:
        return [0.92, 0.08]
    if n_tiers == 3:
        return [0.85, 0.10, 0.05]
    tail = 0.05 / (n_tiers - 2)
    return [0.85, 0.10] + [tail] * (n_tiers - 2)


def _sample_combined(combined_files: list):
    """Sample one (filename, edit_indices) using tier probabilities.

    Tier = number of edits applied (len of edit_indices).
    Longest tier → highest probability.  Uniform within each tier.
    """
    if not combined_files:
        raise ValueError("combined_files is empty")
    tier_dict: dict = {}
    for fname, indices in combined_files:
        t = len(indices)
        tier_dict.setdefault(t, []).append((fname, indices))
    sorted_tiers = sorted(tier_dict.keys(), reverse=True)
    probs = _tier_probs(len(sorted_tiers))
    chosen_tier = random.choices(sorted_tiers, weights=probs)[0]
    return random.choice(tier_dict[chosen_tier])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageAnalogyDataset(Dataset):
    """Image analogy dataset reading from the SIA_datasets/train structure.

    Directory layout::

        <data_root>/
            <edit_category>/        # 3_edits | 4edits | 2edits_non_creature | 3edits_non_creature
                <setting>/          # setting1 | setting2 | …
                    <data_item>/    # data1 | data2 | …
                        input.png           → A  (before)
                        combined1.png       → A' option (edit 1 only)
                        combined12.png      → A' option (edits 1+2)
                        combined123.png     → A' option (edits 1+2+3)  ← longest = 70 %
                        prompt.json         → {"edit1": "…", "edit2": "…", …}

    Sampling procedure for one training pair (A/A', B/B'):

    1. Pick an edit category uniformly at random.
    2. Pick a setting within that category uniformly at random.
    3. Pick two distinct data items from that setting.
    4. Intersect the available combined edit-index sets across both items.
    5. Sample one edit-index set using tier probabilities
       (longest = 70 %, second = 20 %, remaining share 10 %).
    6. Load A/A' from item 1 and B/B' from item 2 using that set.
    7. Build the prompt from the selected edit indices.
    """

    def __init__(self, data_root, mode="concat"):
        self.data_root = Path(data_root)
        self.mode = mode  # kept for interface compatibility; concat logic always used

        # categories[cat][setting] = list of (data_dir, combined_files, prompt_dict)
        self.categories: dict = {}
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
                samples = []

                for data_dir in sorted(setting_dir.iterdir()):
                    if not data_dir.is_dir():
                        continue
                    if not (data_dir / 'input.png').exists():
                        n_skipped += 1
                        continue
                    if not (data_dir / 'prompt.json').exists():
                        n_skipped += 1
                        continue
                    combined = _get_combined_files(data_dir)
                    if not combined:
                        n_skipped += 1
                        continue
                    try:
                        prompt_dict = json.loads((data_dir / 'prompt.json').read_text())
                    except Exception:
                        n_skipped += 1
                        continue
                    samples.append((data_dir, combined, prompt_dict))

                if len(samples) >= 2:
                    self.categories[cat][setting] = samples

        self.cat_names = sorted(self.categories.keys())
        total = sum(len(s) for c in self.categories.values() for s in c.values())
        print(f"SIADataset: {len(self.cat_names)} categories, {total} samples "
              f"({n_skipped} skipped)")
        for cat in self.cat_names:
            n = sum(len(s) for s in self.categories[cat].values())
            print(f"  {cat}: {len(self.categories[cat])} settings, {n} samples")

    def __len__(self):
        return sum(len(s) for c in self.categories.values() for s in c.values())

    def sample_val_pair(self):
        """Return a dict with raw paths and prompt for use in validation.

        Uses the same matched-pair sampling as ``__getitem__`` so the
        validation A' and B' always refer to the same combined-edit subset.
        """
        (cat, setting, data_dir_a, data_dir_b,
         fname_a, fname_b, indices, prompt_a) = self._sample_matched_pair()
        prompt = _build_prompt_from_indices(indices, prompt_a)
        return {
            'a_path':       str(data_dir_a / 'input.png'),
            'a_prime_path': str(data_dir_a / fname_a),
            'b_path':       str(data_dir_b / 'input.png'),
            'b_prime_path': str(data_dir_b / fname_b),
            'edit_name':    f"{cat}/{setting}",
            'prompt':       prompt,
        }

    def _sample_matched_pair(self, max_tries: int = 32):
        """Sample (cat, setting, item_a, item_b, fname_a, fname_b, indices)
        such that A' and B' refer to the *same* combined-edit subset.

        The dataset's items have heterogeneous combinedN.png files (e.g. one
        4-edit item may have ``combined3.png`` while another only has
        ``combined12.png``). To guarantee A→A' and B→B' demonstrate the
        same transformation, we only sample pairs whose available combined
        subsets intersect; we retry a few times before giving up and
        falling back to the longest combined image on both sides.
        """
        for _ in range(max_tries):
            cat = random.choice(self.cat_names)
            settings = self.categories[cat]
            setting = random.choice(list(settings.keys()))
            samples = settings[setting]

            sample_a, sample_b = random.sample(samples, 2)
            data_dir_a, combined_a, prompt_a = sample_a
            data_dir_b, combined_b, prompt_b = sample_b

            map_a = {frozenset(idx): fn for fn, idx in combined_a}
            map_b = {frozenset(idx): fn for fn, idx in combined_b}
            common = set(map_a) & set(map_b)
            if not common:
                continue

            pool = [(map_a[k], sorted(k)) for k in common]
            fname_a, indices = _sample_combined(pool)
            fname_b = map_b[frozenset(indices)]
            return (cat, setting, data_dir_a, data_dir_b,
                    fname_a, fname_b, indices, prompt_a)

        # Final fallback: pick *any* setting; default to the longest combined
        # available on each side independently (visually they may differ but
        # this keeps the dataset usable when intersection is repeatedly
        # empty). In a healthy dataset this branch should be unreachable.
        cat = random.choice(self.cat_names)
        settings = self.categories[cat]
        setting = random.choice(list(settings.keys()))
        sample_a, sample_b = random.sample(settings[setting], 2)
        data_dir_a, combined_a, prompt_a = sample_a
        data_dir_b, combined_b, _ = sample_b
        fname_a, indices_a = max(combined_a, key=lambda x: len(x[1]))
        fname_b, _ = max(combined_b, key=lambda x: len(x[1]))
        return (cat, setting, data_dir_a, data_dir_b,
                fname_a, fname_b, indices_a, prompt_a)

    def __getitem__(self, idx):
        (cat, setting, data_dir_a, data_dir_b,
         fname_a, fname_b, indices, prompt_a) = self._sample_matched_pair()

        a       = Image.open(data_dir_a / 'input.png').convert('RGB')
        a_prime = Image.open(data_dir_a / fname_a).convert('RGB')
        b       = Image.open(data_dir_b / 'input.png').convert('RGB')
        b_prime = Image.open(data_dir_b / fname_b).convert('RGB')
        a, a_prime, b, b_prime = harmonize_images(a, a_prime, b, b_prime)

        # Build the listed-edits prompt; 15 % of the time drop it entirely so
        # AnaLoRA also learns to infer the A→A' transformation purely from
        # the visual demonstration (no edit-text supervision).
        combined_prompt = _build_prompt_from_indices(indices, prompt_a)
        if random.random() < 0.15:
            combined_prompt = ""

        return {
            'A':         a,
            'A_prime':   a_prime,
            'B':         b,
            'B_prime':   b_prime,
            'edit_name': f"{cat}/{setting}",
            'prompt':    combined_prompt,
        }


def create_grid_condition_ids(packed_h, packed_w):
    """Create condition IDs for a 2x2 grid with per-quadrant T values.

    T=10 (top-left, A), T=20 (top-right, A'), T=30 (bottom-left, B),
    T=0 (bottom-right, B' generation target). H,W are global grid coordinates.
    Returns (1, packed_h*packed_w, 4).
    """
    half_h, half_w = packed_h // 2, packed_w // 2
    rows = torch.arange(packed_h).float()
    cols = torch.arange(packed_w).float()
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')
    t_map = torch.zeros(packed_h, packed_w)
    t_map[:half_h, :half_w] = 10   # A
    t_map[:half_h, half_w:] = 20   # A'
    t_map[half_h:, :half_w] = 30   # B
    t_map[half_h:, half_w:] = 0    # B' (generation target)
    l_map = torch.zeros(packed_h, packed_w)
    ids = torch.stack([t_map, grid_rows, grid_cols, l_map], dim=-1)
    return ids.reshape(1, packed_h * packed_w, 4)


def create_b_prime_mask(packed_h=64, packed_w=64):
    """Boolean mask for B' (bottom-right quadrant) in packed latent token space.

    For 1024x1024 grid → VAE 128x128 → packed 64x64.
    Each quadrant is 32x32 packed tokens. B' = bottom-right 32x32.
    """
    mask = torch.zeros(packed_h, packed_w, dtype=torch.bool)
    mask[packed_h // 2:, packed_w // 2:] = True
    return mask.reshape(-1)


def collate_fn(batch):
    return {
        'A': [item['A'] for item in batch],
        'A_prime': [item['A_prime'] for item in batch],
        'B': [item['B'] for item in batch],
        'B_prime': [item['B_prime'] for item in batch],
        'prompts': [item['prompt'] for item in batch],
    }


@torch.no_grad()
def prepare_analogy_latents(pipe, batch, mode="concat"):
    """Encode A, A', B, B' through VAE and prepare latents + IDs.

    Args:
        mode: "concat" (default) - separate images with T=10/20/30 IDs
              "grid" - 2x2 spatial grid with B' mask

    Returns:
        target_latents, condition_latents, target_ids, condition_ids, b_prime_mask
        b_prime_mask is None in concat mode, a boolean tensor in grid mode.
    """
    target_arr, cond_arr = [], []

    for a, a_prime, b, b_prime in zip(batch['A'], batch['A_prime'], batch['B'], batch['B_prime']):
        lat_a = encode_single_image(pipe, a)
        lat_a_prime = encode_single_image(pipe, a_prime)
        lat_b = encode_single_image(pipe, b)
        lat_b_prime = encode_single_image(pipe, b_prime)

        if mode == "grid":
            # Assemble single 2x2 grid: A, A', B are clean conditions; B' is target
            # No separate condition grid — T-coordinates distinguish roles
            top_row = torch.cat([lat_a, lat_a_prime], dim=3)
            bot_target = torch.cat([lat_b, lat_b_prime], dim=3)
            target_grid = torch.cat(
                [top_row, bot_target], dim=2)  # (1, C, 2h, 2w)

            target_arr.append(Flux2KleinPipeline._pack_latents(target_grid))
        else:
            # Concat mode: B' is target, A/A'/B are separate conditions
            target_arr.append(Flux2KleinPipeline._pack_latents(lat_b_prime))
            cond_a = Flux2KleinPipeline._pack_latents(lat_a).squeeze(0)
            cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime).squeeze(0)
            cond_b = Flux2KleinPipeline._pack_latents(lat_b).squeeze(0)
            cond_arr.append(
                torch.cat([cond_a, cond_ap, cond_b], dim=0).unsqueeze(0))

    if mode == "grid":
        packed_h, packed_w = target_grid.shape[2], target_grid.shape[3]
        # Single grid IDs: B' at T=0 (target), A/A'/B at T=10/20/30 (conditions)
        target_ids = create_grid_condition_ids(packed_h, packed_w)
        b_prime_mask = create_b_prime_mask(packed_h, packed_w)
        return (
            torch.cat(target_arr, dim=0),
            None,
            target_ids,
            None,
            b_prime_mask,
        )

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_b_prime)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b])

    return (
        torch.cat(target_arr, dim=0),
        torch.cat(cond_arr, dim=0),
        target_ids,
        condition_ids,
        None,
    )


def add_lora(transformer, args, lora_target_modules):
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
    )
    transformer.add_adapter(transformer_lora_config,
                            adapter_name=args.lora_name)


def get_save_model_hook(accelerator, args):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                print("Type of model in save_model_hook:", type(model))
                if isinstance(accelerator.unwrap_model(model), Flux2Transformer2DModel):
                    Flux2KleinPipeline.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=get_peft_model_state_dict(
                            accelerator.unwrap_model(model), adapter_name=args.lora_name),
                    )
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                weights.pop()  # make sure to pop weight so that corresponding model is not saved again

    return save_model_hook


def get_load_model_hook(accelerator, args):
    def load_model_hook(models, input_dir):
        lora_weights_path = os.path.join(input_dir, "pytorch_lora_weights.safetensors")
        while len(models) > 0:
            model = models.pop()
            if isinstance(accelerator.unwrap_model(model), Flux2Transformer2DModel):
                if not os.path.exists(lora_weights_path):
                    raise FileNotFoundError(
                        f"LoRA checkpoint not found at {lora_weights_path}. "
                        "Expected 'pytorch_lora_weights.safetensors' saved by the save hook."
                    )
                # The file uses diffusers key format: "transformer.<peft_key>"
                # Strip the leading "transformer." to recover peft-compatible keys.
                raw_state_dict = load_safetensors_file(lora_weights_path)
                peft_state_dict = {
                    k[len("transformer."):]: v
                    for k, v in raw_state_dict.items()
                    if k.startswith("transformer.")
                }
                unwrapped = accelerator.unwrap_model(model)
                incompatible = set_peft_model_state_dict(
                    unwrapped, peft_state_dict, adapter_name=args.lora_name
                )
                # set_peft_model_state_dict loads with strict=False, so every
                # frozen base-model weight (not present in the LoRA checkpoint)
                # is reported as "missing". That is expected and harmless. Only
                # genuinely missing LoRA tensors or unexpected keys are problems.
                missing_lora_keys = [
                    k for k in getattr(incompatible, "missing_keys", [])
                    if "lora_" in k
                ]
                unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
                if missing_lora_keys or unexpected_keys:
                    print(
                        "[load_model_hook] Problem loading LoRA weights — "
                        f"missing LoRA keys: {missing_lora_keys}, "
                        f"unexpected keys: {unexpected_keys}"
                    )
                else:
                    n_loaded = sum(1 for k in peft_state_dict if "lora_" in k)
                    print(
                        f"[load_model_hook] Restored {n_loaded} LoRA tensors from "
                        f"{lora_weights_path}"
                    )
            else:
                raise ValueError(f"Unexpected model type in load_model_hook: {type(model)=}.")

    return load_model_hook


def unpack_latents(latents, packed_h, packed_w):
    """Inverse of Flux2 _pack_latents: (B, H*W, C) → (B, C, H, W) in patchified space."""
    batch_size, num_patches, channels = latents.shape
    latents = latents.permute(0, 2, 1).reshape(
        batch_size, channels, packed_h, packed_w)
    return latents


@torch.no_grad()
def log_analogy_validation(pipeline, train_dataset, tokenizer, text_encoder,
                           args, accelerator, global_step):
    """Run analogy inference with LoRA scale=1 and log a comparison strip to wandb."""
    print(f"Running analogy validation at step {global_step}...")

    pipe = pipeline
    device = accelerator.device
    dtype = torch.bfloat16
    mode = getattr(args, 'analogy_mode', 'concat')

    # Pick a fresh random sample from the multi-edit dataset.
    val_meta = train_dataset.sample_val_pair()
    combined_prompt = val_meta['prompt']
    edit_name       = val_meta['edit_name']

    if mode == "grid":
        # Grid mode is an unused ablation here; build a minimal grid prompt
        # by reusing the concat-mode edits list — the visual layout below
        # is what makes the difference, not the wording.
        combined_prompt = (
            "This is a 2x2 grid image. "
            "Top-left is A, top-right is A', bottom-left is B, bottom-right is B'. "
            f"{val_meta['prompt']} "
            "Treat the grid as two edit pairs: A to A' and B to B'."
        )

    val_sample = {
        'A':         Image.open(val_meta['a_path']).convert('RGB'),
        'A_prime':   Image.open(val_meta['a_prime_path']).convert('RGB'),
        'B':         Image.open(val_meta['b_path']).convert('RGB'),
        'B_prime':   Image.open(val_meta['b_prime_path']).convert('RGB'),
        'edit_name': edit_name,
        'prompt':    combined_prompt,
    }

    # Encode images through VAE
    lat_a = encode_single_image(pipe, val_sample['A'])
    lat_a_prime = encode_single_image(pipe, val_sample['A_prime'])
    lat_b = encode_single_image(pipe, val_sample['B'])

    # Text embeddings
    prompt_embeds, text_ids = compute_text_embeddings(
        [val_sample['prompt']], tokenizer, text_encoder
    )
    # Klein: no guidance embeds, pass guidance=None
    guidance = None

    def decode_latents(packed, h, w):
        """Decode packed latents (1, N, C) → PIL image, given patchified spatial dims h, w."""
        unpacked = packed.permute(0, 2, 1).reshape(1, -1, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(
            1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(
            1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    pipe.transformer.set_adapters([args.lora_name], [1])
    num_inference_steps = args.num_val_inference_steps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    if mode == "grid":
        # Single grid: A, A', B are clean conditions; B' starts as noise
        top_row = torch.cat([lat_a, lat_a_prime], dim=3)
        bot_row = torch.cat([lat_b, lat_b.clone()], dim=3)
        grid = torch.cat([top_row, bot_row], dim=2)
        packed_h, packed_w = grid.shape[2], grid.shape[3]

        gen_latents = Flux2KleinPipeline._pack_latents(grid)
        bp_mask = create_b_prime_mask(packed_h, packed_w).to(device)
        clean_non_bp = gen_latents[:, ~bp_mask].clone()
        gen_latents[:, bp_mask] = torch.randn_like(gen_latents[:, bp_mask])

        combined_ids = create_grid_condition_ids(packed_h, packed_w).to(device)

        image_seq_len = gen_latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

        for t in timesteps:
            noise_pred = pipe.transformer(
                hidden_states=gen_latents,
                timestep=t.expand(1).to(dtype) / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0]
            gen_latents = pipe.scheduler.step(
                noise_pred, t, gen_latents, return_dict=False)[0]
            gen_latents[:, ~bp_mask] = clean_non_bp  # keep non-B' clean

        # Decode full grid, crop B' (bottom-right quadrant)
        decoded_grid = decode_latents(gen_latents, packed_h, packed_w)
        grid_w, grid_h = decoded_grid.size
        pred_b_prime = decoded_grid.crop(
            (grid_w // 2, grid_h // 2, grid_w, grid_h))
    else:
        # Concat mode: denoise single B' image from noise
        patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

        cond_a = Flux2KleinPipeline._pack_latents(lat_a)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
        cond_b = Flux2KleinPipeline._pack_latents(lat_b)
        condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
        gen_latents = Flux2KleinPipeline._pack_latents(gen_latents)

        target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
        condition_ids = Flux2KleinPipeline._prepare_image_ids(
            [lat_a, lat_a_prime, lat_b]).to(device)
        combined_ids = torch.cat([target_ids, condition_ids], dim=1)

        image_seq_len = gen_latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

        for t in timesteps:
            latent_model_input = torch.cat(
                [gen_latents, condition_latents], dim=1)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(1).to(dtype) / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0][:, :gen_latents.shape[1]]
            gen_latents = pipe.scheduler.step(
                noise_pred, t, gen_latents, return_dict=False)[0]

        pred_b_prime = decode_latents(gen_latents, patch_h, patch_w)

    # Condition images from val_sample (shown once, not repeated)
    cell_size = 512
    a_img = val_sample['A'].resize((cell_size, cell_size), Image.LANCZOS)
    a_prime_img = val_sample['A_prime'].resize(
        (cell_size, cell_size), Image.LANCZOS)
    b_img = val_sample['B'].resize((cell_size, cell_size), Image.LANCZOS)
    b_prime_gt_img = val_sample['B_prime'].resize(
        (cell_size, cell_size), Image.LANCZOS)

    # Log to wandb: individual images + full strip
    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = [a_img, a_prime_img, b_img, b_prime_gt_img, pred_b_prime]
        labels = ["A", "A'", "B", "B' GT", "B' pred"]

        cell_w, cell_h = cell_size, cell_size
        strip_w = cell_w * len(all_imgs)
        strip_h = cell_h + 30
        strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        for i, (img, label) in enumerate(zip(all_imgs, labels)):
            strip.paste(img.resize((cell_w, cell_h)), (i * cell_w, 30))
            draw.text((i * cell_w + 4, 4), label, fill=(0, 0, 0))

        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip to {strip_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(print_args=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )

    args = OmegaConf.load(parser.parse_args().config)

    if print_args:
        print("="*40, "Arguments", "="*40)
        for arg in args:
            print(f"{arg}: {getattr(args, arg)}")
        print("="*91)

    return args


def main():
    args = parse_args(print_args=True)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=True)],
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
        wandb.define_metric("val/*", step_metric="train/step")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    set_seed(args.seed)

    tokenizer, text_encoder, vae, transformer, noise_scheduler = load_models(
        args)

    # Only train the LoRA layers
    for m in [text_encoder, vae, transformer]:
        m.requires_grad_(False)

    weight_dtype = torch.bfloat16
    for m in [text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=weight_dtype)

    transformer.enable_gradient_checkpointing()

    LORA_TARGET_MODULES = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    args.lora_name = "visual-analogy-analora"
    add_lora(transformer, args, LORA_TARGET_MODULES)

    accelerator.register_save_state_pre_hook(
        get_save_model_hook(accelerator, args))
    accelerator.register_load_state_pre_hook(
        get_load_model_hook(accelerator, args))

    # Make sure the trainable params are in float32. [only upcast trainable parameters (LoRA) into fp32]
    cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters()))
    optimizer = torch.optim.AdamW(  # TODO: Change to prodigy
        [{"params": transformer_lora_parameters, "lr": args.learning_rate}],
        betas=(0.9, 0.999),
        weight_decay=1e-04,
        eps=1e-08,
    )

    analogy_mode = getattr(args, 'analogy_mode', 'concat')
    args.analogy_mode = analogy_mode
    train_dataset = ImageAnalogyDataset(args.data_root, mode=analogy_mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=2, shuffle=True)

    lr_scheduler = get_scheduler(
        "constant",  # TODO: make it argument
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    len_train_dataset = len(train_dataset)
    len_train_dataloader = math.ceil(len_train_dataset / args.train_batch_size)
    num_update_steps_per_epoch = math.ceil(
        len_train_dataloader / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    print("***** Running training *****")
    print(f"  Num examples = {len_train_dataset}")
    print(f"  Num batches each epoch = {len_train_dataloader}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    initial_global_step = 0
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        # Get latest checkpoint
        # A fresh run has no output_dir yet; treat that as "nothing to resume".
        dirs = os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"No checkpoint found in {args.output_dir}. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    pipeline = Flux2KleinPipeline(
        # Own copy: validation calls retrieve_timesteps() on pipe.scheduler, which would
        # otherwise shrink noise_scheduler.timesteps to the 28 inference steps and
        # collapse training timestep sampling onto the lowest-noise one.
        scheduler=copy.deepcopy(noise_scheduler),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(
            transformer, keep_fp32_wrapper=False),
    )

    transformer.train()

    for epoch in range(first_epoch, num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                prompts = batch["prompts"]

                target_latents, condition_latents, target_ids, condition_ids, b_prime_mask = prepare_analogy_latents(
                    pipeline, batch, mode=args.analogy_mode
                )

                device = target_latents.device
                target_ids = target_ids.to(device)

                bsz = target_latents.shape[0]

                # Expand IDs to batch size
                target_ids = target_ids.expand(bsz, -1, -1)

                if condition_ids is not None:
                    # Concat mode: target (T=0) + conditions (T=10/20/30)
                    condition_ids = condition_ids.to(
                        device).expand(bsz, -1, -1)
                    combined_ids = torch.cat(
                        [target_ids, condition_ids], dim=1)
                else:
                    # Grid mode: single grid with per-quadrant T values
                    combined_ids = target_ids

                prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, tokenizer, text_encoder)

                noise = torch.randn_like(target_latents)

                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none", batch_size=bsz)
                indices = (u * (args.max_train_timesteps -
                           args.min_train_timesteps) + args.min_train_timesteps).long()
                indices = indices.clamp(0, len(noise_scheduler.timesteps) - 1)
                timesteps = noise_scheduler.timesteps[indices].to(
                    device=target_latents.device)

                # Add noise via flow matching (grid: only B' quadrant; concat: all tokens)
                sigmas = get_sigmas(noise_scheduler, timesteps, device=target_latents.device,
                                    n_dim=target_latents.ndim, dtype=target_latents.dtype)
                if b_prime_mask is not None:
                    b_prime_mask_dev = b_prime_mask.to(device)
                    noisy_model_input = target_latents.clone()
                    noisy_model_input[:, b_prime_mask_dev] = (
                        1.0 - sigmas) * target_latents[:, b_prime_mask_dev] + sigmas * noise[:, b_prime_mask_dev]
                else:
                    noisy_model_input = (1.0 - sigmas) * \
                        target_latents + sigmas * noise

                # Klein: no guidance embeds, always guidance=None
                guidance = None

                if condition_latents is not None:
                    # Concat mode: append condition tokens
                    latent_model_input = torch.cat(
                        [noisy_model_input, condition_latents], dim=1)
                else:
                    # Grid mode: single grid, conditions embedded spatially
                    latent_model_input = noisy_model_input

                transformer.set_adapters([args.lora_name], [1])
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_ids,
                    return_dict=False,
                )[0]
                if condition_latents is not None:
                    model_pred = model_pred[:, :noisy_model_input.shape[1]]

                # Flow matching loss (grid: only B' quadrant; concat: all tokens)
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigmas)
                flow_target = noise - target_latents
                if b_prime_mask is not None:
                    pred_masked = model_pred[:, b_prime_mask_dev]
                    target_masked = flow_target[:, b_prime_mask_dev]
                else:
                    pred_masked = model_pred
                    target_masked = flow_target
                loss = torch.mean(
                    (weighting.float() * (pred_masked.float() -
                     target_masked.float()) ** 2).reshape(bsz, -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

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
                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        pipeline.transformer = accelerator.unwrap_model(
                            transformer, keep_fp32_wrapper=False)
                        log_analogy_validation(
                            pipeline, train_dataset,
                            tokenizer, text_encoder,
                            args, accelerator, global_step,
                        )
                        transformer.train()

            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process and _USE_WANDB:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
