<div align="center">

# SIA: Selective Image Analogy

### SIGGRAPH Asia 2026

[Amir Alimohammadi](https://alimohammadiamirhossein.github.io/)<sup>1*</sup>, [Kian Izadpanah](https://kianizadpanah.github.io)<sup>1,2*</sup>, [Bardiya Kariminia](https://bardiya2254kariminia.github.io/)<sup>3</sup>, [Yiorgos Chrysanthou](https://www.cs.ucy.ac.cy/~yiorgos/)<sup>2,4</sup>, [Ali Mahdavi-Amiri](https://arash-mham.github.io/)<sup>1</sup>

<sup>1</sup>Simon Fraser University &nbsp; <sup>2</sup>University of Cyprus &nbsp; <sup>3</sup>Shahid Beheshti University &nbsp; <sup>4</sup>CYENS Centre of Excellence<br><sup>*</sup>Equal contribution

<!-- TODO: link the arXiv badge once the paper is on arXiv -->
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://selective-image-analogy.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)](#) [![Video](https://img.shields.io/badge/Video-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=HU010YnVEwI) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Multi--Edit%20Analogy-yellow)](https://huggingface.co/datasets/Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy)

<img src="https://selective-image-analogy.github.io/static/figures/teaser.svg" alt="SIA teaser: one demonstration pair with three edits; SIA transfers any chosen subset of them to a new image" width="100%">

</div>

Image analogy copies the change shown by a before/after pair (**A**, **A'**) onto a new image **B**. A single pair often contains several edits at once, and existing methods transfer all of them. **SIA** keeps the pair fixed and lets you choose which of its edits to apply to B: a token-level mask over the edit phrases in the prompt weakens the unwanted edits in the text embedding and gates a small adapter to them during generation. No new demonstration pair is needed for each combination of edits.

Built on **FLUX.2-Klein**. Supports demonstrations with **2, 3 or 4 simultaneous edits**.

### Paper terms in this code

| Paper | Code |
| --- | --- |
| **AnaLoRA** (learns the full analogy) | Stage 1 — `training/train_analora_flux2_klein.py` |
| **ExLoRA** (masked edit suppression) | Stage 2 — `training/train_exlora_flux2_klein.py`, `ExLoRALinear` |
| **Concept Suppression Vector (CSV)** | Difference-of-means direction — `training/steering_dom.py` |

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (≥24 GB VRAM recommended)
- A HuggingFace account with access to [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)

### Installation

```bash
conda env create -f environment.yml
conda activate visual_analogy
pip install -e .
```

Or install and download the training data into `./data` in one step:

```bash
bash setup.sh
```

### Download the base model

The scripts load models from the local HuggingFace cache and run in offline mode by default, so download FLUX.2-Klein once before training or inference:

```bash
export HF_TOKEN="your_huggingface_token"
huggingface-cli download black-forest-labs/FLUX.2-klein-base-9B
```

Optional: set `WANDB_API_KEY` to log training to Weights & Biases, or `WANDB_MODE=disabled` to turn logging off.

All commands below are run from the repository root.

---

## Dataset

The **Multi-Edit Analogy Dataset** from the paper is on the Hugging Face Hub: [`Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy`](https://huggingface.co/datasets/Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy).

| Subset | Split | Rows | Contents |
| --- | --- | ---: | --- |
| `subjects` | train | 1,605 | source, full edit and partial edits for every subject of 238 edit recipes |
| `analogy_benchmark` | test | 239 | the paper's selective-analogy test cases (A, A', B) |
| `full_transfer` | test | 75 | extra test cases with a ground-truth full-edit B' |

```python
from datasets import load_dataset

train = load_dataset("Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy", "subjects", split="train")
bench = load_dataset("Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy", "analogy_benchmark", split="test")
```

Each row records the number of edits, the subject type (creature or non-creature) and the type of every edit, so subsets are one `filter` away. See the [dataset card](https://huggingface.co/datasets/Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy) for all columns.

The training scripts read the folder layout below, which `setup.sh` downloads.

## Data Structure

Training data lives in `data/` (set by `data_root` in the configs):

```
data/
└── <edit_category>/              # e.g. 2edits, 3edits_non_creature, 4edits
    └── <setting>/                # setting1, setting2, …
        └── <data_item>/          # data1, data2, …
            ├── input.png         # source image (A or B)
            ├── combined1.png     # edit 1 applied
            ├── combined12.png    # edits 1 + 2 applied
            ├── combined123.png   # edits 1 + 2 + 3 applied
            └── prompt.json       # {"edit1": "...", "edit2": "...", ...}
```

Two items from the same setting form an analogy pair: one gives A / A', the other B / B', using the same set of applied edits. The edit prompt is built from `prompt.json`.

---

## Project Structure

```
visual_analogy/              # Core library (pip-installable)
├── models/
│   └── selective_lora.py       # AnaLoRALinear & ExLoRALinear
└── utils/
    ├── hf_utils.py             # HF cache resolution
    └── selective_lora.py       # Injection, masking, save/load utilities

training/                    # Training & inference scripts
├── train_analora_flux2_klein.py       # AnaLoRA (Stage 1)
├── train_exlora_flux2_klein.py        # ExLoRA (Stage 2)
├── infer_single.py                    # Inference with all suppression modes
├── infer_multi.py                     # Inference with a multi-seed preview + Qwen-VL seed judge
├── steering_dom.py                    # CSV (difference-of-means) generation
├── ablation.py                        # CSV × ExLoRA 2×2 ablation runner
├── _check_token_alignment.py          # CPU check for token positions
└── configs/
    ├── train_analora_flux2_klein.yaml
    └── train_exlora_flux2_klein.yaml

evaluation/                  # Evaluation metrics
├── evaluators.py            # CLIP, SigLIP, BLIP, LPIPS, DINOv2, FaceID
├── utils.py
└── example_evaluation.ipynb
```

---

## Training

All training is configured via YAML files in `training/configs/`. Checkpoints are written to `experiments/` by default.

### Stage 1 — AnaLoRA (Global Analogy)

Trains a standard LoRA on **FLUX.2-Klein** using concat-mode analogy (A, A', B as condition images → denoise B'). The loss is computed on the full B' target.

```bash
accelerate launch --mixed_precision bf16 \
    training/train_analora_flux2_klein.py \
    --config training/configs/train_analora_flux2_klein.yaml
```

### Stage 2 — ExLoRA (token-gated LoRA)

Builds on the Stage 1 checkpoint. Injects `ExLoRALinear` modules into the text-stream (context) MM-DiT layers and trains with Partial Prompt Suppression (PPS) loss — enabling per-edit control at inference.

Point `analora_path` in `training/configs/train_exlora_flux2_klein.yaml` at your Stage 1 checkpoint (default: `experiments/analora_flux2_klein/checkpoint-15000`), then run:

```bash
accelerate launch --mixed_precision bf16 \
    training/train_exlora_flux2_klein.py \
    --config training/configs/train_exlora_flux2_klein.yaml
```

---

## Inference

```bash
python training/infer_single.py \
    --config training/configs/train_exlora_flux2_klein.yaml \
    --image_a /path/to/A.png \
    --image_a_prime /path/to/A_prime.png \
    --image_b /path/to/B.png \
    --edits "change pose to running" "change style to oil painting" \
    --output results/result.png
```

The analogy prompt is built from `--edits`. By default the latest checkpoint in the config's `output_dir` is used; pass `--checkpoint_dir` to pick a specific one.

This generates a comparison strip showing: A | A' | B | analora | exlora (no suppression) | exlora (suppress e1) | exlora (suppress e2) | exlora (suppress all).

`training/infer_multi.py` takes the same arguments plus `--seeds 7 30 42 73`: it previews each seed cheaply, lets a Qwen-VL judge (`--qwen_judge_model`, default `Qwen/Qwen3-VL-8B-Instruct`) pick the best one, and uses that seed for all suppression modes. Pass `--no_qwen_judge` to skip the judge and use the first seed.

---

## Concept Suppression Vector (CSV)

`steering_dom.py` computes the CSV, a difference-of-means direction in the text-embedding space:

1. Uses the Qwen3 text encoder to generate N contrastive pos/neg sentence pairs per edit
2. Encodes each via the same pipeline path as diffusion inference
3. Computes `v = normalize(mean(pos) − mean(neg))` — a unit-norm edit direction
4. Caches both the pairs (JSONL) and the vector (`.pt`) for reuse

At inference the CSV is applied to the prompt embeddings at the suppressed edit's token positions (`--csv_mode ablate` removes the component along the direction, as in the paper), complementing ExLoRA masking.

---

## CSV × ExLoRA Ablation

`training/ablation.py` runs a 2×2 grid varying the **Concept Suppression Vector (CSV)** and **ExLoRA** independently — same noise seed in every cell, so any visual difference is attributable to the intervention.

```bash
python training/ablation.py \
    --config        training/configs/train_exlora_flux2_klein.yaml \
    --image_a       /path/to/A.png \
    --image_a_prime /path/to/A_prime.png \
    --image_b       /path/to/B.png \
    --edits "change pose to running" "change style to oil painting" \
    --suppress 1 \
    --csv_alpha 5.0 \
    --output_dir   results/ablation_run
```

Outputs in `--output_dir`:

- `ablation_grid.png` — A | A' | B reference strip + 4-cell grid (rows: ExLoRA off/on, cols: CSV off/on)
- `csv{0,1}_exlora{0,1}.png` — annotated per-cell PNGs
- `summary.txt` — prompt, suppressed indices, decoded edit-token positions, seed
- `metrics.csv` — CLIP / LPIPS / DINOv2 scores per cell (skipped if evaluators unavailable)

---

## Token-Position Check

ExLoRA's token mask and the CSV are both applied at the token positions of each edit in the chat-templated prompt. A CPU-only check verifies that these positions decode back to the edit text and that `build_token_mask` fails loudly on missing edits:

```bash
python training/_check_token_alignment.py
```

---

## Config Options

| Key                             | Description                                                    |
| ------------------------------- | -------------------------------------------------------------- |
| `pretrained_model_name_or_path` | Base model (default: `black-forest-labs/FLUX.2-klein-base-9B`) |
| `data_root`                     | Path to the dataset (default: `data`)                          |
| `output_dir`                    | Where checkpoints and logs are saved                           |
| `analora_path`                  | Stage 1 checkpoint path (ExLoRA only)                          |
| `lora_rank`                     | LoRA rank (default: `16`)                                      |
| `max_train_steps`               | Total training steps                                           |
| `learning_rate`                 | Learning rate (default: `5e-5`)                                |
| `analora_learning_rate`         | LR for the AnaLoRA during ExLoRA training                      |
| `analora_anchor_weight`         | L2 anchor weight to prevent AnaLoRA drift                      |
| `train_batch_size`              | Batch size per GPU                                             |
| `checkpointing_steps`           | Save a checkpoint every N steps                                |
| `validation_steps`              | Run validation every N steps                                   |
| `resolution`                    | Training image resolution (default: `512`)                     |

---

## Evaluation

See `evaluation/example_evaluation.ipynb` for scoring generated images:

- **CLIP / SigLIP / BLIP** — text-image alignment scores
- **LPIPS** — perceptual distance
- **DINOv2** — structural similarity
- **FaceID** — identity preservation

---

## Citation

If you find SIA useful in your research, please cite:

```bibtex
@inproceedings{alimohammadi2026sia,
  title     = {SIA: Selective Image Analogy},
  author    = {Alimohammadi, Amir and Izadpanah, Kian and
               Kariminia, Bardiya and Chrysanthou, Yiorgos and
               Mahdavi-Amiri, Ali},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  publisher = {Association for Computing Machinery}
}
```

Questions and requests are welcome — please open an issue.
