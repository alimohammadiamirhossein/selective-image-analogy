<div align="center">

# SIA: Selective Image Analogy

### SIGGRAPH Asia 2026

[Amir Alimohammadi](https://alimohammadiamirhossein.github.io/)<sup>1*</sup>, [Kian Izadpanah](https://kianizadpanah.github.io)<sup>1,2*</sup>, [Bardiya Kariminia](https://bardiya2254kariminia.github.io/)<sup>3</sup>, [Yiorgos Chrysanthou](https://www.cs.ucy.ac.cy/~yiorgos/)<sup>2,4</sup>, [Ali Mahdavi-Amiri](https://arash-mham.github.io/)<sup>1</sup>

<sup>1</sup>Simon Fraser University &nbsp; <sup>2</sup>University of Cyprus &nbsp; <sup>3</sup>Shahid Beheshti University &nbsp; <sup>4</sup>CYENS Centre of Excellence<br><sup>*</sup>Equal contribution

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://selective-image-analogy.github.io/) [![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Try%20it-blue)](https://amirhossein-alimohammadi-selective-image-analogy.hf.space/) [![arXiv](https://img.shields.io/badge/arXiv-2611.33894-b31b1b)](https://arxiv.org/abs/2611.33894) [![Video](https://img.shields.io/badge/Video-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=HU010YnVEwI) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Multi--Edit%20Analogy-yellow)](https://huggingface.co/datasets/selective-image-analogy/SIA-Multi-Edit-Analogy) [![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-SIA--weights-yellow)](https://huggingface.co/selective-image-analogy/SIA-weights)

<img src="https://selective-image-analogy.github.io/static/figures/teaser.svg" alt="SIA teaser: one demonstration pair with three edits; SIA transfers any chosen subset of them to a new image" width="100%">

</div>

Image analogy copies the change shown by a before/after pair (**A**, **A'**) onto a new image **B**. A single pair often contains several edits at once, and existing methods transfer all of them. **SIA** keeps the pair fixed and lets you choose which of its edits to apply to B: a token-level mask over the edit phrases in the prompt weakens the unwanted edits in the text embedding and gates a small adapter to them during generation. No new demonstration pair is needed for each combination of edits.

This repository contains the code for two backbones, [FLUX.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (`flux2_klein`) and [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (`qwen_image21`), for demonstrations with 2, 3 or 4 simultaneous edits.

## Paper terms in this code

| Paper | Code |
| --- | --- |
| **AnaLoRA** (stage 1, learns the full analogy) | `scripts/train_analora.py`, `AnaLoRALinear` in `sia/lora.py` |
| **ExLoRA** (stage 2, token-gated edit suppression) | `scripts/train_exlora.py`, `ExLoRALinear` in `sia/lora.py`, injection and masking in `sia/adapters.py` |
| **Concept Suppression Vector (CSV)** | `sia/concept_suppression.py` (+ `steering_pool` / `cache_key` in `sia/backbones/`) |

## Repository layout

```
sia/                    Python package (pip install -e .)
  lora.py               AnaLoRALinear, ExLoRALinear
  adapters.py           adapter injection, token masks, checkpoint save/load
  hf.py                 local-or-Hub checkpoint resolution
  data.py, prompt.py    dataset readers and the analogy prompt
  concept_suppression.py  CSV directions and their cache
  inference.py          backbone loading and selective generation
  training.py           accelerate / W&B setup, checkpoint hooks, resume
  viz.py                result strips
  backbones/            flux2_klein.py, qwen_image21.py (same function names in both)
  eval/                 benchmark evaluation
scripts/                train_analora.py, train_exlora.py, infer.py, eval.py, build_csv.py
configs/                {analora,exlora}_{flux2_klein,qwen_image21}.yaml
tests/                  CPU unit tests
demo/                   Gradio app for the Hugging Face Space
```

## Installation

The two backbones need different `diffusers` / `transformers` versions, so use one virtual environment per backbone (tested with Python 3.12 and torch 2.14):

```bash
# FLUX.2-Klein
python -m venv .venv-klein && . .venv-klein/bin/activate
pip install -r requirements.txt && pip install -e .

# Qwen-Image-2.1
python -m venv .venv-qwen && . .venv-qwen/bin/activate
pip install -r requirements-qwen.txt && pip install -e .
```

Base models, the dataset and the SIA weights are downloaded to the Hugging Face cache on first use. FLUX.2-klein-base-9B is gated: accept its license on the Hub and log in (`huggingface-cli login` or `HF_TOKEN`). Once everything is cached, `export HF_HUB_OFFLINE=1` runs from the cache only.

Run all commands from the repository root. Training logs to Weights & Biases when `WANDB_API_KEY` is set; `WANDB_MODE=disabled` turns logging off.

## Dataset

The **Multi-Edit Analogy Dataset** from the paper is on the Hub: [`selective-image-analogy/SIA-Multi-Edit-Analogy`](https://huggingface.co/datasets/selective-image-analogy/SIA-Multi-Edit-Analogy).

| Config | Split | Rows | Contents |
| --- | --- | ---: | --- |
| `subjects` | train | 1,605 | source, full edit and partial edits for every subject of 238 edit recipes |
| `analogy_benchmark` | test | 239 | the paper's selective-analogy test cases (A, A', B) |
| `full_transfer` | test | 75 | extra test cases with a ground-truth full-edit B' |

```python
from datasets import load_dataset

train = load_dataset("selective-image-analogy/SIA-Multi-Edit-Analogy", "subjects", split="train")
bench = load_dataset("selective-image-analogy/SIA-Multi-Edit-Analogy", "analogy_benchmark", split="test")
```

Each row records the number of edits, the subject type (creature or non-creature) and the type of every edit, so subsets are one `filter` away. See the [dataset card](https://huggingface.co/datasets/selective-image-analogy/SIA-Multi-Edit-Analogy) for all columns. The code reads the dataset with `datasets.load_dataset` (config keys `dataset` and `dataset_revision`); there is no separate download step. During training, two subjects of the same edit recipe form an analogy pair: one gives A / A', the other B / B', with the same set of edits.

## Pretrained weights

[`selective-image-analogy/SIA-weights`](https://huggingface.co/selective-image-analogy/SIA-weights) (public) holds one folder per backbone, `flux2_klein/` and `qwen_image21/`, each with `analora/` (stage 1) and `exlora/` (stage 2, self-contained: it also carries the AnaLoRA fine-tuned in stage 2). Every weights option (`--ckpt`, `--analora`, `--checkpoint_dir`, config key `analora_path`) accepts either a local checkpoint directory or a Hub path `<org>/<repo>/<subfolder>[@revision]`, for example `selective-image-analogy/SIA-weights/flux2_klein/exlora`; only that subfolder is downloaded (`sia.hf.resolve_checkpoint`).

The example configs list the planned step counts of the paper runs (`max_train_steps`); the released weights are intermediate checkpoints of those runs: Qwen-Image-2.1 AnaLoRA step 2000 and ExLoRA step 5000, FLUX.2-Klein AnaLoRA step 2000 and ExLoRA step 250 (see the [weights card](https://huggingface.co/selective-image-analogy/SIA-weights)).

The released FLUX.2-Klein checkpoints were trained at the dataset's native 384x512 resolution, which is the default for every FLUX.2-Klein image (training and inference). The environment variable `SIA_NATIVE_RES` overrides it: `0` selects the legacy square recipe (`resolution` x `resolution`), `WxH` another fixed size (`_native_wh` in `sia/backbones/flux2_klein.py`). Qwen-Image-2.1 uses the pipeline's area-based sizing with `resolution: 512`.

## Inference

`scripts/infer.py` takes your own A, A', B images and the list of edits that turn A into A'. It renders AnaLoRA alone and then ExLoRA suppressing every non-empty subset of the edits, all from the same seeded noise, so the images differ only by which edits are suppressed.

```bash
# Qwen-Image-2.1 (run in the Qwen environment)
python scripts/infer.py \
    --config configs/exlora_qwen_image21.yaml \
    --checkpoint_dir selective-image-analogy/SIA-weights/qwen_image21/exlora \
    --image_a A.png --image_a_prime A_prime.png --image_b B.png \
    --edits "add sweater" "add santa hat" "add street background" \
    --output results/result.png

# FLUX.2-Klein (run in the FLUX.2-Klein environment)
python scripts/infer.py \
    --config configs/exlora_flux2_klein.yaml \
    --checkpoint_dir selective-image-analogy/SIA-weights/flux2_klein/exlora \
    --image_a A.png --image_a_prime A_prime.png --image_b B.png \
    --edits "add sweater" "add santa hat" \
    --output results/result.png
```

- `--edits` are the edit texts in order; the analogy prompt is built from them in the training format.
- `--checkpoint_dir` is a Hub path or a local stage 2 checkpoint directory (for example one written by `scripts/train_exlora.py`). Without it, the latest checkpoint in the config's `output_dir` is used.
- Next to `--output` the script writes `analora.png`, one folder per suppression subset (annotated cells and `grid.png`), and the combined grid at `--output`. `--scale_values` sets the ExLoRA scales shown as columns (default `0 0.25 0.5 0.75 1`).
- FLUX.2-Klein resizes every input image to 384x512 (without keeping the aspect ratio), so 3:4 portrait inputs give the best results; the output is resized back to B's size.
- Defaults follow each backbone's recipe: FLUX.2-Klein 28 steps without CFG; Qwen-Image-2.1 40 steps, no CFG, prefix KV cache on (`--no_kv_cache` turns it off), `--resolution` from the config. `--num_steps` and `--seed` (default 0) override them.
- Qwen-Image-2.1 inference peaked at about 34 GB of GPU memory in our tests.

For use from Python, `sia.inference` provides `load_pipeline`, `run_selective_analogy` and `load_qwen_pipeline_with_exlora` (used by the demo).

## Training

Training runs in two stages, each configured by one YAML file per backbone. Checkpoints go to `output_dir` (default `experiments/...`); `resume_from_checkpoint: true` resumes from the latest one.

```bash
# Stage 1: AnaLoRA, learns the full analogy (A, A', B) -> B'
accelerate launch --mixed_precision bf16 scripts/train_analora.py --config configs/analora_flux2_klein.yaml

# Stage 2: ExLoRA, starts from the stage 1 checkpoint given by analora_path
accelerate launch --mixed_precision bf16 scripts/train_exlora.py --config configs/exlora_flux2_klein.yaml
```

For Qwen-Image-2.1 use `configs/{analora,exlora}_qwen_image21.yaml`. Set `analora_path` in the ExLoRA config to your stage 1 checkpoint (local directory or Hub path such as `selective-image-analogy/SIA-weights/flux2_klein/analora`). During stage 2 the AnaLoRA keeps training under an L2 anchor to its stage 1 weights (`analora_learning_rate`, `analora_anchor_weight`).

Hardware: the configs use the batch sizes of our runs (16 for stage 1, 18 for stage 2), which were trained on a single 97 GB GPU. The Qwen-Image-2.1 configs turn gradient checkpointing off (`gradient_checkpointing: false`; about 68 GB peak in our runs). With `train_batch_size: 1` and gradient checkpointing on, a training step peaked at about 34 GB (Qwen-Image-2.1) and 35 GB (FLUX.2-Klein), so smaller GPUs need a smaller batch size, or `gradient_accumulation_steps`, together with gradient checkpointing.

Main config keys:

| Key | Meaning |
| --- | --- |
| `backbone` | `flux2_klein` or `qwen_image21` (inferred from `pretrained_model_name_or_path` if absent) |
| `pretrained_model_name_or_path` | base model id |
| `dataset`, `dataset_revision` | Hub dataset id and optional revision |
| `output_dir` | checkpoints and logs |
| `analora_path` | stage 1 checkpoint (ExLoRA only) |
| `lora_rank` | LoRA rank (16) |
| `learning_rate`, `analora_learning_rate`, `analora_anchor_weight` | optimiser settings (the last two for ExLoRA only) |
| `train_batch_size`, `max_train_steps` | batch size per optimizer step, total steps |
| `checkpointing_steps`, `validation_steps` | checkpoint and validation intervals |
| `resolution` | Qwen-Image-2.1 target area is `resolution`² at each image's aspect ratio |
| `gradient_checkpointing` | trade speed for memory |

## Concept Suppression Vector (CSV)

The CSV is a difference-of-means direction in the text-embedding space: the backbone's text encoder writes contrastive sentence pairs for each edit, both sides are encoded through the same path as at inference, and `normalize(mean(pos) - mean(neg))` is the edit direction. It is applied to the prompt embeddings at the suppressed edits' tokens, on top of ExLoRA. Directions are cached per edit in `~/.cache/sia/csv/<backbone>` (`--csv_cache`, alias `--steering_cache_dir`, in all three scripts); a cached direction is reused, and pair sampling is not seeded. The number of pairs per new direction (`--csv_pairs`, alias `--num_steering_pairs`; `--pairs` in `build_csv.py`) defaults to 64 in `eval.py` and `build_csv.py` but 32 in `infer.py`; the cache key does not include it, so a cached direction is reused whatever count it was built with (a warning is printed).

- `infer.py`: `--csv_mode ablate` removes the component along the direction (as in the paper; `--embedding_scales` adds a push after it), `--csv_mode add` with non-zero `--embedding_scales` pushes along it.
- `eval.py generate`: `--csv 1 --csv_mode ablate --csv_lambda 1` (projection) or `--csv 1 --csv_mode add --csv_k K`.

`scripts/build_csv.py` precomputes the directions of every distinct edit of an evaluation split:

```bash
python scripts/build_csv.py --config configs/exlora_qwen_image21.yaml --split analogy_benchmark
```

CSV is optional. In our experiments it helped most with AnaLoRA alone and with early ExLoRA checkpoints, and changed results little on top of the fully trained ExLoRA.

## Evaluation

`scripts/eval.py` runs the benchmark evaluation of the paper in four resumable steps. For each `analogy_benchmark` case (239) and every non-empty proper subset of its edits it generates B' keeping that subset; a VLM judge (Qwen3-VL-32B-Instruct) scores Retained Accuracy, Preservation, Suppression and Edit Recall, and CLIP directional scores are computed too. The `full_transfer` split (75 cases with ground truth) is scored with LPIPS, DINOv2 distance and CLIP-I. `--split dev_train` uses training subjects instead, for choosing settings without touching the test set.

```bash
# 1. generate (outputs go to experiments/eval_<backbone>/<name>/)
python scripts/eval.py generate --config configs/exlora_qwen_image21.yaml --name sia \
    --ckpt selective-image-analogy/SIA-weights/qwen_image21/exlora
python scripts/eval.py generate --config configs/exlora_qwen_image21.yaml --name sia_csv \
    --ckpt selective-image-analogy/SIA-weights/qwen_image21/exlora --csv 1 --csv_mode ablate

# 2. score, 3. report, 4. paired comparison of two runs
python scripts/eval.py score   --names sia sia_csv --stages clip judge
python scripts/eval.py report  --names sia sia_csv
python scripts/eval.py compare --method sia_csv --baseline sia
```

- `--ckpt` takes a stage 2 checkpoint (local directory or Hub path), `--analora` a stage 1 checkpoint (AnaLoRA only, suppressed edits deleted from the prompt). `--arm NAME:METHOD[:VALUE]` (repeatable) writes several methods from one model load; `--limit N` evaluates the first N cases.
- For FLUX.2-Klein, pass `--config configs/exlora_flux2_klein.yaml` to `generate` and `--out experiments/eval_flux2_klein` to `score`, `report` and `compare` (their default output root is the Qwen-Image-2.1 one; they stop with an error when no `<out>/<name>/` folder exists). `--dataset` and `--dataset_revision` select the dataset in every subcommand and in `build_csv.py`.
- The judge needs about 70 GB of GPU memory (it does not fit on a 48 GB card); `--stages clip` runs on any GPU.

## Tests

```bash
pip install -e .[test]
pytest tests
```

The tests run on CPU. Tests that need a backbone's tokenizer or the dataset are skipped when these are not in the Hugging Face cache; run the suite in each environment to cover both backbones.

## Demo

Try SIA in the browser on the [Hugging Face Space](https://amirhossein-alimohammadi-selective-image-analogy.hf.space/). `demo/` is its Gradio app (Qwen-Image-2.1, `qwen_image21/exlora` weights, no CSV).

## License

- Code: [MIT](LICENSE).
- Dataset: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- Weights: each backbone folder follows its base model's license. `flux2_klein/`: [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md); its section 2(e) requires content filtering or manual review of outputs and disclosure that outputs are AI-generated when you generate with the `flux2_klein` weights. `qwen_image21/`: Qwen RESEARCH LICENSE AGREEMENT (non-commercial research and evaluation only). Built with Qwen.

## Citation

```bibtex
@inproceedings{alimohammadi2026sia,
  title     = {SIA: Selective Image Analogy},
  author    = {Alimohammadi, Amir and Izadpanah, Kian and
               Kariminia, Bardiya and Chrysanthou, Yiorgos and
               Mahdavi-Amiri, Ali},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3829340.3842205}
}
```

Questions and requests are welcome: please open an issue.
