# MIDI-LLM Fine-tuning · QLoRA on Llama 3.2 1B for Text-to-MIDI Music Generation

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/Pyramidk/midi-llm-demo)
[![Model](https://img.shields.io/badge/🤗-Merged%20Model-yellow)](https://huggingface.co/Pyramidk/midi-llm-merged)
[![Base](https://img.shields.io/badge/Base-slseanwu%2FMIDI--LLM-lightgrey)](https://github.com/slSeanWU/MIDI-LLM)

> Course project for **DDA4220** (CUHK-Shenzhen). Fine-tunes [slSeanWU/MIDI-LLM](https://github.com/slSeanWU/MIDI-LLM)
> (Llama 3.2 1B) with **QLoRA**, extends the vocab with **velocity / chord / CFG tokens**, and ships a
> deployable Gradio webapp with live progress and in-browser MIDI playback.

---

## 🎵 Try it now

Open the demo and paste any music description — the model returns a downloadable, in-browser-playable `.mid`:

**[https://huggingface.co/spaces/Pyramidk/midi-llm-demo](https://huggingface.co/spaces/Pyramidk/midi-llm-demo)**

Running on free HF Spaces CPU — generation takes 30–180 s depending on token count.

---

## ✨ What's new vs. the base model

| Improvement | Detail |
|---|---|
| **77 added special tokens** | 16 velocity buckets (`<vel_0>`…`<vel_15>`), 60 chord tokens (`<C>`, `<Cmaj7>`, `<Dm>`…), 1 CFG token (`[UNCOND]`) |
| **QLoRA fine-tuning** | 4-bit quantized base + LoRA (r=16, α=32, dropout=0.1) — **only 0.38 % trainable params** (4.7 M / 1.24 B) |
| **Classifier-Free Guidance** | 15 % of training samples replace text with `[UNCOND]` so generation can be CFG-steered at inference |
| **Memory-efficient** | Trains on a **single 8 GB RTX 4060** with `paged_adamw_8bit`; checkpoint is **~80 MB** vs. 3.3 GB full model |
| **Deployable webapp** | FastAPI for local use + Gradio for HF Spaces, with live token-streaming progress and `html-midi-player` browser playback |

---

## 📊 Evaluation results

20 prompts × 2048 tokens each, fine-tuned vs. original base. See
[results/evaluation_results_20251219_231707/EVALUATION_REPORT.md](results/evaluation_results_20251219_231707/EVALUATION_REPORT.md)
for the full report.

| Metric | Original | Fine-tuned | Δ |
|---|---|---|---|
| Pitch range | 47.45 | 51.50 | **+8.54 %** |
| Pitch std (variation) | 11.78 | 12.59 | **+6.90 %** |
| Note density | 25.77 | 22.46 | −12.86 % |
| Avg pitch | 56.76 | 54.76 | −3.52 % |
| IOI mean | 0.090 | 0.076 | −15.25 % |
| Generation success | 100 % | 100 % | — |

Training loss converged from **11.95 → 1.91** over 375 steps (3 epochs, 1000-sample midicaps subset).

---

## 🛠️ Setup

A GPU with **8 GB+ VRAM** and CUDA 12.x is recommended for training; inference works on CPU but is slow.

```bash
# 1. Conda environment
conda create -n midi-llm python=3.11 -y
conda activate midi-llm

# 2. Audio toolchain (optional, only for MIDI → audio synthesis)
conda install -c conda-forge ffmpeg fluidsynth

# 3. PyTorch (example for CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Project deps
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

```bash
cd scripts
python prepare_training_data.py     # extract velocity + chord tokens from midicaps
python train_safe.py                # QLoRA fine-tune
```

Outputs go to `outputs/safe_train_YYYYMMDD_HHMMSS/final_model/` (~80 MB LoRA adapter).

### Inference (interactive)

```bash
python scripts/generate_interactive.py
```

Prompts you in the terminal; saves `.mid` files to `generated_outputs/`.

### Evaluation

```bash
python scripts/quick_evaluation.py        # generate 20 samples original vs fine-tuned
python scripts/compute_midi_metrics.py    # pitch / density / IOI / etc.
```

### Webapp

Two variants under `webapp/`:

- **`webapp/app.py`** — FastAPI + plain HTML, runs locally with GPU; better UX, latency-sensitive
- **`webapp/hf_deploy/app.py`** — Gradio for HF Spaces deployment with live progress + in-browser playback

To deploy your own Space, see [`webapp/hf_deploy/DEPLOY.md`](webapp/hf_deploy/DEPLOY.md).

---

## 📁 Project structure

```
MIDI-LLM/
├── scripts/                              # core pipeline
│   ├── prepare_training_data.py          # extract velocity/chord tokens from MIDI
│   ├── train_safe.py                     # QLoRA fine-tuning
│   ├── generate_interactive.py           # CLI text → MIDI inference
│   ├── quick_evaluation.py               # 20-sample generation eval
│   └── compute_midi_metrics.py           # pitch/density/IOI analysis
├── midi_llm/                             # utility library
├── webapp/
│   ├── app.py                            # FastAPI demo (local)
│   ├── merge_model.py                    # merge LoRA into base for fast loading
│   └── hf_deploy/                        # Gradio app + HF Spaces deploy files
├── results/
│   └── evaluation_results_20251219_231707/
│       ├── EVALUATION_REPORT.md          # detailed report (loss curve, metrics)
│       ├── evaluation_report.json
│       └── midi_metrics.json
├── assets/
│   ├── evaluation_set_lakh_ids.txt       # 896 Lakh IDs used in the paper's eval
│   └── example_prompts.txt
└── requirements.txt
```

Large artifacts excluded from this repo (must be obtained separately):
- `outputs/` — training checkpoints
- `lmd_full/` — Lakh MIDI Dataset
- `data/train.json` — full training metadata (409 MB)
- `webapp/merged_model/` — merged inference model (3.3 GB; available on
  [HF Hub](https://huggingface.co/Pyramidk/midi-llm-merged) instead)

---

## 🙏 Attribution

This project is built on top of [**slSeanWU/MIDI-LLM**](https://github.com/slSeanWU/MIDI-LLM) by Shih-Lun Wu,
Yoon Kim, and Cheng-Zhi Anna Huang (NeurIPS AI4Music Workshop 2025). The base
[Llama-3.2-1B model](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B) and the Anticipation tokenizer are
their work — this repo adds QLoRA fine-tuning, vocab extensions, evaluation tooling, and a Gradio
deployment.

```bibtex
@inproceedings{wu2025midillm,
  title={{MIDI-LLM}: Adapting large language models for text-to-{MIDI} music generation},
  author={Wu, Shih-Lun and Kim, Yoon and Huang, Cheng-Zhi Anna},
  booktitle={Proc. NeurIPS AI4Music Workshop},
  year={2025}
}
```

License: see [LICENSE.md](LICENSE.md).
