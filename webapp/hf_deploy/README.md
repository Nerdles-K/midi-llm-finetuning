---
title: MIDI-LLM Demo
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.20.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
---

# MIDI-LLM: Text-to-Music with Fine-tuned Llama 3.2 1B

A QLoRA-fine-tuned **Llama 3.2 1B** for MIDI music generation from text prompts.

## Improvements over base
- **16 velocity tokens** `<vel_0>`–`<vel_15>` for dynamic control
- **60 chord tokens** for harmonic conditioning
- **1 CFG token** `[UNCOND]` for classifier-free guidance during training
- Trained with QLoRA (r=16, α=32) — only 0.38% trainable parameters

## Note
Running on **free CPU tier** — each generation takes ~60–180 seconds.
For real-time generation see the [source repo](https://github.com/) (requires GPU).

## How to use
1. Type a music description (e.g. "a cheerful piano melody")
2. Adjust max tokens (256 ≈ ~10s of music)
3. Click Generate, wait, download the .mid file
