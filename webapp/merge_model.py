"""
One-time script: merge LoRA adapter into base model and save locally.
After running this, app.py loads from merged_model/ — much faster startup.

Usage:
    cd webapp
    python merge_model.py

Time: ~2-4 minutes (CPU merge, then saves ~2.5 GB)
Result: webapp/merged_model/ (tokenizer + merged weights)
"""
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR = Path(__file__).parent.parent
LORA_CHECKPOINT = BASE_DIR / "outputs" / "safe_train_20251219_185027" / "final_model"
BASE_MODEL = "slseanwu/MIDI-LLM_Llama-3.2-1B"
OUTPUT_DIR = Path(__file__).parent / "merged_model"

VELOCITY_TOKENS = [f"<vel_{i}>" for i in range(16)]
CHORD_TOKENS = [
    "<C>", "<Cm>", "<C#>", "<C#m>", "<D>", "<Dm>", "<D#>", "<D#m>",
    "<E>", "<Em>", "<F>", "<Fm>", "<F#>", "<F#m>", "<G>", "<Gm>",
    "<G#>", "<G#m>", "<A>", "<Am>", "<A#>", "<A#m>", "<B>", "<Bm>",
    "<Cmaj7>", "<Cmin7>", "<C7>", "<Cdim>", "<Caug>",
    "<Dmaj7>", "<Dmin7>", "<D7>", "<Ddim>", "<Daug>",
    "<Emaj7>", "<Emin7>", "<E7>", "<Edim>", "<Eaug>",
    "<Fmaj7>", "<Fmin7>", "<F7>", "<Fdim>", "<Faug>",
    "<Gmaj7>", "<Gmin7>", "<G7>", "<Gdim>", "<Gaug>",
    "<Amaj7>", "<Amin7>", "<A7>", "<Adim>", "<Aaug>",
    "<Bmaj7>", "<Bmin7>", "<B7>", "<Bdim>", "<Baug>",
]
SPECIAL_TOKENS = VELOCITY_TOKENS + CHORD_TOKENS + ["[UNCOND]"]

if OUTPUT_DIR.exists() and (OUTPUT_DIR / "model.safetensors").exists():
    print(f"Merged model already exists at {OUTPUT_DIR}")
    print("Delete it first if you want to re-merge.")
    sys.exit(0)

print("Step 1/4  Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
print(f"          Vocab size after extension: {len(tokenizer)}")

print("Step 2/4  Loading base model on CPU (bf16)...")
print("          This takes ~1-2 min depending on disk speed...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
base_model.resize_token_embeddings(len(tokenizer))

print(f"Step 3/4  Loading LoRA adapter from {LORA_CHECKPOINT.name}...")
peft_model = PeftModel.from_pretrained(
    base_model, str(LORA_CHECKPOINT), torch_dtype=torch.bfloat16
)

print("Step 3/4  Merging LoRA weights into base (this is fast)...")
merged = peft_model.merge_and_unload()
merged.eval()

print(f"Step 4/4  Saving merged model to {OUTPUT_DIR} ...")
OUTPUT_DIR.mkdir(exist_ok=True)
merged.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(OUTPUT_DIR))

size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
print(f"\nDone! Saved {size_mb:.0f} MB to {OUTPUT_DIR}")
print("Next: python app.py   (startup will be ~4x faster)")
