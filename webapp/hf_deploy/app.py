"""
MIDI-LLM Demo on HF Spaces (CPU).

Features:
- Live elapsed-time + token-progress during generation (streaming)
- In-browser MIDI playback via html-midi-player
- 5 pre-rendered example outputs (input prompt + ready-to-play MIDI)
"""
import base64
import queue
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from anticipation.convert import events_to_midi

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Pyramidk/midi-llm-merged"
EXAMPLES_DIR = Path(__file__).parent / "examples"

LLAMA_VOCAB_SIZE = 128256
AMT_GPT2_BOS_ID = 55026
SYSTEM_PROMPT = (
    "You are a world-class composer. "
    "Please compose some music according to the following description: "
)

# ── Load model once at startup ─────────────────────────────────────────────────
print(f"Loading tokenizer from Hub: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("Loading merged model from Hub on CPU (fp32)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()
print("Model ready.")

# ── Pre-load example MIDIs (input prompt + base64 MIDI) ────────────────────────
EXAMPLES_INFO = [
    ("upbeat jazz with piano and drums", "sample_03.mid"),
    ("dramatic film score", "sample_15.mid"),
    ("melancholic cello solo", "sample_18.mid"),
]


def _build_examples_html() -> str:
    cards = []
    for prompt_text, fn in EXAMPLES_INFO:
        p = EXAMPLES_DIR / fn
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        data_url = f"data:audio/midi;base64,{b64}"
        cards.append(
            f'''
            <div style="background:#1f2937;border-radius:12px;padding:14px 16px;margin:10px 0;
                        border:1px solid #374151;">
              <p style="margin:0 0 10px 0;color:#a78bfa;font-weight:600;font-size:14px;
                        font-family:system-ui;">"{prompt_text}"</p>
              <midi-player src="{data_url}" sound-font></midi-player>
            </div>
            '''
        )
    return "\n".join(cards) if cards else ""


EXAMPLES_HTML = _build_examples_html()


# ── Custom streamer that keeps token IDs and yields them live ──────────────────
class IDStreamer:
    """Captures token IDs from model.generate so we can convert to MIDI later,
    while also exposing them as a Queue for live progress updates."""

    def __init__(self):
        self.queue: "queue.Queue[int | None]" = queue.Queue()
        self.token_ids: list[int] = []
        self._first = True

    def put(self, value):
        # First call holds the prompt tokens; we don't want those.
        if self._first:
            self._first = False
            return
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "flatten"):
            ids = value.flatten().tolist()
        elif isinstance(value, list):
            ids = value
        else:
            ids = [int(value)]
        self.token_ids.extend(ids)
        for tid in ids:
            self.queue.put(tid)

    def end(self):
        self.queue.put(None)


# ── Generator that yields HTML updates ─────────────────────────────────────────
def _progress_html(elapsed: float, n: int, target: int) -> str:
    pct = min(100, int(n * 100 / max(target, 1)))
    return f"""
    <div style="padding:14px 16px;background:#1f2937;border-radius:10px;
                border:1px solid #374151;font-family:system-ui;">
      <p style="margin:0 0 4px 0;color:#e5e7eb;font-size:14px;">
        Generating… <b style="color:#a78bfa;font-size:16px;">{elapsed:.1f}s</b> elapsed
      </p>
      <p style="margin:0 0 10px 0;color:#9ca3af;font-size:12px;">
        {n} / {target} tokens
      </p>
      <div style="height:6px;background:#374151;border-radius:3px;overflow:hidden;">
        <div style="height:100%;width:{pct}%;background:#7c3aed;
                    transition:width 0.3s ease;"></div>
      </div>
    </div>
    """


def _err(msg: str) -> str:
    return f'<div style="color:#fca5a5;padding:12px 16px;font-family:system-ui;">{msg}</div>'


def generate_midi(prompt: str, max_new_tokens: float):
    if not prompt.strip():
        yield _err("Please enter a description first.")
        return

    full_prompt = SYSTEM_PROMPT + prompt + " "
    input_ids = tokenizer(full_prompt, return_tensors="pt", padding=False)["input_ids"]
    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]])
    input_ids = torch.cat([input_ids, midi_bos], dim=1)
    target = int(max_new_tokens)

    streamer = IDStreamer()

    def _run_generate():
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                max_new_tokens=target,
                temperature=1.0,
                top_p=0.98,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    yield _progress_html(0.0, 0, target)
    start = time.time()
    last_yield = 0.0

    while thread.is_alive() or not streamer.queue.empty():
        try:
            tid = streamer.queue.get(timeout=0.4)
            if tid is None:
                break
        except queue.Empty:
            pass
        elapsed = time.time() - start
        n = len(streamer.token_ids)
        if elapsed - last_yield >= 0.6:
            yield _progress_html(elapsed, n, target)
            last_yield = elapsed

    thread.join()
    elapsed = time.time() - start

    valid = [t - LLAMA_VOCAB_SIZE for t in streamer.token_ids if t >= LLAMA_VOCAB_SIZE]
    if not valid:
        yield _err(
            f"Generated {len(streamer.token_ids)} tokens but none mapped to MIDI events. "
            "Try a different prompt or increase token count."
        )
        return

    try:
        midi_obj = events_to_midi(valid)
    except Exception as e:
        yield _err(f"MIDI conversion error: {e}")
        return

    tmp = Path("/tmp") / f"midi_{uuid.uuid4().hex[:8]}.mid"
    midi_obj.save(str(tmp))
    midi_bytes = tmp.read_bytes()
    b64 = base64.b64encode(midi_bytes).decode()
    data_url = f"data:audio/midi;base64,{b64}"

    decoded = tokenizer.decode(streamer.token_ids[:80], skip_special_tokens=False)
    has_velocity = "<vel_" in decoded
    has_chord = any(c in decoded for c in ["<C>", "<Dm>", "<G>", "<Am>", "<F>"])

    yield f"""
    <div style="padding:16px;background:#1f2937;border-radius:12px;
                border:1px solid #374151;font-family:system-ui;">
      <p style="margin:0 0 12px 0;color:#86efac;font-weight:600;">
        Done in {elapsed:.1f}s
      </p>
      <midi-player src="{data_url}" sound-font></midi-player>
      <p style="margin:14px 0 10px 0;font-size:13px;color:#9ca3af;">
        Total tokens: {len(streamer.token_ids)} ·
        Valid MIDI: {len(valid)} ·
        Velocity: {has_velocity} ·
        Chord: {has_chord} ·
        Size: {len(midi_bytes):,} B
      </p>
      <a href="{data_url}" download="midi_llm_output.mid"
         style="display:inline-block;padding:8px 16px;background:#7c3aed;color:white;
                border-radius:6px;text-decoration:none;font-size:13px;font-weight:500;">
        ⬇ Download .mid
      </a>
    </div>
    """


# ── UI ─────────────────────────────────────────────────────────────────────────
HEAD = """
<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.77,npm/@magenta/music@1.23.1/es6/core.js,npm/html-midi-player@1.5.0"></script>
<style>
  midi-player { width: 100%; }
  midi-player::part(control-panel) {
    background: #0f172a;
    border-radius: 8px;
  }
</style>
"""

DESCRIPTION = """
# MIDI-LLM · Text-to-Music Demo

Fine-tuned **Llama 3.2 1B** with QLoRA on the midicaps dataset.
Vocab extended by **16 velocity + 60 chord + 1 CFG** tokens for explicit dynamics & harmony control.

> Free CPU tier — generation **30–180 seconds** depending on token count.
> Live playback via [html-midi-player](https://github.com/cifkao/html-midi-player).
"""

with gr.Blocks(head=HEAD, title="MIDI-LLM Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            prompt_in = gr.Textbox(
                label="Music Description",
                value="a cheerful piano melody",
                lines=2,
            )
            tokens_in = gr.Slider(
                minimum=64, maximum=512, value=128, step=32,
                label="Max new tokens (longer = more music, slower)",
            )
            btn = gr.Button("Generate MIDI", variant="primary", size="lg")

        with gr.Column(scale=3):
            output_html = gr.HTML(label="Result")

    btn.click(
        generate_midi,
        inputs=[prompt_in, tokens_in],
        outputs=output_html,
    )

    if EXAMPLES_HTML:
        gr.Markdown(
            "## Pre-generated Examples\n\n"
            "These were generated by the fine-tuned model during evaluation. "
            "Click ▶ on any player to hear the model's actual output for that prompt."
        )
        gr.HTML(EXAMPLES_HTML)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
