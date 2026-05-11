"""
MIDI-LLM Web Demo
FastAPI app: text prompt -> fine-tuned Llama 3.2 1B -> MIDI file -> browser playback

Run (normal):    python app.py
Run (test mode): SKIP_MODEL=1 python app.py   -- skips model loading, uses a sample MIDI
"""

import os
import uuid
import asyncio
import threading
import concurrent.futures
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from anticipation.convert import events_to_midi

# ── Config ─────────────────────────────────────────────────────────────────────
SKIP_MODEL = os.environ.get("SKIP_MODEL", "0") == "1"   # test mode flag

BASE_DIR = Path(__file__).parent.parent
LORA_CHECKPOINT = BASE_DIR / "outputs" / "safe_train_20251219_185027" / "final_model"
BASE_MODEL = "slseanwu/MIDI-LLM_Llama-3.2-1B"
MERGED_MODEL_DIR = Path(__file__).parent / "merged_model"   # produced by merge_model.py
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

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

SYSTEM_PROMPT = "You are a world-class composer. Please compose some music according to the following description: "
AMT_GPT2_BOS_ID = 55026
LLAMA_VOCAB_SIZE = 128256

# ── Global model state ─────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_ready = False
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_generation_lock = threading.Lock()


def _load_model_sync():
    global _model, _tokenizer, _model_ready

    if SKIP_MODEL:
        print("SKIP_MODEL=1: skipping model load (test mode)")
        _model_ready = True
        return

    if MERGED_MODEL_DIR.exists() and (MERGED_MODEL_DIR / "model.safetensors").exists():
        # Fast path: pre-merged model (no 4-bit quantization, no LoRA loading)
        print(f"Loading tokenizer from {MERGED_MODEL_DIR.name}/...")
        _tokenizer = AutoTokenizer.from_pretrained(
            str(MERGED_MODEL_DIR), local_files_only=True
        )
        print("Loading merged model (bf16, no quantization)...")
        _model = AutoModelForCausalLM.from_pretrained(
            str(MERGED_MODEL_DIR),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
    else:
        # Fallback: base model + 4-bit quantization + LoRA (slow, ~60-90s)
        print("[slow path] Run merge_model.py once to enable fast startup.")
        print("Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

        print("Loading base model (4-bit quantization)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model.resize_token_embeddings(len(_tokenizer))

        print(f"Loading LoRA adapter from {LORA_CHECKPOINT}...")
        _model = PeftModel.from_pretrained(
            base_model, str(LORA_CHECKPOINT), torch_dtype=torch.bfloat16
        )

    _model.eval()
    _model_ready = True
    print("Model ready!")


def _generate_sync(prompt: str):
    if SKIP_MODEL:
        # Return fake token ids for testing (copy an existing MIDI file)
        existing = list((BASE_DIR / "results" / "evaluation_results_20251219_231707" / "fine_tuned").glob("*.mid"))
        if existing:
            import mido
            mid = mido.MidiFile(str(existing[0]))
            # Return dummy data with valid structure
            return "<C> <vel_10> test", [LLAMA_VOCAB_SIZE + 100] * 500
        return "<test>", []

    full_prompt = SYSTEM_PROMPT + prompt + " "
    input_ids = _tokenizer(full_prompt, return_tensors="pt", padding=False)["input_ids"]
    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]])
    input_ids = torch.cat([input_ids, midi_bos], dim=1).to("cuda")

    with torch.no_grad():
        outputs = _model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.98,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    generated_text = _tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_midi_str = generated_text[len(full_prompt):].strip()
    token_ids = outputs[0][input_ids.shape[1]:].cpu().tolist()
    return generated_midi_str, token_ids


def _tokens_to_midi_file(token_ids, output_path):
    if SKIP_MODEL:
        # Copy an existing sample MIDI for testing
        import shutil
        existing = list((BASE_DIR / "results" / "evaluation_results_20251219_231707" / "fine_tuned").glob("*.mid"))
        if existing:
            shutil.copy(existing[0], output_path)
            return True
        return False

    valid = [t - LLAMA_VOCAB_SIZE for t in token_ids if t >= LLAMA_VOCAB_SIZE]
    if not valid:
        return False
    try:
        midi_obj = events_to_midi(valid)
        midi_obj.save(str(output_path))
        return True
    except Exception as e:
        print(f"MIDI conversion error: {e}")
        return False


# ── FastAPI app ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _load_model_sync)
    yield


app = FastAPI(title="MIDI-LLM Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/health")
async def health():
    return {"ready": _model_ready}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/generate")
async def generate(request: GenerateRequest):
    if not _model_ready:
        return JSONResponse({"error": "Model is still loading. Please wait."}, status_code=503)

    if not _generation_lock.acquire(blocking=False):
        return JSONResponse(
            {"error": "Another generation is in progress. Please try again shortly."},
            status_code=429,
        )

    try:
        loop = asyncio.get_running_loop()
        midi_str, token_ids = await loop.run_in_executor(
            _executor, _generate_sync, request.prompt
        )

        filename = f"midi_{uuid.uuid4().hex[:8]}.mid"
        filepath = STATIC_DIR / filename
        success = _tokens_to_midi_file(token_ids, filepath)

        if not success:
            return JSONResponse(
                {"error": "MIDI conversion failed — no valid tokens generated."},
                status_code=500,
            )

        valid_count = sum(1 for t in token_ids if t >= LLAMA_VOCAB_SIZE)
        has_velocity = "<vel_" in midi_str
        has_chord = any(
            c in midi_str for c in ["<C>", "<Dm>", "<G>", "<Am>", "<F>", "<Cmaj7>"]
        )

        return JSONResponse({
            "midi_url": f"/static/{filename}",
            "token_count": len(token_ids),
            "valid_midi_tokens": valid_count,
            "has_velocity": has_velocity,
            "has_chord": has_chord,
        })
    finally:
        _generation_lock.release()


# ── Frontend HTML ──────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MIDI-LLM · Text-to-Music Demo</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/combine/npm/tone@14,npm/@magenta/music@1.23.1/es6/core.js,npm/html-midi-player@1.5.0"></script>
  <style>
    body { font-family: system-ui, sans-serif; }
    midi-player { width: 100%; }
    midi-player::part(control-panel) {
      background: #1e293b;
      border-radius: 8px;
      padding: 8px 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spin { animation: spin 1s linear infinite; }
    @keyframes pulse-bar { 0%,100% { width: 15%; } 50% { width: 85%; } }
    .pulse-bar { animation: pulse-bar 2.5s ease-in-out infinite; }
  </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">

  <!-- Header -->
  <div class="border-b border-gray-800 px-8 py-4 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center font-bold text-sm">M</div>
      <div>
        <h1 class="font-semibold text-white text-sm">MIDI-LLM Demo</h1>
        <p class="text-xs text-gray-500">Llama 3.2 1B &middot; LoRA Fine-tuned &middot; Velocity + Chord + CFG tokens</p>
      </div>
    </div>
    <div id="model-badge" class="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full bg-gray-800 text-yellow-400">
      <div class="w-2 h-2 rounded-full bg-yellow-400 animate-pulse"></div>
      Loading model...
    </div>
  </div>

  <!-- Main -->
  <div class="max-w-2xl mx-auto px-6 py-10 space-y-6">

    <!-- Input card -->
    <div class="bg-gray-900 rounded-2xl p-6 border border-gray-800 space-y-4">
      <label class="block text-sm font-medium text-gray-300">Music Description</label>
      <textarea
        id="prompt" rows="2"
        placeholder="a cheerful piano melody..."
        class="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-white placeholder-gray-500
               focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 resize-none text-sm"
      ></textarea>

      <!-- Example chips -->
      <div class="flex flex-wrap gap-2 items-center">
        <span class="text-xs text-gray-500">Try:</span>
        <button class="text-xs px-3 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-300
                       hover:border-purple-500 hover:text-purple-300 transition-colors"
          onclick="setPrompt('a cheerful piano melody')">cheerful piano</button>
        <button class="text-xs px-3 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-300
                       hover:border-purple-500 hover:text-purple-300 transition-colors"
          onclick="setPrompt('upbeat jazz with piano and drums')">upbeat jazz</button>
        <button class="text-xs px-3 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-300
                       hover:border-purple-500 hover:text-purple-300 transition-colors"
          onclick="setPrompt('melancholic cello solo')">melancholic cello</button>
        <button class="text-xs px-3 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-300
                       hover:border-purple-500 hover:text-purple-300 transition-colors"
          onclick="setPrompt('dramatic film score with strings')">film score</button>
        <button class="text-xs px-3 py-1 rounded-full bg-gray-800 border border-gray-700 text-gray-300
                       hover:border-purple-500 hover:text-purple-300 transition-colors"
          onclick="setPrompt('electronic dance music with synthesizers')">EDM</button>
      </div>

      <button id="gen-btn" onclick="generate()"
        class="w-full py-3 rounded-xl font-medium text-sm bg-purple-600 hover:bg-purple-500 text-white
               transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
        Generate MIDI &nbsp;&nbsp;Ctrl+Enter
      </button>
    </div>

    <!-- Progress card -->
    <div id="progress-card" class="hidden bg-gray-900 rounded-2xl p-6 border border-gray-800">
      <div class="flex items-center gap-4">
        <div class="w-8 h-8 rounded-full border-2 border-purple-500 border-t-transparent spin flex-shrink-0"></div>
        <div class="flex-1 min-w-0">
          <p class="text-sm font-medium text-white">Composing your music...</p>
          <p class="text-xs text-gray-400 mt-0.5">Generating 2048 tokens — usually 60–90 seconds on RTX 4060</p>
        </div>
        <span id="timer" class="text-xl font-mono font-semibold text-purple-400 flex-shrink-0">0s</span>
      </div>
      <div class="mt-4 h-1 bg-gray-800 rounded-full overflow-hidden">
        <div class="h-full bg-purple-600 rounded-full pulse-bar"></div>
      </div>
    </div>

    <!-- Error card -->
    <div id="error-card" class="hidden bg-red-950 rounded-2xl p-4 border border-red-900">
      <p class="text-sm text-red-300" id="error-msg"></p>
    </div>

    <!-- Result card -->
    <div id="result-card" class="hidden bg-gray-900 rounded-2xl p-6 border border-gray-800 space-y-5">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="font-semibold text-white">Generated MIDI</h2>
          <p class="text-xs text-gray-500 mt-0.5" id="result-prompt-label"></p>
        </div>
        <a id="download-btn" href="#" download="generated.mid"
           class="text-xs px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 transition-colors">
          Download .mid
        </a>
      </div>

      <!-- MIDI Player -->
      <midi-player id="midi-player" sound-font visualizer="#midi-viz"></midi-player>

      <!-- Piano roll visualizer -->
      <midi-visualizer type="piano-roll" id="midi-viz"
        style="height: 140px; display: block; overflow: hidden; border-radius: 8px; background: #0f172a;">
      </midi-visualizer>

      <!-- Stats -->
      <div class="grid grid-cols-4 gap-3">
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <p class="text-base font-bold text-white" id="stat-tokens">-</p>
          <p class="text-xs text-gray-400 mt-0.5">Total tokens</p>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <p class="text-base font-bold text-white" id="stat-midi">-</p>
          <p class="text-xs text-gray-400 mt-0.5">MIDI tokens</p>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <p class="text-base font-bold" id="stat-vel">-</p>
          <p class="text-xs text-gray-400 mt-0.5">Velocity</p>
        </div>
        <div class="bg-gray-800 rounded-xl p-3 text-center">
          <p class="text-base font-bold" id="stat-chord">-</p>
          <p class="text-xs text-gray-400 mt-0.5">Chord</p>
        </div>
      </div>

      <button onclick="reset()"
        class="w-full py-2.5 rounded-xl text-sm text-gray-400 border border-gray-700 hover:bg-gray-800 transition-colors">
        Generate Another
      </button>
    </div>

  </div>

  <script>
    let timerInterval = null;

    async function checkReady() {
      try {
        const r = await fetch('/health');
        const d = await r.json();
        if (d.ready) {
          const badge = document.getElementById('model-badge');
          badge.innerHTML = '<div class="w-2 h-2 rounded-full bg-green-400"></div> Model ready';
          badge.className = 'flex items-center gap-2 text-xs px-3 py-1.5 rounded-full bg-gray-800 text-green-400';
          document.getElementById('gen-btn').disabled = false;
        } else {
          document.getElementById('gen-btn').disabled = true;
          setTimeout(checkReady, 2000);
        }
      } catch {
        setTimeout(checkReady, 3000);
      }
    }

    document.getElementById('gen-btn').disabled = true;
    checkReady();

    function setPrompt(text) {
      document.getElementById('prompt').value = text;
      document.getElementById('prompt').focus();
    }

    function startTimer() {
      const start = Date.now();
      timerInterval = setInterval(() => {
        document.getElementById('timer').textContent =
          Math.floor((Date.now() - start) / 1000) + 's';
      }, 1000);
    }

    function stopTimer() {
      if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    }

    function showProgress() {
      document.getElementById('progress-card').classList.remove('hidden');
      document.getElementById('result-card').classList.add('hidden');
      document.getElementById('error-card').classList.add('hidden');
      document.getElementById('gen-btn').disabled = true;
      document.getElementById('timer').textContent = '0s';
      startTimer();
    }

    function showError(msg) {
      stopTimer();
      document.getElementById('progress-card').classList.add('hidden');
      document.getElementById('error-card').classList.remove('hidden');
      document.getElementById('error-msg').textContent = msg;
      document.getElementById('gen-btn').disabled = false;
    }

    function showResult(data, prompt) {
      stopTimer();
      document.getElementById('progress-card').classList.add('hidden');
      document.getElementById('result-card').classList.remove('hidden');
      document.getElementById('gen-btn').disabled = false;

      document.getElementById('result-prompt-label').textContent = '"' + prompt + '"';
      document.getElementById('midi-player').src = data.midi_url;
      document.getElementById('download-btn').href = data.midi_url;

      document.getElementById('stat-tokens').textContent = data.token_count;
      document.getElementById('stat-midi').textContent = data.valid_midi_tokens;

      const vel = document.getElementById('stat-vel');
      vel.textContent = data.has_velocity ? 'Yes' : 'No';
      vel.className = 'text-base font-bold ' + (data.has_velocity ? 'text-green-400' : 'text-gray-500');

      const chord = document.getElementById('stat-chord');
      chord.textContent = data.has_chord ? 'Yes' : 'No';
      chord.className = 'text-base font-bold ' + (data.has_chord ? 'text-green-400' : 'text-gray-500');
    }

    function reset() {
      document.getElementById('result-card').classList.add('hidden');
      document.getElementById('error-card').classList.add('hidden');
      document.getElementById('prompt').focus();
    }

    async function generate() {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) { document.getElementById('prompt').focus(); return; }

      showProgress();

      try {
        const response = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt }),
        });
        const data = await response.json();
        if (!response.ok) { showError(data.error || 'Generation failed.'); return; }
        showResult(data, prompt);
      } catch (err) {
        showError('Network error: ' + err.message);
      }
    }

    document.getElementById('prompt').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate();
    });
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
