"""
快速评估脚本：对比微调前后的模型
生成少量样本进行质量评估
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
from datetime import datetime
from pathlib import Path
from anticipation.convert import events_to_midi
import time

# =============================================================================
# 配置
# =============================================================================

BASE_MODEL = "slseanwu/MIDI-LLM_Llama-3.2-1B"
FINE_TUNED_CHECKPOINT = "../outputs/safe_train_20251219_185027/final_model"

# Velocity and Chord tokens
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
    "<Bmaj7>", "<Bmin7>", "<B7>", "<Bdim>", "<Baug>"
]
SPECIAL_TOKENS = VELOCITY_TOKENS + CHORD_TOKENS + ["[UNCOND]"]

# Generation parameters
MAX_NEW_TOKENS = 2048
TEMPERATURE = 1.0
TOP_P = 0.98
NUM_SAMPLES = 20  # 生成 20 个样本（快速评估）

# System prompt and MIDI tokens
SYSTEM_PROMPT = "You are a world-class composer. Please compose some music according to the following description: "
AMT_GPT2_BOS_ID = 55026
LLAMA_VOCAB_SIZE = 128256

# Test prompts (covering different styles and emotions)
TEST_PROMPTS = [
    "a cheerful piano melody",
    "a sad violin solo",
    "upbeat jazz with piano and drums",
    "gentle acoustic guitar",
    "energetic rock guitar riff",
    "classical piano sonata",
    "electronic dance music with synthesizers",
    "slow blues with harmonica",
    "fast-paced metal with heavy guitars",
    "ambient atmospheric soundscape",
    "folk music with acoustic instruments",
    "romantic orchestral strings",
    "funky bass groove",
    "happy children's song",
    "dramatic film score",
    "relaxing spa music",
    "intense action movie theme",
    "melancholic cello solo",
    "triumphant brass fanfare",
    "mysterious dark ambient"
]

# =============================================================================
# Model Loading
# =============================================================================

def load_model(use_lora=False):
    """Load model (original or fine-tuned)"""
    
    print("="*70)
    print(f"Loading {'Fine-tuned' if use_lora else 'Original'} Model...")
    print("="*70)
    
    # Tokenizer
    print("\n1. Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    if use_lora:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": SPECIAL_TOKENS
        })
        print(f"   Vocabulary: {len(tokenizer) - num_added} + {num_added} = {len(tokenizer)}")
    else:
        print(f"   Vocabulary: {len(tokenizer)} (original)")
    
    # Base model
    print("\n2. Loading Base Model")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if use_lora:
        base_model.resize_token_embeddings(len(tokenizer))
    
    print("   ✓ Base model loaded")
    
    # Load LoRA if needed
    if use_lora:
        print(f"\n3. Loading LoRA Adapter")
        model = PeftModel.from_pretrained(
            base_model,
            FINE_TUNED_CHECKPOINT,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print("   ✓ LoRA loaded")
    else:
        print(f"\n3. Using Original Model")
        model = base_model
        model.eval()
        print("   ✓ Original model ready")
    
    print("\n" + "="*70)
    print("Model Ready!")
    print("="*70 + "\n")
    
    return model, tokenizer

def generate_midi(model, tokenizer, prompt):
    """Generate MIDI from text prompt"""
    
    # Full prompt
    full_prompt = SYSTEM_PROMPT + prompt + " "
    
    # Tokenize
    llama_input = tokenizer(full_prompt, return_tensors="pt", padding=False)
    input_ids = llama_input["input_ids"]
    
    # Add MIDI BOS token
    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]])
    input_ids = torch.cat([input_ids, midi_bos], dim=1)
    input_ids = input_ids.to("cuda")
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    gen_time = time.time() - start_time
    
    # Extract generated tokens
    prompt_length = input_ids.shape[1]
    generated_token_ids = outputs[0][prompt_length:].cpu().tolist()
    
    return generated_token_ids, gen_time

def tokens_to_midi_file(token_ids, output_path):
    """Convert token IDs to MIDI file"""
    
    try:
        # Filter valid MIDI tokens
        valid_tokens = [t - LLAMA_VOCAB_SIZE for t in token_ids if t >= LLAMA_VOCAB_SIZE]
        
        if not valid_tokens:
            return False
        
        # Convert to MIDI
        midi_obj = events_to_midi(valid_tokens)
        midi_obj.save(str(output_path))
        
        return True
        
    except Exception as e:
        print(f"  ✗ MIDI conversion error: {e}")
        return False

# =============================================================================
# Evaluation
# =============================================================================

def run_evaluation():
    """Run quick evaluation on both models"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"../results/evaluation_results_{timestamp}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        "timestamp": timestamp,
        "num_samples": NUM_SAMPLES,
        "prompts": TEST_PROMPTS[:NUM_SAMPLES],
        "original_model": {},
        "fine_tuned_model": {},
        "comparison": {}
    }
    
    # Evaluate both models
    for model_type in ["original", "fine_tuned"]:
        use_lora = (model_type == "fine_tuned")
        
        print("\n" + "="*70)
        print(f"Evaluating {model_type.upper()} Model")
        print("="*70)
        
        # Load model
        model, tokenizer = load_model(use_lora=use_lora)
        
        # Create output directory
        model_dir = output_dir / model_type
        model_dir.mkdir(exist_ok=True)
        
        # Generate samples
        success_count = 0
        total_time = 0
        total_tokens = 0
        
        for i, prompt in enumerate(TEST_PROMPTS[:NUM_SAMPLES]):
            print(f"\n[{i+1}/{NUM_SAMPLES}] Generating: {prompt}")
            
            try:
                # Generate
                token_ids, gen_time = generate_midi(model, tokenizer, prompt)
                total_time += gen_time
                total_tokens += len(token_ids)
                
                # Save MIDI
                midi_file = model_dir / f"sample_{i+1:02d}.mid"
                success = tokens_to_midi_file(token_ids, midi_file)
                
                if success:
                    success_count += 1
                    print(f"  ✓ Saved: {midi_file.name} ({len(token_ids)} tokens, {gen_time:.2f}s)")
                else:
                    print(f"  ✗ Failed to convert")
                    
            except Exception as e:
                print(f"  ✗ Generation error: {e}")
        
        # Save statistics
        avg_time = total_time / NUM_SAMPLES if NUM_SAMPLES > 0 else 0
        avg_tokens = total_tokens / NUM_SAMPLES if NUM_SAMPLES > 0 else 0
        
        results[f"{model_type}_model"] = {
            "success_count": success_count,
            "success_rate": success_count / NUM_SAMPLES,
            "avg_generation_time": avg_time,
            "avg_tokens_per_sample": avg_tokens,
            "total_time": total_time
        }
        
        print(f"\n{model_type.upper()} Model Statistics:")
        print(f"  Success: {success_count}/{NUM_SAMPLES} ({100*success_count/NUM_SAMPLES:.1f}%)")
        print(f"  Avg Time: {avg_time:.2f}s per sample")
        print(f"  Avg Tokens: {avg_tokens:.0f} tokens per sample")
        
        # Clean up GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()
    
    # Save comparison results
    results["comparison"] = {
        "success_rate_improvement": (
            results["fine_tuned_model"]["success_rate"] - 
            results["original_model"]["success_rate"]
        ),
        "speed_improvement": (
            results["original_model"]["avg_generation_time"] - 
            results["fine_tuned_model"]["avg_generation_time"]
        )
    }
    
    # Save JSON report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {results['original_model']['success_count']} Original model samples")
    print(f"  - {results['fine_tuned_model']['success_count']} Fine-tuned model samples")
    print(f"  - evaluation_report.json")
    print("\nNext steps:")
    print("  1. Listen to the MIDI files to compare quality")
    print("  2. Run FAD and CLAP metrics if needed")
    print("  3. Check evaluation_report.json for statistics")
    print("="*70)
    
    return output_dir, results

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    output_dir, results = run_evaluation()
