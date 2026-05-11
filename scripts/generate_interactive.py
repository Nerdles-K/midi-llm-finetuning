"""
交互式 MIDI 生成：输入提示词，生成 MIDI 文件
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from datetime import datetime
from pathlib import Path
from anticipation.convert import events_to_midi

# =============================================================================
# 配置
# =============================================================================

BASE_MODEL = "slseanwu/MIDI-LLM_Llama-3.2-1B"
LORA_CHECKPOINT = None  # Set to None for original model, or path for fine-tuned model
# LORA_CHECKPOINT = "../outputs/safe_train_20251219_185027/final_model"  # Uncomment for fine-tuned

# 新增的 tokens
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

# 生成参数
MAX_NEW_TOKENS = 2048  # Increase to allow full MIDI generation
TEMPERATURE = 1.0      # Match original model's temperature
TOP_P = 0.98           # Match original model's top_p

# System prompt (critical for MIDI generation!)
SYSTEM_PROMPT = "You are a world-class composer. Please compose some music according to the following description: "

# MIDI special token
AMT_GPT2_BOS_ID = 55026
LLAMA_VOCAB_SIZE = 128256

# =============================================================================
# 初始化模型（只加载一次）
# =============================================================================

def load_model():
    """Load model and tokenizer"""
    
    print("="*70)
    print("Loading Model...")
    print("="*70)
    
    # Tokenizer
    print("\n1. Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Only add special tokens if using fine-tuned model
    if LORA_CHECKPOINT:
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": SPECIAL_TOKENS
        })
        print(f"   Vocabulary: {len(tokenizer) - num_added} + {num_added} = {len(tokenizer)}")
    else:
        print(f"   Vocabulary: {len(tokenizer)} (original, no new tokens)")
    
    # Base model
    print("\n2. Loading Base Model (4-bit quantization)")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Resize embeddings only if using fine-tuned model
    if LORA_CHECKPOINT:
        base_model.resize_token_embeddings(len(tokenizer))
    
    print("   ✓ Base model loaded")
    
    # Load LoRA only if checkpoint is specified
    if LORA_CHECKPOINT:
        print(f"\n3. Loading LoRA Adapter")
        print(f"   Checkpoint: {LORA_CHECKPOINT}")
        model = PeftModel.from_pretrained(
            base_model,
            LORA_CHECKPOINT,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print("   ✓ LoRA loaded")
    else:
        print(f"\n3. Using Original Model (No LoRA)")
        model = base_model
        model.eval()
        print("   ✓ Original model ready")
    
    print("\n" + "="*70)
    print("Model Ready!")
    print("="*70)
    
    return model, tokenizer

def generate_midi(model, tokenizer, prompt):
    """Generate MIDI tokens and IDs"""
    
    # Add system prompt (CRITICAL!)
    full_prompt = SYSTEM_PROMPT + prompt + " "
    
    # Tokenize
    llama_input = tokenizer(full_prompt, return_tensors="pt", padding=False)
    input_ids = llama_input["input_ids"]
    
    # Add MIDI BOS token (CRITICAL! Signals start of MIDI generation)
    midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]])
    input_ids = torch.cat([input_ids, midi_bos], dim=1)
    
    # Move to device
    input_ids = input_ids.to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_midi_str = generated_text[len(full_prompt):].strip()
    
    # Get token IDs (only the generated part, excluding prompt + BOS)
    prompt_length = input_ids.shape[1]
    generated_token_ids = outputs[0][prompt_length:].cpu().tolist()
    
    return generated_midi_str, generated_token_ids

def tokens_to_midi_file(token_ids, output_path):
    """Convert token IDs to MIDI file
    
    Args:
        token_ids: List of token IDs (already in MIDI range)
        output_path: Path to save .mid file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Filter only valid MIDI tokens (128256 to 183286)
        LLAMA_VOCAB_SIZE = 128256
        valid_tokens = [t - LLAMA_VOCAB_SIZE for t in token_ids if t >= LLAMA_VOCAB_SIZE]
        
        if not valid_tokens:
            print("  ✗ No valid MIDI tokens found")
            return False
        
        # Convert to MIDI using anticipation
        midi_obj = events_to_midi(valid_tokens)
        midi_obj.save(str(output_path))
        
        return True
        
    except Exception as e:
        print(f"  ✗ MIDI conversion error: {e}")
        return False

def save_output(prompt, midi_tokens_str, midi_token_ids):
    """Save generated output (text tokens + MIDI file)"""
    
    os.makedirs("../generated_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text tokens
    txt_file = f"../generated_outputs/output_{timestamp}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write(f"Generated MIDI tokens:\n{midi_tokens_str}\n")
    
    # Convert to MIDI file
    midi_file = f"../generated_outputs/output_{timestamp}.mid"
    success = tokens_to_midi_file(midi_token_ids, midi_file)
    
    return txt_file, midi_file, success

# =============================================================================
# 主交互循环
# =============================================================================

def main():
    """Interactive generation loop"""
    
    # Load model (only once)
    model, tokenizer = load_model()
    
    print("\n" + "="*70)
    print("Interactive MIDI Generation")
    print("="*70)
    print("\nInstructions:")
    print("  - Enter text description (e.g., a cheerful piano melody)")
    print("  - Enter 'quit' or 'exit' to quit")
    print("  - Enter 'examples' to view example prompts")
    print("="*70)
    
    # 示例提示
    examples = [
        "a cheerful piano melody",
        "a sad violin solo",
        "upbeat jazz with piano",
        "gentle acoustic guitar",
        "energetic rock guitar riff",
        "classical piano sonata",
        "electronic dance music",
        "slow blues with harmonica"
    ]
    
    generation_count = 0
    
    while True:
        print("\n" + "-"*70)
        
        # Get user input
        try:
            prompt = input("\nEnter prompt: ").strip()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        # Handle special commands
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if prompt.lower() == 'examples':
            print("\nExample prompts:")
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            continue
        
        if not prompt:
            print("Please enter a valid prompt")
            continue
        
        # Generate MIDI
        print(f"\nGenerating...")
        try:
            midi_tokens_str, midi_token_ids = generate_midi(model, tokenizer, prompt)
            generation_count += 1
            
            # Display results
            print(f"\n✓ Generation complete!")
            print(f"\nGenerated MIDI tokens (first 500 chars):")
            print(f"{midi_tokens_str[:500]}...")
            
            # Check for new tokens
            has_velocity = '<vel_' in midi_tokens_str
            has_chord = any(c in midi_tokens_str for c in ['<C>', '<Dm>', '<G>', '<Am>'])
            
            print(f"\nFeatures:")
            print(f"  Contains Velocity tokens: {'✓' if has_velocity else '✗'}")
            print(f"  Contains Chord tokens: {'✓' if has_chord else '✗'}")
            
            # Save
            txt_file, midi_file, midi_success = save_output(prompt, midi_tokens_str, midi_token_ids)
            print(f"\n✓ Saved tokens: {txt_file}")
            if midi_success:
                print(f"✓ Saved MIDI file: {midi_file}")
            else:
                print(f"✗ MIDI file conversion failed")
            
        except Exception as e:
            print(f"\n✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Exit statistics
    print("\n" + "="*70)
    print(f"Total generations: {generation_count}")
    print("="*70)

if __name__ == "__main__":
    main()
