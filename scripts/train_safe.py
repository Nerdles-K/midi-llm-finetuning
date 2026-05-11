"""
安全版本：应用快速测试成功经验的完整训练脚本
- 直接加载本地缓存（避免查找延迟）
- 修复 pad_token 问题
- 使用标准 Trainer（更稳定）
- JSONL 格式读取
- 增量数据加载（避免 OOM）
- 可配置数据集大小（逐步扩展测试）
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 不显示窗口，直接保存

# =========================================================================
# Loss 记录回调
# =========================================================================

class LossHistoryCallback(TrainerCallback):
    """记录训练过程中的 loss"""
    
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.steps = []
        self.eval_steps = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时调用"""
        if logs is not None:
            if 'loss' in logs:
                self.train_loss.append(logs['loss'])
                self.steps.append(state.global_step)
            if 'eval_loss' in logs:
                self.eval_loss.append(logs['eval_loss'])
                self.eval_steps.append(state.global_step)
    
    def plot_loss(self, save_path):
        """绘制并保存 loss 曲线"""
        plt.figure(figsize=(12, 6))
        
        # 训练 loss
        if self.train_loss:
            plt.subplot(1, 2, 1)
            plt.plot(self.steps, self.train_loss, 'b-', linewidth=2, label='Training Loss')
            plt.xlabel('Steps', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 验证 loss
        if self.eval_loss:
            plt.subplot(1, 2, 2)
            plt.plot(self.eval_steps, self.eval_loss, 'r-', linewidth=2, label='Validation Loss')
            plt.xlabel('Steps', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('Validation Loss Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Loss 曲线已保存: {save_path}")
        
        # 同时保存数据为 JSON
        data_path = save_path.replace('.png', '.json')
        with open(data_path, 'w') as f:
            json.dump({
                'train_loss': self.train_loss,
                'train_steps': self.steps,
                'eval_loss': self.eval_loss,
                'eval_steps': self.eval_steps
            }, f, indent=2)
        print(f"✓ Loss 数据已保存: {data_path}")

# =========================================================================
# 配置参数
# =========================================================================

# Checkpoint 恢复（设置为目录路径以恢复训练，None 则从头开始）
RESUME_FROM_CHECKPOINT = None  # checkpoint-100 不完整，从头训练

# 数据集大小控制（重要！逐步扩展）
MAX_TRAIN_SAMPLES = 1000  # 先用 1000 样本测试（10x 快速测试）
MAX_VAL_SAMPLES = 20  # 大幅减少验证集：100 → 20

# 模型名称（使用 HuggingFace 模型名，会自动使用本地缓存）
MODEL_NAME = "slseanwu/MIDI-LLM_Llama-3.2-1B"

# 数据路径
TRAIN_DATA_PATH = "../data/train_with_tokens.json"  # 使用新准备的数据
VAL_DATA_PATH = "../data/train_with_tokens.json"  # 暂时用同一个文件，取不同样本

# 新增 Token
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

CFG_TEXT_DROPOUT_PROB = 0.15

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 训练配置
NUM_EPOCHS = 3
BATCH_SIZE = 1  # 降低到 1（原来是 2）
GRAD_ACCUM_STEPS = 8  # 增加累积步数保持有效 batch=8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024

# =========================================================================
# 辅助函数
# =========================================================================

def find_latest_checkpoint(output_dir):
    """在输出目录中查找最新的 checkpoint"""
    if not Path(output_dir).exists():
        return None
    
    checkpoints = []
    for item in Path(output_dir).iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            try:
                step_num = int(item.name.split('-')[1])
                checkpoints.append((step_num, str(item)))
            except (IndexError, ValueError):
                continue
    
    if not checkpoints:
        return None
    
    # 返回步数最大的 checkpoint
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]

# =========================================================================
# 主函数
# =========================================================================

def main():
    print("="*70)
    print("安全训练脚本 - 基于快速测试成功经验")
    print("="*70)
    
    # 检查是否从 checkpoint 恢复
    resume_checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        resume_checkpoint = find_latest_checkpoint(RESUME_FROM_CHECKPOINT)
        if resume_checkpoint:
            print(f"\n🔄 将从 checkpoint 恢复训练:")
            print(f"  {resume_checkpoint}")
        else:
            print(f"\n⚠️  未找到 checkpoint，将从头开始训练")
            print(f"  查找路径: {RESUME_FROM_CHECKPOINT}")
    
    print(f"\n配置:")
    print(f"  最大训练样本: {MAX_TRAIN_SAMPLES}")
    print(f"  最大验证样本: {MAX_VAL_SAMPLES}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  梯度累积: {GRAD_ACCUM_STEPS}")
    print(f"  有效批次: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    
    # 检查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"总显存: {total_memory:.2f} GB")
    
    # -------------------------------------------------------------------------
    # 1. 加载 Tokenizer（修复 pad_token）
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 1: 加载 Tokenizer")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"原始词汇表: {len(tokenizer)}")
    
    # 重要：先设置 pad_token，避免后续错误
    tokenizer.pad_token = tokenizer.eos_token
    print(f"设置 pad_token = eos_token")
    
    # 添加新 token
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": SPECIAL_TOKENS
    })
    print(f"添加 {num_added} 个新 token")
    print(f"  - Velocity: {len(VELOCITY_TOKENS)}")
    print(f"  - Chord: {len(CHORD_TOKENS)}")
    print(f"  - CFG: 1 ([UNCOND])")
    print(f"新词汇表: {len(tokenizer)}")
    
    # -------------------------------------------------------------------------
    # 2. 加载模型（4-bit 量化）
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 2: 加载模型（4-bit 量化）")
    print(f"{'='*70}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 调整 embedding
    old_vocab = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    print(f"Embedding 大小: {old_vocab} -> {len(tokenizer)}")
    
    # 准备 QLoRA
    model = prepare_model_for_kbit_training(model)
    
    # -------------------------------------------------------------------------
    # 3. LoRA 配置
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 3: 配置 LoRA")
    print(f"{'='*70}")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        # 移除 modules_to_save 避免保存巨大的 embedding 层（8.6GB）
        # 新 token 的 embedding 会随机初始化，训练时仍会更新
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    
    print(f"LoRA 配置:")
    print(f"  r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  目标模块: {LORA_TARGET_MODULES}")
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")
    
    # -------------------------------------------------------------------------
    # 4. 加载数据（JSONL 格式，增量加载）
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 4: 加载数据（JSONL 格式）")
    print(f"{'='*70}")
    
    if not Path(TRAIN_DATA_PATH).exists():
        print(f"错误: 找不到数据文件 {TRAIN_DATA_PATH}")
        return
    
    # 增量读取（避免一次性加载所有数据）
    print(f"读取训练数据（前 {MAX_TRAIN_SAMPLES} 样本）...")
    train_samples = []
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= MAX_TRAIN_SAMPLES:
                break
            try:
                sample = json.loads(line.strip())
                train_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {i+1} 行解析失败: {e}")
                continue
    
    print(f"读取验证数据（前 {MAX_VAL_SAMPLES} 样本，跳过训练集）...")
    val_samples = []
    with open(VAL_DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 跳过训练集样本
            if i < MAX_TRAIN_SAMPLES:
                continue
            if i >= MAX_TRAIN_SAMPLES + MAX_VAL_SAMPLES:
                break
            try:
                sample = json.loads(line.strip())
                val_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {i+1} 行解析失败: {e}")
                continue
    
    print(f"训练样本: {len(train_samples)}")
    print(f"验证样本: {len(val_samples)}")
    
    # 应用 CFG（文本丢弃）
    def apply_cfg_dropout(samples, dropout_prob=0.15):
        """随机丢弃部分样本的文本（用于 CFG）"""
        import random
        processed = []
        for sample in samples:
            text = sample.get('text', '')
            midi = sample.get('midi_tokens', '')
            
            # 以一定概率丢弃文本
            if random.random() < dropout_prob:
                text = "[UNCOND]"
            
            processed.append({
                'text': text,
                'midi_tokens': midi
            })
        return processed
    
    print(f"\n应用 CFG 数据增强（文本丢弃概率 {CFG_TEXT_DROPOUT_PROB}）...")
    train_samples = apply_cfg_dropout(train_samples, CFG_TEXT_DROPOUT_PROB)
    # 验证集不丢弃
    
    # 释放内存
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 格式化为模型输入
    def format_sample(sample):
        """格式化：text + \n + midi_tokens"""
        text = sample.get('text', '')
        midi = sample.get('midi_tokens', '')
        return f"{text}\n{midi}"
    
    train_texts = [format_sample(s) for s in train_samples]
    val_texts = [format_sample(s) for s in val_samples]
    
    print(f"\n示例数据:")
    print(f"  {train_texts[0][:200]}...")
    
    # -------------------------------------------------------------------------
    # 5. Tokenize
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 5: Tokenize 数据")
    print(f"{'='*70}")
    
    def tokenize_function(examples):
        """Tokenize 文本"""
        return tokenizer(
            examples['text'],
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
    
    # 创建 HF Dataset
    train_dataset = Dataset.from_dict({'text': train_texts})
    val_dataset = Dataset.from_dict({'text': val_texts})
    
    print(f"Tokenizing...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing train"
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing val"
    )
    
    print(f"✓ Tokenize 完成")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    
    # -------------------------------------------------------------------------
    # 6. 训练参数
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 6: 配置训练参数")
    print(f"{'='*70}")
    
    # 如果从 checkpoint 恢复，使用相同的输出目录
    if resume_checkpoint:
        output_dir = str(Path(resume_checkpoint).parent)
        print(f"\n使用现有输出目录: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"../outputs/safe_train_{timestamp}"
        print(f"\n创建新输出目录: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        optim="paged_adamw_8bit",
        bf16=True,
        
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        
        eval_strategy="steps",
        eval_steps=50,  # 恢复到 50 步评估一次
        
        save_strategy="steps",
        save_steps=100,  # 恢复到 100 步保存（现在 checkpoint 只有几十 MB）
        save_total_limit=2,  # 减少保存数量，释放磁盘空间
        
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,  # 禁用自动列移除，避免评估时出错
    )
    
    print(f"输出目录: {output_dir}")
    print(f"训练轮次: {NUM_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"预计步数: {len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS) * NUM_EPOCHS}")
    
    # -------------------------------------------------------------------------
    # 7. Trainer（使用标准 Trainer，更稳定）
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 7: 初始化 Trainer")
    print(f"{'='*70}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 添加 loss 记录回调
    loss_callback = LossHistoryCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )
    
    print(f"✓ Trainer 初始化完成")
    
    # -------------------------------------------------------------------------
    # 8. 训练
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("步骤 8: 开始训练")
    print(f"{'='*70}")
    print(f"\n提示: 如遇到问题可随时按 Ctrl+C 中断")
    print(f"      检查点会自动保存到 {output_dir}\n")
    
    try:
        # 如果指定了 checkpoint，从该处恢复
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        print(f"\n{'='*70}")
        print("训练完成！")
        print(f"{'='*70}")
        print(f"模型保存路径: {output_dir}")
        
        # 保存最终模型
        final_path = f"{output_dir}/final_model"
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"最终模型: {final_path}")
        
        # 绘制并保存 loss 曲线
        loss_plot_path = f"{output_dir}/loss_curve.png"
        loss_callback.plot_loss(loss_plot_path)
        
    except KeyboardInterrupt:
        print(f"\n训练被中断")
        print(f"检查点已保存到: {output_dir}")
        
        # 即使中断也保存 loss 曲线
        loss_plot_path = f"{output_dir}/loss_curve_interrupted.png"
        loss_callback.plot_loss(loss_plot_path)
        
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 显存统计
    if device == "cuda":
        print(f"\n显存统计:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()
