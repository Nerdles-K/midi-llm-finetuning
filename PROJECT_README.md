# MIDI-LLM 微调项目

基于 MIDI-LLM (Llama 3.2 1B) 的参数高效微调，添加 Velocity、Chord 和 CFG 控制。

---

## 📁 项目结构

```
MIDI-LLM/
├── scripts/                          # 核心脚本
│   ├── prepare_training_data.py      # 数据准备：从 MIDI 提取 tokens
│   ├── train_safe.py                 # 训练脚本：LoRA 微调
│   ├── generate_interactive.py       # 推理脚本：交互式生成 MIDI
│   ├── quick_evaluation.py           # 快速评估：生成测试样本
│   └── compute_midi_metrics.py       # MIDI 特征分析
│
├── data/                             # 数据文件
│   ├── train.json                    # 原始元数据
│   └── train_with_tokens.json        # 处理后的训练数据
│
├── results/                          # 评估结果
│   └── evaluation_results_*/         # 评估输出
│       ├── EVALUATION_REPORT.md      # 详细评估报告
│       ├── original/                 # 原始模型生成的 MIDI
│       ├── fine_tuned/               # 微调模型生成的 MIDI
│       ├── evaluation_report.json    # 基础统计
│       └── midi_metrics.json         # MIDI 特征分析
│
├── outputs/                          # 训练输出
│   └── safe_train_*/                 # 训练 checkpoint
│       └── final_model/              # 最终 LoRA 权重
│
├── generated_outputs/                # 生成的 MIDI 文件
│
├── lmd_full/                         # Lakh MIDI Dataset
│
├── archive/                          # 归档文件（不常用）
│
├── midi_llm/                         # 工具库
├── assets/                           # 资源文件
├── requirements.txt                  # Python 依赖
└── README.md                         # 本文件
```

---

## 🚀 快速开始

### 1. 环境设置

```bash
# 创建 conda 环境
conda create -n midi-llm python=3.11 -y
conda activate midi-llm

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
cd scripts
python prepare_training_data.py
```

**输出**: `../data/train_with_tokens.json` (1000 个样本，包含 velocity/chord tokens)

### 3. 训练

```bash
python train_safe.py
```

**配置**:
- LoRA: r=16, alpha=32
- Batch size: 1 × 8 (gradient accumulation)
- Epochs: 3
- GPU: RTX 4060 (8GB VRAM)

**输出**: `../outputs/safe_train_YYYYMMDD_HHMMSS/final_model/`

### 4. 推理

```bash
python generate_interactive.py
```

**配置选项**:
- 使用原始模型: `LORA_CHECKPOINT = None`
- 使用微调模型: `LORA_CHECKPOINT = "../outputs/safe_train_*/final_model"`

**输出**: `../generated_outputs/output_YYYYMMDD_HHMMSS.mid`

### 5. 评估

```bash
# 生成评估样本
python quick_evaluation.py

# 计算 MIDI 特征
python compute_midi_metrics.py
```

**输出**: `../results/evaluation_results_YYYYMMDD_HHMMSS/`

---

## 🔧 技术改进

### 1. Velocity Tokens (16 个)
- 将 MIDI velocity (0-127) 映射到 16 个等级
- Token: `<vel_0>` ~ `<vel_15>`
- 从 MIDI note_on 消息提取

### 2. Chord Tokens (60 个)
- 覆盖常用和弦: 基础三和弦、七和弦、变化和弦
- Token: `<C>`, `<Cmaj7>`, `<Dm>`, etc.
- 从 midicaps 元数据提取

### 3. CFG Token (1 个)
- 实现 Classifier-Free Guidance
- Token: `[UNCOND]`
- 15% 训练样本使用无条件标记

### 4. LoRA 微调
- 参数高效：仅训练 0.38% 参数
- 内存友好：8GB VRAM 可训练
- Checkpoint 小：~50-100MB

---

## 📊 评估结果

详见：[results/evaluation_results_*/EVALUATION_REPORT.md](results/)

### 关键指标

| 指标 | 原始模型 | 微调模型 | 改进 |
|------|---------|---------|------|
| 成功率 | 100% | 100% | 持平 |
| 生成速度 | 55.08s | 82.00s | -48.9% |
| 音高范围 | 47.5 | 51.5 | +8.5% |
| 音高变化 | 11.78 | 12.59 | +6.9% |
| 音符密度 | 25.77 | 22.46 | -12.9% |

---

## 📦 依赖项

核心依赖:
- `torch==2.7.1`
- `transformers==4.57.3`
- `peft==0.18.0`
- `datasets==4.3.0`
- `accelerate==1.12.0`
- `anticipation==1.0` (MIDI 处理)
- `mido==1.3.3` (MIDI I/O)

完整依赖见 [requirements.txt](requirements.txt)

---

## 🛠️ 开发指南

### 修改训练配置

编辑 `scripts/train_safe.py`:

```python
# LoRA 配置
lora_config = LoraConfig(
    r=16,              # 增大以提升表达能力
    lora_alpha=32,     # 通常为 r 的 2 倍
    lora_dropout=0.1,
)

# 训练配置
training_args = TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
```

### 添加新 Token

1. 在 `prepare_training_data.py` 中定义新 token
2. 在 `train_safe.py` 中添加到 `SPECIAL_TOKENS`
3. 在 `generate_interactive.py` 中同步添加

### 自定义评估

编辑 `quick_evaluation.py`:

```python
# 修改测试提示词
TEST_PROMPTS = [
    "your custom prompt 1",
    "your custom prompt 2",
    ...
]

# 修改样本数量
NUM_SAMPLES = 20
```

---

## 📖 相关资源

- **原始模型**: [slseanwu/MIDI-LLM_Llama-3.2-1B](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B)
- **数据集**: midicaps (基于 Lakh MIDI Dataset)
- **论文**: MIDI-LLM: Expanding MIDI Generation with Text-to-Music Models

---

## 📝 许可证

见 [LICENSE.md](LICENSE.md)

---

## 🙏 致谢

- MIDI-LLM 原始项目团队
- Hugging Face PEFT 团队
- Lakh MIDI Dataset

---

**最后更新**: 2025年12月20日
