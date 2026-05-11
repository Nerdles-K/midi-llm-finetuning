"""
数据准备脚本：从 MIDI 文件提取 tokens 并与 caption 配对

步骤：
1. 读取 train.json（包含 caption 和 MIDI 文件路径）
2. 从 lmd_full/ 读取 MIDI 文件
3. 转换为 tokens，并添加 velocity 和 chord 信息
4. 生成训练数据：{"text": caption, "midi_tokens": tokens_str}
"""

import json
import os
import random
from pathlib import Path
from tqdm import tqdm

print("="*70)
print("MIDI-LLM 数据准备脚本 (含 Velocity + Chord + CFG)")
print("="*70)

# 检查依赖
print("\n检查依赖...")
try:
    from anticipation.convert import midi_to_events
    import mido
    print("✓ anticipation 已安装")
    print("✓ mido 已安装")
except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    print("\n请先安装依赖:")
    print("  pip install mido")
    exit(1)

# =========================================================================
# 配置
# =========================================================================

INPUT_FILE = "../data/train.json"
OUTPUT_FILE = "../data/train_with_tokens.json"
MIDI_BASE_DIR = "../lmd_full"
MAX_SAMPLES = 1000  # 先测试 1000 个样本

# CFG 配置
CFG_TEXT_DROPOUT = 0.15  # 15% 的样本将文本替换为 [UNCOND]

# Velocity 量化（将 0-127 映射到 0-15）
def quantize_velocity(velocity):
    """将 MIDI velocity (0-127) 量化到 16 个级别 (0-15)"""
    return min(15, velocity // 8)

# =========================================================================
# Velocity 和 Chord 提取
# =========================================================================

def extract_velocity_from_midi(midi_path):
    """从 MIDI 文件提取每个音符的 velocity"""
    try:
        mid = mido.MidiFile(midi_path)
        velocities = []
        
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    vel_level = quantize_velocity(msg.velocity)
                    velocities.append(vel_level)
        
        return velocities
    except:
        return []

def get_chord_tokens(metadata):
    """从元数据中获取和弦 tokens"""
    # train.json 中已经有 all_chords 字段
    chords = metadata.get('all_chords', [])
    if not chords:
        return []
    
    # 返回唯一的和弦列表
    unique_chords = []
    for chord in chords:
        if chord and chord not in unique_chords:
            unique_chords.append(chord)
    
    return unique_chords[:10]  # 最多取前 10 个和弦

# =========================================================================
# Token 增强
# =========================================================================

def enhance_tokens_with_velocity_and_chords(tokens_str, velocities, chords):
    """在 tokens 中插入 velocity 和 chord 信息"""
    tokens = tokens_str.split()
    enhanced = []
    
    # 在开头添加和弦信息
    if chords:
        for chord in chords:
            enhanced.append(f"<{chord}>")
    
    # 为每个音符添加 velocity token
    velocity_idx = 0
    for i, token in enumerate(tokens):
        # 假设每隔一定间隔是一个音符（简化处理）
        # 实际应该解析 token 类型，这里每 3 个 token 插入一个 velocity
        if i % 3 == 0 and velocity_idx < len(velocities):
            enhanced.append(f"<vel_{velocities[velocity_idx]}>")
            velocity_idx += 1
        
        enhanced.append(token)
    
    return " ".join(enhanced)

# Token 格式化
def format_tokens(tokens):
    """将 token 列表格式化为字符串"""
    return " ".join([f"<{t}>" for t in tokens])

# =========================================================================
# 主处理流程
# =========================================================================

def process_data():
    """处理数据：读取 MIDI，提取 tokens，与 caption 配对"""
    
    print(f"\n读取元数据: {INPUT_FILE}")
    
    # 统计
    total_samples = 0
    success_count = 0
    error_count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        # 先统计总行数（用于进度条）
        print("统计样本数...")
        total_lines = sum(1 for _ in open(INPUT_FILE, 'r', encoding='utf-8'))
        if MAX_SAMPLES:
            total_lines = min(total_lines, MAX_SAMPLES)
        
        print(f"将处理 {total_lines:,} 个样本\n")
        
        # 重新打开文件处理
        f_in.seek(0)
        
        for line_num, line in enumerate(tqdm(f_in, total=total_lines, desc="处理中"), 1):
            if MAX_SAMPLES and line_num > MAX_SAMPLES:
                break
            
            try:
                # 解析元数据
                metadata = json.loads(line)
                
                # 获取字段
                caption = metadata.get('caption', '')
                midi_path = metadata.get('location', '')
                
                if not caption or not midi_path:
                    error_count += 1
                    continue
                
                # 应用 CFG dropout
                if random.random() < CFG_TEXT_DROPOUT:
                    caption = "[UNCOND]"
                
                # 构建完整 MIDI 文件路径
                full_midi_path = Path(midi_path)
                
                if not full_midi_path.exists():
                    error_count += 1
                    continue
                
                # 从 MIDI 文件提取 tokens
                try:
                    tokens = midi_to_events(str(full_midi_path))
                    tokens_str = format_tokens(tokens)
                    
                    # 提取 velocity 信息
                    velocities = extract_velocity_from_midi(str(full_midi_path))
                    
                    # 获取和弦信息
                    chords = get_chord_tokens(metadata)
                    
                    # 增强 tokens（添加 velocity 和 chord）
                    if velocities or chords:
                        tokens_str = enhance_tokens_with_velocity_and_chords(
                            tokens_str, velocities, chords
                        )
                    
                    # 创建训练样本
                    training_sample = {
                        'text': caption,
                        'midi_tokens': tokens_str
                    }
                    
                    # 写入输出文件
                    f_out.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                    
                    success_count += 1
                    
                except Exception as e:
                    # MIDI 转换失败
                    error_count += 1
                    if line_num <= 10:  # 只显示前 10 个错误
                        tqdm.write(f"  ✗ 样本 {line_num}: MIDI 转换失败 ({midi_path}): {e}")
                    continue
                
            except json.JSONDecodeError:
                error_count += 1
                continue
            except Exception as e:
                error_count += 1
                if line_num <= 10:
                    tqdm.write(f"  ✗ 样本 {line_num}: {e}")
                continue
    
    # 输出统计
    print(f"\n{'='*70}")
    print("处理完成！")
    print(f"{'='*70}")
    print(f"成功: {success_count:,} 个样本")
    print(f"失败: {error_count:,} 个样本")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"{'='*70}\n")
    
    # 显示第一个样本
    if success_count > 0:
        print("第一个样本预览:")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            first_sample = json.loads(f.readline())
            print(f"  Text: {first_sample['text'][:100]}...")
            tokens = first_sample['midi_tokens']
            print(f"  MIDI tokens (前 300 字符): {tokens[:300]}...")
            
            # 检查新 token
            has_vel = '<vel_' in tokens
            has_chord = any(c in tokens for c in ['<C>', '<Dm>', '<G>', '<Am>'])
            has_uncond = '[UNCOND]' in first_sample['text']
            
            print(f"\n  新 Token 统计:")
            print(f"    包含 Velocity tokens: {'✓' if has_vel else '✗'}")
            print(f"    包含 Chord tokens: {'✓' if has_chord else '✗'}")
            print(f"    CFG dropout 示例: {'✓' if has_uncond else '✗ (查看更多样本)'}")

if __name__ == "__main__":
    try:
        process_data()
    except KeyboardInterrupt:
        print("\n\n中断！部分数据已保存到:", OUTPUT_FILE)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
