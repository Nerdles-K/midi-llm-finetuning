"""
基于 MIDI 的直接评估（不转换为音频）
使用 mido 分析 MIDI 文件特征
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mido
from collections import Counter, defaultdict

# =============================================================================
# 配置
# =============================================================================

EVALUATION_DIR = "../results/evaluation_results_20251219_231707"

# =============================================================================
# MIDI 特征提取
# =============================================================================

def extract_midi_features(midi_path):
    """从 MIDI 文件提取音乐特征 (使用 mido)"""
    
    try:
        midi_file = mido.MidiFile(str(midi_path))
        
        features = {}
        
        # 1. 总时长（秒）
        features['duration'] = midi_file.length
        
        # 2. 音符统计
        notes = []
        velocities = []
        note_times = []
        current_time = 0
        active_notes = defaultdict(list)  # track -> list of (start_time, pitch)
        
        for i, track in enumerate(midi_file.tracks):
            track_time = 0
            for msg in track:
                track_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
                    velocities.append(msg.velocity)
                    note_times.append(track_time)
                    active_notes[i].append((track_time, msg.note))
        
        features['num_notes'] = len(notes)
        features['note_density'] = len(notes) / max(features['duration'], 0.1)
        
        # 3. 音高范围
        if notes:
            features['pitch_range'] = max(notes) - min(notes)
            features['avg_pitch'] = np.mean(notes)
            features['pitch_std'] = np.std(notes)
        else:
            features['pitch_range'] = 0
            features['avg_pitch'] = 0
            features['pitch_std'] = 0
        
        # 4. 动态范围（velocity）
        if velocities:
            features['avg_velocity'] = np.mean(velocities)
            features['velocity_std'] = np.std(velocities)
        else:
            features['avg_velocity'] = 0
            features['velocity_std'] = 0
        
        # 5. 节奏特征
        if len(note_times) > 1:
            inter_onset_intervals = np.diff(note_times)
            # Convert from MIDI ticks to seconds
            inter_onset_intervals = inter_onset_intervals / midi_file.ticks_per_beat
            features['avg_ioi'] = np.mean(inter_onset_intervals)
            features['ioi_std'] = np.std(inter_onset_intervals)
        else:
            features['avg_ioi'] = 0
            features['ioi_std'] = 0
        
        # 6. 轨道数量
        features['num_instruments'] = len(midi_file.tracks)
        
        # 7. 音符时长估计（简化）
        features['avg_note_duration'] = features['duration'] / max(features['num_notes'], 1)
        features['note_duration_std'] = 0  # mido 不直接提供，设为 0
        
        # 8. 速度估计
        tempo = 500000  # 默认 120 BPM
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
        features['avg_tempo'] = mido.tempo2bpm(tempo)
        
        return features
        
    except Exception as e:
        print(f"  ✗ Error analyzing {midi_path.name}: {e}")
        return None

def compute_midi_features():
    """计算两个模型的 MIDI 特征"""
    
    print("="*70)
    print("Step 1: Extracting MIDI Features")
    print("="*70)
    
    eval_dir = Path(EVALUATION_DIR)
    
    all_features = {"original": [], "fine_tuned": []}
    
    for model_type in ["original", "fine_tuned"]:
        midi_dir = eval_dir / model_type
        midi_files = sorted(midi_dir.glob("*.mid"))
        
        print(f"\n{model_type.upper()} Model:")
        
        for midi_file in tqdm(midi_files, desc=f"Analyzing {model_type}"):
            features = extract_midi_features(midi_file)
            if features:
                all_features[model_type].append(features)
    
    return all_features

# =============================================================================
# 特征对比
# =============================================================================

def compare_features(features_dict):
    """对比两个模型的特征"""
    
    print("\n" + "="*70)
    print("MIDI Features Comparison")
    print("="*70)
    
    feature_names = [
        'duration', 'num_notes', 'note_density',
        'pitch_range', 'avg_pitch', 'pitch_std',
        'avg_velocity', 'velocity_std',
        'avg_ioi', 'ioi_std',
        'num_instruments',
        'avg_note_duration', 'note_duration_std',
        'avg_tempo'
    ]
    
    comparison = {}
    
    for feature_name in feature_names:
        orig_values = [f[feature_name] for f in features_dict["original"]]
        fine_values = [f[feature_name] for f in features_dict["fine_tuned"]]
        
        orig_mean = np.mean(orig_values)
        fine_mean = np.mean(fine_values)
        
        comparison[feature_name] = {
            "original_mean": orig_mean,
            "fine_tuned_mean": fine_mean,
            "difference": fine_mean - orig_mean,
            "percent_change": ((fine_mean - orig_mean) / max(orig_mean, 0.001)) * 100
        }
        
        print(f"\n{feature_name}:")
        print(f"  Original:     {orig_mean:.4f}")
        print(f"  Fine-tuned:   {fine_mean:.4f}")
        print(f"  Difference:   {fine_mean - orig_mean:+.4f} ({comparison[feature_name]['percent_change']:+.2f}%)")
    
    print("\n" + "="*70)
    
    return comparison

# =============================================================================
# 文本对齐评估（基于启发式）
# =============================================================================

def compute_text_midi_alignment(features_dict):
    """基于 MIDI 特征计算文本对齐得分"""
    
    print("\n" + "="*70)
    print("Step 2: Computing Text-MIDI Alignment (Heuristic)")
    print("="*70)
    
    # 读取提示词
    eval_dir = Path(EVALUATION_DIR)
    with open(eval_dir / "evaluation_report.json", 'r') as f:
        report = json.load(f)
    prompts = report["prompts"]
    
    results = {"original": [], "fine_tuned": []}
    
    for model_type in ["original", "fine_tuned"]:
        print(f"\n{model_type.upper()} Model:")
        
        features_list = features_dict[model_type]
        
        for i, (features, prompt) in enumerate(zip(features_list, prompts)):
            score = 0.5  # 基础分
            
            prompt_lower = prompt.lower()
            
            # 情绪/能量匹配
            # 高能量关键词
            if any(word in prompt_lower for word in ["energetic", "upbeat", "fast", "intense", "triumphant"]):
                if features['avg_velocity'] > 80:
                    score += 0.15
                if features['note_density'] > 5:
                    score += 0.10
                if features['avg_tempo'] > 120:
                    score += 0.10
            
            # 低能量关键词
            if any(word in prompt_lower for word in ["sad", "gentle", "slow", "relaxing", "melancholic", "ambient"]):
                if features['avg_velocity'] < 70:
                    score += 0.15
                if features['note_density'] < 5:
                    score += 0.10
                if features['avg_tempo'] < 100:
                    score += 0.10
            
            # 速度匹配
            if "fast" in prompt_lower and features['avg_tempo'] > 120:
                score += 0.10
            if "slow" in prompt_lower and features['avg_tempo'] < 100:
                score += 0.10
            
            # 音高范围
            if any(word in prompt_lower for word in ["piano", "guitar", "violin"]):
                if features['pitch_range'] > 24:  # 至少 2 个八度
                    score += 0.10
            
            # 确保分数在 0-1 范围内
            score = max(0.0, min(1.0, score))
            
            results[model_type].append(score)
    
    # 统计
    original_avg = np.mean(results["original"])
    finetuned_avg = np.mean(results["fine_tuned"])
    improvement = finetuned_avg - original_avg
    
    print("\n" + "="*70)
    print("Text-MIDI Alignment Summary:")
    print("="*70)
    print(f"Original Model:   {original_avg:.4f}")
    print(f"Fine-tuned Model: {finetuned_avg:.4f}")
    print(f"Improvement:      {improvement:+.4f} ({(improvement/max(original_avg,0.001))*100:+.2f}%)")
    print("="*70)
    
    return {
        "original_avg": original_avg,
        "fine_tuned_avg": finetuned_avg,
        "improvement": improvement,
        "original_scores": results["original"],
        "fine_tuned_scores": results["fine_tuned"]
    }

# =============================================================================
# 保存结果
# =============================================================================

def save_metrics(feature_comparison, alignment_results):
    """保存评估指标到 JSON"""
    
    metrics = {
        "midi_feature_comparison": feature_comparison,
        "text_midi_alignment": alignment_results,
        "note": "MIDI-based evaluation (no audio conversion needed)"
    }
    
    eval_dir = Path(EVALUATION_DIR)
    output_file = eval_dir / "midi_metrics.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_file}")
    
    return output_file

# =============================================================================
# 主程序
# =============================================================================

def main():
    """运行完整的评估流程"""
    
    print("\n" + "="*70)
    print("MIDI-LLM Direct MIDI Evaluation")
    print("="*70)
    print("\nUsing:")
    print("  - mido for MIDI feature extraction")
    print("  - Heuristic text-MIDI alignment")
    print("  - No audio conversion or external dependencies needed!")
    print("="*70)
    
    # Step 1: 提取 MIDI 特征
    features_dict = compute_midi_features()
    
    if not features_dict["original"] or not features_dict["fine_tuned"]:
        print("\n✗ No features extracted. Aborting.")
        return
    
    # Step 2: 特征对比
    feature_comparison = compare_features(features_dict)
    
    # Step 3: 文本对齐
    alignment_results = compute_text_midi_alignment(features_dict)
    
    # Step 4: 保存结果
    output_file = save_metrics(feature_comparison, alignment_results)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    print("\nKey Findings:")
    print(f"  - Analyzed {len(features_dict['original'])} samples per model")
    print(f"  - Text-MIDI alignment (Original): {alignment_results['original_avg']:.4f}")
    print(f"  - Text-MIDI alignment (Fine-tuned): {alignment_results['fine_tuned_avg']:.4f}")
    print(f"  - Improvement: {alignment_results['improvement']:+.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
