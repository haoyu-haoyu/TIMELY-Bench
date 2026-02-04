"""
生成 Reasoning Chain - 基于规则自动生成诊断推理链
不使用 LLM，纯代码实现
"""

import json
from pathlib import Path
from tqdm import tqdm

# 路径配置
EPISODES_DIR = Path('/Users/wanghaoyu/Downloads/临床时序 × 文本对齐融合基准/训练基线模型/TIMELY-Bench_Final/episodes/episodes_enhanced')


def generate_sepsis_reasoning(syndrome):
    """生成 Sepsis 诊断推理链"""
    reasoning = {
        "conclusion": syndrome.get('detected', False),
        "evidence": [],
        "chain": [],
        "confidence": syndrome.get('confidence', 0)
    }
    
    sirs = syndrome.get('sirs', {})
    infection = syndrome.get('infection', {})
    
    # SIRS 证据
    if sirs.get('detected'):
        criteria = sirs.get('criteria', [])
        reasoning['evidence'].append(f"SIRS criteria met: {', '.join(criteria)}")
        reasoning['chain'].append(f"SIRS ≥2 criteria ({len(criteria)} found)")
    elif sirs.get('partial'):
        reasoning['evidence'].append("Partial SIRS (1 criterion)")
        reasoning['chain'].append("SIRS partial")
    
    # 感染证据
    if infection.get('detected'):
        n_evidence = infection.get('n_evidence', 0)
        reasoning['evidence'].append(f"Infection evidence found ({n_evidence} sources)")
        reasoning['chain'].append("Infection confirmed from notes")
    
    # 条件诊断
    if syndrome.get('condition_based'):
        reasoning['evidence'].append("Sepsis in medical conditions")
        reasoning['chain'].append("Pre-existing sepsis diagnosis")
    
    # 最终推理
    if reasoning['conclusion']:
        reasoning['chain'].append("SIRS + Infection = Sepsis confirmed")
    else:
        reasoning['chain'].append("Insufficient criteria for Sepsis")
    
    return reasoning


def generate_aki_reasoning(syndrome):
    """生成 AKI 诊断推理链"""
    reasoning = {
        "conclusion": syndrome.get('detected', False),
        "evidence": [],
        "chain": [],
        "confidence": syndrome.get('confidence', 0)
    }
    
    stage = syndrome.get('stage', 0)
    baseline = syndrome.get('baseline_creatinine', 0)
    max_ratio = syndrome.get('max_ratio', 1.0)
    max_delta = syndrome.get('max_delta', 0)
    
    if stage > 0:
        reasoning['evidence'].append(f"KDIGO Stage {stage} criteria met")
        reasoning['evidence'].append(f"Baseline creatinine: {baseline:.2f} mg/dL")
        
        if max_ratio >= 1.5:
            reasoning['evidence'].append(f"Creatinine ratio: {max_ratio:.2f}x baseline")
            reasoning['chain'].append(f"Creatinine increased {max_ratio:.1f}x")
        
        if max_delta >= 0.3:
            reasoning['evidence'].append(f"Creatinine delta: +{max_delta:.2f} mg/dL")
            reasoning['chain'].append(f"Creatinine increased by {max_delta:.2f}")
        
        reasoning['chain'].append(f"AKI Stage {stage} confirmed by KDIGO")
    
    if syndrome.get('condition_based'):
        reasoning['evidence'].append("AKI in medical conditions")
        reasoning['chain'].append("Pre-existing AKI diagnosis")
    
    if not reasoning['conclusion']:
        reasoning['chain'].append("No significant creatinine change detected")
    
    return reasoning


def generate_ards_reasoning(syndrome):
    """生成 ARDS 诊断推理链"""
    reasoning = {
        "conclusion": syndrome.get('detected', False),
        "evidence": [],
        "chain": [],
        "confidence": syndrome.get('confidence', 0)
    }
    
    hypoxemia = syndrome.get('hypoxemia', {})
    resp_failure = syndrome.get('resp_failure', {})
    text_evidence = syndrome.get('text_evidence', [])
    
    if hypoxemia.get('detected'):
        n_hours = hypoxemia.get('n_hours', 0)
        reasoning['evidence'].append(f"Hypoxemia detected (SpO2 <90% for {n_hours} hours)")
        reasoning['chain'].append("Hypoxemia present")
    
    if resp_failure.get('detected'):
        n_hours = resp_failure.get('n_hours', 0)
        reasoning['evidence'].append(f"Respiratory failure (RR >30 for {n_hours} hours)")
        reasoning['chain'].append("Respiratory distress")
    
    if text_evidence:
        reasoning['evidence'].append(f"Text mentions: {', '.join(text_evidence[:3])}")
        reasoning['chain'].append("ARDS mentioned in notes")
    
    if reasoning['conclusion']:
        reasoning['chain'].append("Hypoxemia + respiratory signs = ARDS criteria")
    else:
        reasoning['chain'].append("Insufficient criteria for ARDS")
    
    return reasoning


def process_episode(ep_file):
    """处理单个 Episode"""
    ep = json.load(open(ep_file))
    
    syndrome = ep.get('reasoning', {}).get('syndrome_detection', {})
    
    reasoning_chain = {
        "sepsis": generate_sepsis_reasoning(syndrome.get('sepsis', {})),
        "aki": generate_aki_reasoning(syndrome.get('aki', {})),
        "ards": generate_ards_reasoning(syndrome.get('ards', {}))
    }
    
    # 更新 Episode
    ep['reasoning']['reasoning_chain'] = reasoning_chain
    
    # 保存
    with open(ep_file, 'w') as f:
        json.dump(ep, f, indent=2, ensure_ascii=False)
    
    return True


def main():
    print("=" * 70)
    print("Reasoning Chain 生成 - 规则方法")
    print("=" * 70)
    
    episode_files = list(EPISODES_DIR.glob('*.json'))
    print(f"待处理 Episodes: {len(episode_files):,}")
    
    success = 0
    for ep_file in tqdm(episode_files, desc="生成 Reasoning Chain"):
        try:
            if process_episode(ep_file):
                success += 1
        except Exception as e:
            continue
    
    print(f"\n完成！成功: {success:,}")


if __name__ == "__main__":
    main()
