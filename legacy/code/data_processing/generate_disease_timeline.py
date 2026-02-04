"""
Generate disease timelines using DeepSeek API.
Est. cost: ~$50 for 74,829 episodes.
"""

import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

# DeepSeek API 配置 (兼容 OpenAI API)
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-1ca70ff9ccfa46cb924aa760b5bdde10')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 路径配置
EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/disease_timelines')

# 初始化客户端
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def summarize_vitals(vitals, max_points=10):
    """摘要生命体征数据"""
    if not vitals:
        return "No vitals data"
    
    summary = []
    step = max(1, len(vitals) // max_points)
    for v in vitals[::step][:max_points]:
        hour = v.get('hour', '?')
        hr = v.get('heart_rate', '-')
        temp = v.get('temperature', '-')
        sbp = v.get('sbp', '-')
        spo2 = v.get('spo2', '-')
        summary.append(f"H{hour}: HR={hr}, T={temp}, BP={sbp}, SpO2={spo2}")
    
    return "; ".join(summary)


def summarize_labs(labs, max_points=5):
    """摘要实验室检查"""
    if not labs:
        return "No labs data"
    
    summary = []
    for l in labs[:max_points]:
        hour = l.get('hour', '?')
        cr = l.get('creatinine', '-')
        lactate = l.get('lactate', '-')
        wbc = l.get('wbc', '-')
        summary.append(f"H{hour}: Cr={cr}, Lac={lactate}, WBC={wbc}")
    
    return "; ".join(summary)


def summarize_patterns(patterns, max_patterns=10):
    """摘要检测到的模式"""
    if not patterns:
        return "No patterns detected"
    
    pattern_list = []
    for p in patterns[:max_patterns]:
        name = p.get('pattern_name', 'unknown')
        hour = p.get('hour', '?')
        pattern_list.append(f"{name}@H{hour}")
    
    return ", ".join(pattern_list)


def build_prompt(episode):
    """构建 LLM 提示"""
    vitals = episode.get('timeseries', {}).get('vitals', [])
    labs = episode.get('timeseries', {}).get('labs', [])
    patterns = episode.get('reasoning', {}).get('detected_patterns', [])

    prompt = f"""Analyze this ICU patient's early disease progression and generate a timeline.
Use only information from the first 24 hours of ICU stay. Ignore discharge summaries or future outcomes.

Vitals summary (0-24h): {summarize_vitals(vitals)}
Labs summary (0-24h): {summarize_labs(labs)}
Detected patterns (0-24h): {summarize_patterns(patterns)}

Generate a disease timeline in JSON format:
{{
  "primary_disease": "sepsis/aki/ards/none",
  "onset_hour": <first hour of disease manifestation>,
  "phases": [
    {{"hour": 0, "phase": "admission", "severity": 0.1}},
    {{"hour": X, "phase": "early_signs", "severity": 0.3}},
    {{"hour": Y, "phase": "disease_confirmed", "severity": 0.7}}
  ],
  "key_events": ["event1 at hour X", "event2 at hour Y"],
  "prognosis": "improving/stable/deteriorating"
}}

Only output valid JSON, no explanation."""

    return prompt


def call_deepseek(prompt, max_retries=3):
    """调用 DeepSeek API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a clinical AI assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def parse_timeline(response_text):
    """解析 LLM 响应为 JSON"""
    if not response_text:
        return None
    
    try:
        # 尝试直接解析
        return json.loads(response_text)
    except:
        # 尝试提取 JSON 块
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except:
            pass
    return None


def process_episode(ep_file):
    """处理单个 Episode"""
    try:
        ep = json.load(open(ep_file))
        stay_id = ep.get('stay_id')
        
        # 构建提示
        prompt = build_prompt(ep)
        
        # 调用 API
        response = call_deepseek(prompt)
        
        # 解析结果
        timeline = parse_timeline(response)
        
        if timeline:
            return {
                'stay_id': stay_id,
                'disease_timeline': timeline,
                'raw_response': response[:500] if response else None
            }
        else:
            return {
                'stay_id': stay_id,
                'disease_timeline': None,
                'error': 'parse_failed'
            }
    except Exception as e:
        return {
            'stay_id': ep_file.stem.split('_')[-1],
            'disease_timeline': None,
            'error': str(e)
        }


def main():
    print("=" * 70)
    print("Disease Timeline 生成 - 使用 DeepSeek API")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 Episode 文件
    episode_files = list(EPISODES_DIR.glob('*.json'))
    print(f"待处理 Episodes: {len(episode_files):,}")
    
    # 估算成本
    est_tokens = len(episode_files) * 3000  # 每条约 3000 tokens
    est_cost = est_tokens / 1_000_000 * 0.42  # DeepSeek V3 合计价格
    print(f"预估成本: ~${est_cost:.2f}")
    
    # 处理
    results = []
    success_count = 0
    
    for ep_file in tqdm(episode_files, desc="生成 Disease Timeline"):
        result = process_episode(ep_file)
        results.append(result)
        
        if result.get('disease_timeline'):
            success_count += 1
        
        # 每 100 条保存一次
        if len(results) % 100 == 0:
            pd_results = [r for r in results if r.get('disease_timeline')]
            with open(OUTPUT_DIR / 'disease_timelines_partial.json', 'w') as f:
                json.dump(pd_results, f, indent=2)
        
        # 速率限制
        time.sleep(0.1)  # 防止超过 API 限制
    
    # 保存最终结果
    with open(OUTPUT_DIR / 'disease_timelines.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n完成！")
    print(f"  成功: {success_count:,}")
    print(f"  失败: {len(results) - success_count:,}")
    print(f"  保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
