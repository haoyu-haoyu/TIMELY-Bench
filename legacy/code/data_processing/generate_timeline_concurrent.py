"""
高并发版本 - 使用 DeepSeek API 生成 Disease Timeline
使用 asyncio 和 aiohttp 实现高并发
"""

import json
import os
import asyncio
import aiohttp
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import random

# 配置
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-1ca70ff9ccfa46cb924aa760b5bdde10')
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/disease_timelines')

# 并发设置
MAX_CONCURRENT = 100  # 超高并发
SAMPLE_SIZE = None    # None = 全量处理


def summarize_vitals(vitals, max_points=8):
    if not vitals:
        return "No vitals"
    summary = []
    step = max(1, len(vitals) // max_points)
    for v in vitals[::step][:max_points]:
        hr = v.get('heart_rate', '-')
        temp = v.get('temperature', '-')
        summary.append(f"HR={hr},T={temp}")
    return ";".join(summary)


def summarize_patterns(patterns, max_patterns=5):
    if not patterns:
        return "None"
    names = [p.get('pattern_name', 'unk') for p in patterns[:max_patterns]]
    return ",".join(names)


def build_prompt(episode):
    vitals = episode.get('timeseries', {}).get('vitals', [])
    patterns = episode.get('reasoning', {}).get('detected_patterns', [])

    return f"""ICU patient early-course analysis. Use only first 24h data; ignore discharge summaries or future outcomes.
Vitals (0-24h): {summarize_vitals(vitals)}
Patterns (0-24h): {summarize_patterns(patterns)}

Output JSON disease timeline:
{{"primary_disease":"sepsis/aki/none","onset_hour":X,"phases":[{{"hour":0,"phase":"admission","severity":0.1}},{{"hour":X,"phase":"onset","severity":0.5}}],"prognosis":"stable/deteriorating"}}

Only valid JSON:"""


async def call_deepseek_async(session, prompt, semaphore):
    """异步调用 DeepSeek API"""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        try:
            async with session.post(DEEPSEEK_BASE_URL, headers=headers, json=payload, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content']
                else:
                    return None
        except Exception as e:
            return None


def parse_timeline(response_text):
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except:
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
        except:
            pass
    return None


async def process_episode_async(session, ep_file, semaphore):
    """异步处理单个 Episode"""
    try:
        with open(ep_file) as f:
            ep = json.load(f)
        
        stay_id = ep.get('stay_id')
        prompt = build_prompt(ep)
        
        response = await call_deepseek_async(session, prompt, semaphore)
        timeline = parse_timeline(response)
        
        return {
            'stay_id': stay_id,
            'disease_timeline': timeline,
            'success': timeline is not None
        }
    except Exception as e:
        return {'stay_id': None, 'disease_timeline': None, 'success': False}


async def main_async():
    print("=" * 70)
    print("Disease Timeline 高并发生成 - DeepSeek API")
    print(f"并发数: {MAX_CONCURRENT}, 采样: {SAMPLE_SIZE}")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有文件
    all_files = list(EPISODES_DIR.glob('*.json'))
    if SAMPLE_SIZE:
        episode_files = random.sample(all_files, min(SAMPLE_SIZE, len(all_files)))
    else:
        episode_files = all_files  # 全量处理
    print(f"处理 Episodes: {len(episode_files):,}")
    
    # 创建信号量限制并发
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # 创建 aiohttp session
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_episode_async(session, f, semaphore) for f in episode_files]
        results = await tqdm_asyncio.gather(*tasks, desc="生成中")
    
    # 统计结果
    success = sum(1 for r in results if r.get('success'))
    print(f"\n完成！成功: {success:,} / {len(results):,}")
    
    # 保存结果
    with open(OUTPUT_DIR / 'disease_timelines_sample.json', 'w') as f:
        json.dump([r for r in results if r.get('disease_timeline')], f, indent=2)
    
    print(f"保存到: {OUTPUT_DIR / 'disease_timelines_sample.json'}")
    
    # 显示样例
    print("\n样例输出:")
    for r in results[:3]:
        if r.get('disease_timeline'):
            print(f"  stay_id={r['stay_id']}: {r['disease_timeline'].get('primary_disease')}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
