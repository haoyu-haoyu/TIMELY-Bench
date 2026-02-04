"""
将 Disease Timeline 合并到 Episode 文件中
"""

import json
from pathlib import Path
from tqdm import tqdm

EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
TIMELINE_FILE = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/disease_timelines/disease_timelines_sample.json')


def main():
    print("=" * 70)
    print("合并 Disease Timeline 到 Episodes")
    print("=" * 70)
    
    # 加载 timelines
    timelines = json.load(open(TIMELINE_FILE))
    timeline_dict = {d['stay_id']: d['disease_timeline'] for d in timelines if d.get('disease_timeline')}
    print(f"加载 Timeline: {len(timeline_dict):,}")
    
    # 更新 Episodes
    episode_files = list(EPISODES_DIR.glob('*.json'))
    print(f"待更新 Episodes: {len(episode_files):,}")
    
    updated = 0
    for ep_file in tqdm(episode_files, desc="合并中"):
        try:
            ep = json.load(open(ep_file))
            stay_id = ep.get('stay_id')
            
            if stay_id in timeline_dict:
                if 'reasoning' not in ep:
                    ep['reasoning'] = {}
                ep['reasoning']['disease_timeline'] = timeline_dict[stay_id]
                
                with open(ep_file, 'w') as f:
                    json.dump(ep, f, indent=2, ensure_ascii=False)
                updated += 1
        except Exception as e:
            continue
    
    print(f"\n完成！更新: {updated:,}")


if __name__ == "__main__":
    main()
