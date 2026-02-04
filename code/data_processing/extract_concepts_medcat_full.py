"""
使用完整 MedCAT 模型进行概念提取（24h 窗口）

输出两类产物：
1) note 级概念明细（逐条笔记的 CUI/类型/时间）
2) stay 级聚合特征（概念数、类型计数等）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
import os
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# MedCAT import
try:
    from medcat.cat import CAT
    from medcat.cdb import CDB
    from medcat.vocab import Vocab
    MEDCAT_AVAILABLE = True
except ImportError:
    MEDCAT_AVAILABLE = False
    print("警告: MedCAT 未安装，使用关键词匹配备选方案")

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'medcat_full'
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / 'models' / 'medcat'

# 固定的语义类型映射（避免高维稀疏）
SEMANTIC_TYPE_MAPPING = {
    'T047': 'disease',      # Disease or Syndrome
    'T033': 'finding',      # Finding
    'T184': 'symptom',      # Sign or Symptom
    'T121': 'drug',         # Pharmacologic Substance
    'T061': 'procedure',    # Therapeutic or Preventive Procedure
    'T023': 'body_part',    # Body Part
    'T046': 'pathologic',   # Pathologic Function
    'T037': 'injury',       # Injury or Poisoning
}

NEGATION_MARKERS = ('neg', 'no', 'denied', 'absent', 'without')
HISTORICAL_MARKERS = ('histor', 'past', 'prior')

NOTE_FIELDS = [
    'stay_id', 'note_id', 'note_type', 'note_category', 'chart_hour',
    'cui', 'name', 'type_ids', 'context_similarity', 'start', 'end',
    'negation', 'temporality', 'experiencer',
]


def resolve_model_path(explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        return Path(explicit_path)
    env_path = os.environ.get('MEDCAT_MODEL_PATH')
    if env_path:
        return Path(env_path)
    if DEFAULT_MODEL_DIR.exists():
        # 优先选择 zip，其次是目录
        zip_candidates = list(DEFAULT_MODEL_DIR.glob('*.zip'))
        if zip_candidates:
            return zip_candidates[0]
        dir_candidates = [c for c in DEFAULT_MODEL_DIR.iterdir() if c.is_dir()]
        if dir_candidates:
            return dir_candidates[0]
    return None


def load_medcat_model(model_path: Path):
    """加载 MedCAT 模型"""
    if not MEDCAT_AVAILABLE:
        return None
    
    print("加载 MedCAT 模型...")
    
    try:
        cat = CAT.load_model_pack(str(model_path))
        print("MedCAT 模型加载成功!")
        return cat
    except Exception as e:
        print(f"MedCAT 模型加载失败: {e}")
        return None


def extract_concepts_medcat(text: str, cat) -> list:
    """使用 MedCAT 提取医学概念"""
    if cat is None:
        return []
    
    try:
        doc = cat.get_entities(text)
        concepts = []
        for ent_id, ent_data in doc['entities'].items():
            meta = ent_data.get('meta_anns', {}) or {}
            negation = meta.get('Negation', {}).get('value') if isinstance(meta, dict) else None
            temporality = meta.get('Temporality', {}).get('value') if isinstance(meta, dict) else None
            experiencer = meta.get('Experiencer', {}).get('value') if isinstance(meta, dict) else None

            concepts.append({
                'cui': ent_data['cui'],
                'name': ent_data['pretty_name'],
                'type': ent_data.get('type_ids', []),
                'context_similarity': ent_data.get('context_similarity', 0),
                'start': ent_data.get('start'),
                'end': ent_data.get('end'),
                'negation': negation,
                'temporality': temporality,
                'experiencer': experiencer,
            })
        return concepts
    except Exception as e:
        return []


def extract_concepts_keywords(text: str) -> list:
    """关键词匹配备选方案"""
    keywords = {
        'sepsis': 'C0036690',
        'pneumonia': 'C0032285',
        'acute kidney injury': 'C0022660',
        'respiratory failure': 'C0035229',
        'heart failure': 'C0018801',
        'fever': 'C0015967',
        'hypotension': 'C0020649',
        'tachycardia': 'C0039231',
        'hypoxia': 'C0242184',
        'shock': 'C0036974',
        'infection': 'C0009450',
        'diabetes': 'C0011849',
        'hypertension': 'C0020538',
        'anemia': 'C0002871',
        'edema': 'C0013604'
    }
    
    text_lower = text.lower()
    concepts = []
    for keyword, cui in keywords.items():
        if keyword in text_lower:
            concepts.append({
                'cui': cui,
                'name': keyword,
                'type': ['keyword_match'],
                'context_similarity': 0.5
            })
    return concepts


def _is_negated(value) -> bool:
    if value is None:
        return False
    val = str(value).strip().lower()
    if not val or val == 'nan':
        return False
    return any(marker in val for marker in NEGATION_MARKERS)


def _is_historical(value) -> bool:
    if value is None:
        return False
    val = str(value).strip().lower()
    if not val or val == 'nan':
        return False
    return any(marker in val for marker in HISTORICAL_MARKERS)


def aggregate_features(entities: List[Dict]) -> Dict:
    """将实体聚合为固定特征集合（区分否定/历史）"""
    if not entities:
        return {
            'n_concepts': 0,
            'n_unique_cuis': 0,
            'avg_similarity': 0.0,
            'n_concepts_neg': 0,
            'n_unique_cuis_neg': 0,
            'avg_similarity_neg': 0.0,
        }

    pos_entities = []
    neg_entities = []
    for e in entities:
        if _is_negated(e.get('negation')) or _is_historical(e.get('temporality')):
            neg_entities.append(e)
        else:
            pos_entities.append(e)

    def build_stats(items, suffix):
        cuis = [e.get('cui') for e in items if e.get('cui')]
        type_ids = [tid for e in items for tid in (e.get('type') or [])]
        sim_vals = [e.get('context_similarity', 0) for e in items]
        stats = {
            f'n_concepts{suffix}': len(items),
            f'n_unique_cuis{suffix}': len(set(cuis)),
            f'avg_similarity{suffix}': float(sum(sim_vals) / max(len(sim_vals), 1)) if items else 0.0,
        }
        for tid, name in SEMANTIC_TYPE_MAPPING.items():
            stats[f'type_{name}{suffix}'] = type_ids.count(tid)
        return stats

    features = {}
    features.update(build_stats(pos_entities, ''))
    features.update(build_stats(neg_entities, '_neg'))
    return features


def process_episodes(
    sample_size: Optional[int],
    window_hours: int,
    model_path: Optional[str],
    output_dir: Path,
    max_notes: int,
    max_chars: int,
    min_text_len: int,
):
    """处理 Episode 并提取概念（24h 窗口）"""
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = resolve_model_path(model_path)
    if not resolved_model or not resolved_model.exists():
        raise FileNotFoundError(
            "未找到 MedCAT 模型包。请设置 MEDCAT_MODEL_PATH 或将模型放到 models/medcat/ 下。"
        )

    cat = load_medcat_model(resolved_model)

    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    if sample_size:
        episode_files = episode_files[:sample_size]
    print(f"处理 {len(episode_files)} 个 episodes...")

    note_output_path = output_dir / f'medcat_note_concepts_{window_hours}h.csv'
    agg_output_path = output_dir / f'medcat_features_{window_hours}h.csv'

    agg_rows = []

    with open(note_output_path, 'w', newline='') as note_f:
        writer = csv.DictWriter(note_f, fieldnames=NOTE_FIELDS)
        writer.writeheader()

        for ep_file in tqdm(episode_files, desc="MedCAT 抽取"):
            try:
                with open(ep_file) as f:
                    ep = json.load(f)

                stay_id = ep.get('stay_id')
                if stay_id is None:
                    stay_id = int(ep_file.stem.split('_')[-1])

                clinical = ep.get('clinical_text', {})
                notes = clinical.get('notes', [])
                if not isinstance(notes, list):
                    notes = []
                n_notes_total = len(notes)
                n_notes_used = 0

                all_entities = []

                note_iter = notes if max_notes <= 0 else notes[:max_notes]
                for note in note_iter:
                    if not isinstance(note, dict):
                        continue
                    chart_hour = note.get('chart_hour')
                    try:
                        chart_hour = float(chart_hour)
                    except (TypeError, ValueError):
                        continue
                    if chart_hour < 0 or chart_hour >= window_hours:
                        continue

                    text = note.get('text_full') or note.get('text_relevant') or note.get('text', '')
                    if not text or len(text.strip()) < min_text_len:
                        continue

                    n_notes_used += 1
                    text = text[:max_chars]

                    if cat:
                        entities = extract_concepts_medcat(text, cat)
                    else:
                        entities = extract_concepts_keywords(text)

                    for ent in entities:
                        type_ids = ent.get('type') or []
                        writer.writerow({
                            'stay_id': stay_id,
                            'note_id': note.get('note_id'),
                            'note_type': note.get('note_type'),
                            'note_category': note.get('note_category'),
                            'chart_hour': chart_hour,
                            'cui': ent.get('cui'),
                            'name': ent.get('name'),
                            'type_ids': '|'.join(type_ids) if isinstance(type_ids, list) else str(type_ids),
                            'context_similarity': ent.get('context_similarity', 0),
                            'start': ent.get('start'),
                            'end': ent.get('end'),
                            'negation': ent.get('negation'),
                            'temporality': ent.get('temporality'),
                            'experiencer': ent.get('experiencer'),
                        })

                    all_entities.extend(entities)

                features = aggregate_features(all_entities)
                features.update({
                    'stay_id': stay_id,
                    'window_hours': window_hours,
                    'n_notes_total': n_notes_total,
                    'n_notes_used': n_notes_used,
                })
                agg_rows.append(features)

            except Exception:
                continue

    df = pd.DataFrame(agg_rows).fillna(0)
    df.to_csv(agg_output_path, index=False)

    print(f"\n完成！")
    print(f"  Note 级输出: {note_output_path}")
    print(f"  聚合特征输出: {agg_output_path}")
    if not df.empty:
        print(f"  平均概念数: {df['n_concepts'].mean():.1f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0, help='处理的 episode 数量 (0=全部)')
    parser.add_argument('--window-hours', type=int, default=24, help='笔记窗口（小时）')
    parser.add_argument('--model-path', type=str, default=None, help='MedCAT 模型路径（zip 或目录）')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR), help='输出目录')
    parser.add_argument('--max-notes', type=int, default=0, help='每个 episode 最大处理笔记数 (0=全部)')
    parser.add_argument('--max-chars', type=int, default=10000, help='每条笔记最大文本长度')
    parser.add_argument('--min-text-len', type=int, default=20, help='最短文本长度（过滤过短文本）')
    args = parser.parse_args()
    
    # 如果 sample=0，处理全部
    sample_size = args.sample if args.sample > 0 else None
    process_episodes(
        sample_size=sample_size,
        window_hours=args.window_hours,
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        max_notes=args.max_notes,
        max_chars=args.max_chars,
        min_text_len=args.min_text_len,
    )
