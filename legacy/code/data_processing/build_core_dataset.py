"""
TIMELY-Bench-Core 核心数据集构建器

目标：从74K+患者中筛选2000-5000个高质量episodes

质量标准：
1. data_quality_score >= 0.6 (时序数据完整性)
2. n_patterns >= 20 (临床模式丰富度)
3. has_aligned_spans = True (有临床文本)
4. 疾病标签覆盖平衡 (Sepsis/AKI/ARDS/正常)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from tqdm import tqdm

# ==========================================
# 配置
# ==========================================

ROOT_DIR = Path(__file__).parent
COHORT_FILE = ROOT_DIR / 'merge_output' / 'cohort_final.csv'
TIMESERIES_FILE = ROOT_DIR / 'timeseries.csv'
NOTE_TIME_FILE = ROOT_DIR / 'note_time.csv'
PATTERNS_FILE = ROOT_DIR / 'pattern_detection' / 'detected_patterns_24h.csv'
ALIGNMENT_FILE = ROOT_DIR / 'temporal_alignment' / 'temporal_textual_alignment.csv'  # 关键：alignment数据

OUTPUT_DIR = ROOT_DIR / 'episodes_core'
SAMPLE_DIR = ROOT_DIR / 'episodes_sample'

# 质量阈值
MIN_QUALITY_SCORE = 0.55  # 略微放宽以获取足够样本
MIN_PATTERNS = 15         # 至少15个临床模式
MIN_VITAL_COVERAGE = 0.7  # 至少70%时间点有生命体征

# 目标数量
TARGET_EPISODES = 3000    # 目标3000个高质量episodes


# ==========================================
# 质量评估器
# ==========================================

@dataclass
class EpisodeQuality:
    """Episode质量评估结果"""
    stay_id: int

    # 时序数据质量
    vital_coverage: float       # 生命体征覆盖率
    lab_coverage: float         # 实验室检查覆盖率
    n_timepoints: int           # 时间点数量

    # 模式丰富度
    n_patterns: int             # 检测到的模式数
    n_unique_patterns: int      # 唯一模式类型数
    n_severe_patterns: int      # 严重模式数

    # 文本对齐 (核心：使用alignment数据)
    n_notes: int                # 临床笔记数
    n_alignments: int           # 时序-文本对齐数 (关键指标)
    has_alignment: bool         # 是否有alignment数据
    has_radiology: bool         # 有放射学报告
    has_nursing: bool           # 有护理笔记

    # 标签信息
    has_sepsis: bool
    has_aki: bool
    has_ards: bool
    mortality: int

    # 综合评分
    quality_score: float = 0.0

    def calculate_score(self) -> float:
        """计算综合质量评分 (0-1) - 优先alignment数据"""
        score = 0.0

        # 时序质量 (30%) - 降低权重以给alignment更多空间
        score += 0.15 * min(self.vital_coverage, 1.0)
        score += 0.08 * min(self.lab_coverage, 1.0)
        score += 0.07 * min(self.n_timepoints / 24, 1.0)

        # 模式丰富度 (30%)
        score += 0.12 * min(self.n_patterns / 50, 1.0)
        score += 0.10 * min(self.n_unique_patterns / 10, 1.0)
        score += 0.08 * min(self.n_severe_patterns / 5, 1.0)

        # 时序-文本对齐 (40%) - 关键指标，提高权重
        if self.has_alignment:
            score += 0.20  # 有alignment数据直接加分
            score += 0.10 * min(self.n_alignments / 100, 1.0)  # alignment数量
        score += 0.05 * min(self.n_notes / 5, 1.0)
        score += 0.03 if self.has_radiology else 0
        score += 0.02 if self.has_nursing else 0

        self.quality_score = round(score, 3)
        return self.quality_score


class QualityAnalyzer:
    """质量分析器"""

    def __init__(self):
        self.cohort_df = None
        self.timeseries_df = None
        self.notes_df = None
        self.patterns_df = None
        self.alignment_df = None  # 新增：alignment数据
        self.alignment_patient_set = set()  # 有alignment数据的患者集合

    def load_data(self):
        """加载所有数据"""
        print("Loading data for quality analysis...")

        if COHORT_FILE.exists():
            self.cohort_df = pd.read_csv(COHORT_FILE)
            self.cohort_df['stay_id'] = self.cohort_df['stay_id'].astype(int)
            print(f"   Cohort: {len(self.cohort_df)} patients")

        if TIMESERIES_FILE.exists():
            self.timeseries_df = pd.read_csv(TIMESERIES_FILE)
            self.timeseries_df['stay_id'] = self.timeseries_df['stay_id'].astype(int)
            print(f"   Timeseries: {len(self.timeseries_df)} records")
            print(f"      Unique patients: {self.timeseries_df['stay_id'].nunique()}")

        if NOTE_TIME_FILE.exists():
            self.notes_df = pd.read_csv(NOTE_TIME_FILE)
            if 'stay_id' in self.notes_df.columns:
                self.notes_df['stay_id'] = self.notes_df['stay_id'].astype(int)
            print(f"   Notes: {len(self.notes_df)} notes")
            print(f"      Unique patients: {self.notes_df['stay_id'].nunique()}")

        if PATTERNS_FILE.exists():
            self.patterns_df = pd.read_csv(PATTERNS_FILE)
            self.patterns_df['stay_id'] = self.patterns_df['stay_id'].astype(int)
            print(f"   Patterns: {len(self.patterns_df)} patterns")
            print(f"      Unique patients: {self.patterns_df['stay_id'].nunique()}")

        # 关键：加载alignment数据
        if ALIGNMENT_FILE.exists():
            self.alignment_df = pd.read_csv(ALIGNMENT_FILE)
            self.alignment_df['stay_id'] = self.alignment_df['stay_id'].astype(int)
            self.alignment_patient_set = set(self.alignment_df['stay_id'].unique())
            print(f"   Alignment: {len(self.alignment_df)} alignments")
            print(f"      Unique patients with alignment: {len(self.alignment_patient_set)}")
        else:
            print(f"   Alignment file not found: {ALIGNMENT_FILE}")

    def evaluate_patient(self, stay_id: int) -> Optional[EpisodeQuality]:
        """评估单个患者的质量"""

        # 获取cohort信息
        patient = self.cohort_df[self.cohort_df['stay_id'] == stay_id]
        if len(patient) == 0:
            return None
        patient = patient.iloc[0]

        # 时序数据
        if self.timeseries_df is not None:
            patient_ts = self.timeseries_df[
                (self.timeseries_df['stay_id'] == stay_id) &
                (self.timeseries_df['hour'] < 24)
            ]
            n_timepoints = len(patient_ts)

            # 生命体征覆盖率
            vital_cols = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2']
            if len(patient_ts) > 0 and all(c in patient_ts.columns for c in vital_cols):
                vital_coverage = 1 - patient_ts[vital_cols].isna().mean().mean()
            else:
                vital_coverage = 0.0

            # 实验室覆盖率
            lab_cols = ['creatinine', 'potassium', 'sodium', 'wbc', 'hemoglobin']
            if len(patient_ts) > 0 and all(c in patient_ts.columns for c in lab_cols):
                lab_coverage = 1 - patient_ts[lab_cols].isna().mean().mean()
            else:
                lab_coverage = 0.0
        else:
            n_timepoints = 0
            vital_coverage = 0.0
            lab_coverage = 0.0

        # 模式检测
        if self.patterns_df is not None:
            patient_patterns = self.patterns_df[self.patterns_df['stay_id'] == stay_id]
            n_patterns = len(patient_patterns)
            n_unique_patterns = patient_patterns['pattern_name'].nunique() if len(patient_patterns) > 0 else 0
            n_severe_patterns = len(patient_patterns[patient_patterns['severity'] == 'severe']) if 'severity' in patient_patterns.columns and len(patient_patterns) > 0 else 0
        else:
            n_patterns = 0
            n_unique_patterns = 0
            n_severe_patterns = 0

        # 临床笔记
        if self.notes_df is not None and 'stay_id' in self.notes_df.columns:
            patient_notes = self.notes_df[self.notes_df['stay_id'] == stay_id]
            n_notes = len(patient_notes)

            if 'note_type' in patient_notes.columns:
                note_types = patient_notes['note_type'].str.lower().tolist()
                has_radiology = any('radiol' in str(t) for t in note_types)
                has_nursing = any('nurs' in str(t) for t in note_types)
            elif 'category' in patient_notes.columns:
                note_types = patient_notes['category'].str.lower().tolist()
                has_radiology = any('radiol' in str(t) for t in note_types)
                has_nursing = any('nurs' in str(t) for t in note_types)
            else:
                has_radiology = n_notes > 0  # 假设有笔记就有放射学
                has_nursing = False
        else:
            n_notes = 0
            has_radiology = False
            has_nursing = False

        # 关键：时序-文本对齐数据 (核心指标)
        has_alignment = stay_id in self.alignment_patient_set
        n_alignments = 0
        if has_alignment and self.alignment_df is not None:
            patient_alignments = self.alignment_df[self.alignment_df['stay_id'] == stay_id]
            n_alignments = len(patient_alignments)
            # 从alignment数据补充notes信息
            if 'note_type' in patient_alignments.columns:
                align_note_types = patient_alignments['note_type'].str.lower().tolist()
                has_radiology = has_radiology or any('radiol' in str(t) for t in align_note_types)
                has_nursing = has_nursing or any('nurs' in str(t) for t in align_note_types)
            # 补充n_notes
            if 'note_id' in patient_alignments.columns:
                unique_notes = patient_alignments['note_id'].nunique()
                n_notes = max(n_notes, unique_notes)

        # 标签信息
        has_sepsis = bool(patient.get('has_sepsis', False) or patient.get('has_sepsis_final', False))
        has_aki = bool(patient.get('has_aki', False) or patient.get('has_aki_final', False))
        has_ards = bool(patient.get('has_ards', False))
        mortality = int(patient.get('label_mortality', 0)) if pd.notna(patient.get('label_mortality')) else 0

        quality = EpisodeQuality(
            stay_id=stay_id,
            vital_coverage=round(vital_coverage, 3),
            lab_coverage=round(lab_coverage, 3),
            n_timepoints=n_timepoints,
            n_patterns=n_patterns,
            n_unique_patterns=n_unique_patterns,
            n_severe_patterns=n_severe_patterns,
            n_notes=n_notes,
            n_alignments=n_alignments,  # 新增
            has_alignment=has_alignment,  # 新增
            has_radiology=has_radiology,
            has_nursing=has_nursing,
            has_sepsis=has_sepsis,
            has_aki=has_aki,
            has_ards=has_ards,
            mortality=mortality
        )
        quality.calculate_score()

        return quality

    def analyze_all(self, sample_size: Optional[int] = None,
                     prioritize_alignment: bool = True) -> List[EpisodeQuality]:
        """分析所有或采样患者 - 优先有alignment数据的患者"""

        # 获取有时序数据的患者
        if self.timeseries_df is not None:
            all_stay_ids = set(self.timeseries_df['stay_id'].unique())
        else:
            all_stay_ids = set(self.cohort_df['stay_id'].unique())

        # 关键改进：优先选择有alignment数据的患者
        if prioritize_alignment and self.alignment_patient_set:
            alignment_ids = list(all_stay_ids & self.alignment_patient_set)
            other_ids = list(all_stay_ids - self.alignment_patient_set)

            print(f"\nPatient distribution:")
            print(f"   With alignment data: {len(alignment_ids)}")
            print(f"   Without alignment data: {len(other_ids)}")

            if sample_size:
                # 优先选择有alignment的患者
                n_alignment = min(len(alignment_ids), int(sample_size * 0.9))  # 90%来自alignment
                n_other = min(len(other_ids), sample_size - n_alignment)

                stay_ids = random.sample(alignment_ids, n_alignment) if len(alignment_ids) >= n_alignment else alignment_ids
                if n_other > 0 and other_ids:
                    stay_ids.extend(random.sample(other_ids, min(n_other, len(other_ids))))
            else:
                # 先分析alignment患者，再分析其他
                stay_ids = alignment_ids + other_ids
        else:
            stay_ids = list(all_stay_ids)
            if sample_size and sample_size < len(stay_ids):
                stay_ids = random.sample(stay_ids, sample_size)

        print(f"\nAnalyzing {len(stay_ids)} patients...")

        qualities = []
        for stay_id in tqdm(stay_ids, desc="Evaluating"):
            q = self.evaluate_patient(stay_id)
            if q:
                qualities.append(q)

        return qualities

    def generate_report(self, qualities: List[EpisodeQuality]) -> Dict:
        """生成质量分析报告"""

        if not qualities:
            return {"error": "No quality data"}

        df = pd.DataFrame([vars(q) for q in qualities])

        report = {
            "total_patients": len(df),

            # 质量分数分布
            "quality_score": {
                "mean": round(df['quality_score'].mean(), 3),
                "median": round(df['quality_score'].median(), 3),
                "std": round(df['quality_score'].std(), 3),
                "min": round(df['quality_score'].min(), 3),
                "max": round(df['quality_score'].max(), 3),
                "above_0.5": int((df['quality_score'] >= 0.5).sum()),
                "above_0.55": int((df['quality_score'] >= 0.55).sum()),
                "above_0.6": int((df['quality_score'] >= 0.6).sum()),
                "above_0.7": int((df['quality_score'] >= 0.7).sum()),
            },

            # 时序数据
            "timeseries": {
                "avg_vital_coverage": round(df['vital_coverage'].mean(), 3),
                "avg_lab_coverage": round(df['lab_coverage'].mean(), 3),
                "avg_timepoints": round(df['n_timepoints'].mean(), 1),
            },

            # 模式丰富度
            "patterns": {
                "avg_patterns": round(df['n_patterns'].mean(), 1),
                "avg_unique_patterns": round(df['n_unique_patterns'].mean(), 1),
                "avg_severe_patterns": round(df['n_severe_patterns'].mean(), 1),
                "with_patterns_15+": int((df['n_patterns'] >= 15).sum()),
                "with_patterns_30+": int((df['n_patterns'] >= 30).sum()),
            },

            # 文本覆盖
            "text_coverage": {
                "avg_notes": round(df['n_notes'].mean(), 1),
                "with_notes": int((df['n_notes'] > 0).sum()),
                "with_radiology": int(df['has_radiology'].sum()),
                "with_nursing": int(df['has_nursing'].sum()),
            },

            # 时序-文本对齐 (核心指标)
            "alignment_coverage": {
                "with_alignment": int(df['has_alignment'].sum()),
                "alignment_rate": round(df['has_alignment'].mean(), 3),
                "avg_alignments": round(df['n_alignments'].mean(), 1),
                "with_50+_alignments": int((df['n_alignments'] >= 50).sum()),
            },

            # 疾病分布
            "disease_distribution": {
                "sepsis": int(df['has_sepsis'].sum()),
                "aki": int(df['has_aki'].sum()),
                "ards": int(df['has_ards'].sum()),
                "mortality": int(df['mortality'].sum()),
                "sepsis_rate": round(df['has_sepsis'].mean(), 3),
                "aki_rate": round(df['has_aki'].mean(), 3),
            },

            # 高质量候选 (核心指标：必须有alignment数据)
            "high_quality_candidates": {
                "quality>=0.55_and_patterns>=15": int(
                    ((df['quality_score'] >= 0.55) & (df['n_patterns'] >= 15)).sum()
                ),
                "quality>=0.55_and_patterns>=15_and_alignment": int(
                    ((df['quality_score'] >= 0.55) & (df['n_patterns'] >= 15) & (df['has_alignment'] == True)).sum()
                ),
                "quality>=0.6_and_patterns>=20": int(
                    ((df['quality_score'] >= 0.6) & (df['n_patterns'] >= 20)).sum()
                ),
                "quality>=0.6_and_patterns>=20_and_alignment": int(
                    ((df['quality_score'] >= 0.6) & (df['n_patterns'] >= 20) & (df['has_alignment'] == True)).sum()
                ),
                "quality>=0.6_and_patterns>=20_and_notes>=1": int(
                    ((df['quality_score'] >= 0.6) & (df['n_patterns'] >= 20) & (df['n_notes'] >= 1)).sum()
                ),
                "with_alignment_and_50+_alignments": int(
                    ((df['has_alignment'] == True) & (df['n_alignments'] >= 50)).sum()
                ),
            }
        }

        return report


# ==========================================
# 核心数据集构建器
# ==========================================

class CoreDatasetBuilder:
    """核心数据集构建器"""

    def __init__(self, analyzer: QualityAnalyzer):
        self.analyzer = analyzer
        self.selected_episodes = []

    def select_episodes(self, qualities: List[EpisodeQuality],
                        target_size: int = TARGET_EPISODES,
                        require_alignment: bool = True) -> List[EpisodeQuality]:
        """选择高质量episodes，保持疾病分布平衡

        Args:
            qualities: 质量评估结果列表
            target_size: 目标数量
            require_alignment: 是否强制要求有alignment数据 (核心约束)
        """

        # 核心过滤：优先选择有alignment数据的患者
        if require_alignment:
            alignment_qualities = [q for q in qualities if q.has_alignment]
            no_alignment_qualities = [q for q in qualities if not q.has_alignment]
            print(f"\nAlignment filter:")
            print(f"   With alignment: {len(alignment_qualities)}")
            print(f"   Without alignment: {len(no_alignment_qualities)}")
            # 优先使用有alignment的患者
            sorted_qualities = sorted(alignment_qualities, key=lambda x: x.quality_score, reverse=True)
        else:
            sorted_qualities = sorted(qualities, key=lambda x: x.quality_score, reverse=True)

        # 分层采样策略
        selected = []

        # 目标分布 (按疾病组)
        target_distribution = {
            'sepsis_only': target_size * 0.25,      # 25% 仅Sepsis
            'aki_only': target_size * 0.20,         # 20% 仅AKI
            'sepsis_aki': target_size * 0.15,       # 15% Sepsis+AKI
            'ards': target_size * 0.10,             # 10% ARDS
            'mortality': target_size * 0.10,        # 10% 死亡
            'normal': target_size * 0.20,           # 20% 正常/其他
        }

        groups = defaultdict(list)
        for q in sorted_qualities:
            if q.quality_score < MIN_QUALITY_SCORE or q.n_patterns < MIN_PATTERNS:
                continue

            # 分组
            if q.mortality == 1:
                groups['mortality'].append(q)
            elif q.has_ards:
                groups['ards'].append(q)
            elif q.has_sepsis and q.has_aki:
                groups['sepsis_aki'].append(q)
            elif q.has_sepsis:
                groups['sepsis_only'].append(q)
            elif q.has_aki:
                groups['aki_only'].append(q)
            else:
                groups['normal'].append(q)

        print("\nGroup sizes (filtered by quality + alignment):")
        for group, items in groups.items():
            print(f"   {group}: {len(items)}")

        # 从每组选择
        for group, target in target_distribution.items():
            available = groups[group]
            n_select = min(int(target), len(available))
            selected.extend(available[:n_select])
            print(f"   Selected {n_select} from {group}")

        # 如果还没达到目标，从剩余有alignment的患者中补充
        all_candidates = [q for q in sorted_qualities
                         if q.quality_score >= MIN_QUALITY_SCORE
                         and q.n_patterns >= MIN_PATTERNS
                         and q not in selected]

        remaining = target_size - len(selected)
        if remaining > 0 and all_candidates:
            selected.extend(all_candidates[:remaining])
            print(f"   Supplemented {min(remaining, len(all_candidates))} from remaining candidates")

        self.selected_episodes = selected

        # 打印alignment统计
        n_with_alignment = sum(1 for q in selected if q.has_alignment)
        print(f"\nSelected {len(selected)} episodes")
        print(f"   With alignment: {n_with_alignment} ({n_with_alignment/len(selected)*100:.1f}%)")

        return selected

    def get_selection_summary(self) -> Dict:
        """获取选择摘要"""
        if not self.selected_episodes:
            return {}

        df = pd.DataFrame([vars(q) for q in self.selected_episodes])

        return {
            "total_selected": len(df),
            "quality_score_range": [round(df['quality_score'].min(), 3),
                                   round(df['quality_score'].max(), 3)],
            "avg_quality_score": round(df['quality_score'].mean(), 3),
            "disease_distribution": {
                "sepsis": int(df['has_sepsis'].sum()),
                "aki": int(df['has_aki'].sum()),
                "ards": int(df['has_ards'].sum()),
                "mortality": int(df['mortality'].sum()),
            },
            "avg_patterns": round(df['n_patterns'].mean(), 1),
            "avg_notes": round(df['n_notes'].mean(), 1),
            # 新增：alignment统计 (核心指标)
            "alignment_stats": {
                "with_alignment": int(df['has_alignment'].sum()),
                "alignment_rate": round(df['has_alignment'].mean(), 3),
                "avg_alignments": round(df['n_alignments'].mean(), 1),
                "max_alignments": int(df['n_alignments'].max()),
            },
        }

    def export_selection(self, output_file: Path):
        """导出选中的stay_ids"""
        stay_ids = [q.stay_id for q in self.selected_episodes]

        df = pd.DataFrame([vars(q) for q in self.selected_episodes])
        df.to_csv(output_file, index=False)

        print(f"Exported {len(stay_ids)} episodes to {output_file}")


# ==========================================
# 主函数
# ==========================================

def main():
    print("=" * 70)
    print("TIMELY-Bench-Core Dataset Builder")
    print("=" * 70)

    # 初始化分析器
    analyzer = QualityAnalyzer()
    analyzer.load_data()

    # 分析所有有时序数据的患者
    print("\n" + "=" * 70)
    print("Phase 1: Quality Analysis")
    print("=" * 70)

    # 先采样分析，了解分布
    print("\nSampling 5000 patients for initial analysis...")
    sample_qualities = analyzer.analyze_all(sample_size=5000)

    sample_report = analyzer.generate_report(sample_qualities)

    print("\n📈 Sample Quality Report:")
    print("-" * 50)
    print(f"Total sampled: {sample_report['total_patients']}")
    print(f"\nQuality Score Distribution:")
    qs = sample_report['quality_score']
    print(f"  Mean: {qs['mean']}, Median: {qs['median']}")
    print(f"  >=0.5: {qs['above_0.5']}, >=0.55: {qs['above_0.55']}, >=0.6: {qs['above_0.6']}")

    print(f"\nPattern Richness:")
    pt = sample_report['patterns']
    print(f"  Avg patterns: {pt['avg_patterns']}")
    print(f"  With 15+ patterns: {pt['with_patterns_15+']}")
    print(f"  With 30+ patterns: {pt['with_patterns_30+']}")

    print(f"\nText Coverage:")
    tc = sample_report['text_coverage']
    print(f"  Avg notes: {tc['avg_notes']}")
    print(f"  With notes: {tc['with_notes']}")
    print(f"  With radiology: {tc['with_radiology']}")

    print(f"\nDisease Distribution:")
    dd = sample_report['disease_distribution']
    print(f"  Sepsis: {dd['sepsis']} ({dd['sepsis_rate']*100:.1f}%)")
    print(f"  AKI: {dd['aki']} ({dd['aki_rate']*100:.1f}%)")
    print(f"  ARDS: {dd['ards']}")
    print(f"  Mortality: {dd['mortality']}")

    print(f"\nHigh-Quality Candidates:")
    hq = sample_report['high_quality_candidates']
    for key, count in hq.items():
        print(f"  {key}: {count}")

    # 保存报告
    report_file = ROOT_DIR / 'quality_analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(sample_report, f, indent=2)
    print(f"\nReport saved to {report_file}")

    # 如果样本足够好，进行全量分析并选择
    print("\n" + "=" * 70)
    print("Phase 2: Full Analysis & Selection")
    print("=" * 70)

    # 估算需要分析多少才能获得足够的高质量样本
    high_quality_rate = hq['quality>=0.55_and_patterns>=15'] / sample_report['total_patients']
    estimated_needed = int(TARGET_EPISODES / max(high_quality_rate, 0.1)) + 1000

    print(f"\n📐 Estimated high-quality rate: {high_quality_rate*100:.1f}%")
    print(f"📐 Need to analyze approximately: {estimated_needed} patients")

    # 进行更大规模分析
    analysis_size = min(estimated_needed, 20000)  # 最多分析2万
    print(f"\nAnalyzing {analysis_size} patients...")

    all_qualities = analyzer.analyze_all(sample_size=analysis_size)

    # 构建核心数据集
    builder = CoreDatasetBuilder(analyzer)
    selected = builder.select_episodes(all_qualities, target_size=TARGET_EPISODES, require_alignment=True)

    # 导出选择结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_file = OUTPUT_DIR / 'core_episode_selection.csv'
    builder.export_selection(selection_file)

    # 打印最终摘要
    summary = builder.get_selection_summary()
    print("\n" + "=" * 70)
    print("Final Selection Summary")
    print("=" * 70)
    print(f"Total selected: {summary.get('total_selected', 0)}")
    print(f"Quality score range: {summary.get('quality_score_range', [])}")
    print(f"Average quality score: {summary.get('avg_quality_score', 0)}")
    print(f"Average patterns: {summary.get('avg_patterns', 0)}")
    print(f"Average notes: {summary.get('avg_notes', 0)}")
    print(f"\nDisease distribution: {summary.get('disease_distribution', {})}")

    # 核心指标：Alignment统计
    align_stats = summary.get('alignment_stats', {})
    print(f"\nAlignment Statistics (核心指标):")
    print(f"   With alignment: {align_stats.get('with_alignment', 0)} ({align_stats.get('alignment_rate', 0)*100:.1f}%)")
    print(f"   Average alignments per episode: {align_stats.get('avg_alignments', 0)}")
    print(f"   Max alignments: {align_stats.get('max_alignments', 0)}")

    print("\nCore dataset selection complete!")
    print(f"Selection saved to: {selection_file}")


if __name__ == "__main__":
    main()
