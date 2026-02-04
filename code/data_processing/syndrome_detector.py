"""
综合征检测器 - 基于组合规则 + 文本证据
充分利用现有的 pattern_annotations 和 MedCAT 概念
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class SyndromeDetector:
    """综合征检测器"""
    
    def __init__(self, rules_path: str = None, medcat_df: pd.DataFrame = None, use_condition_labels: bool = False):
        if rules_path and Path(rules_path).exists():
            self.rules = json.load(open(rules_path))
        else:
            self.rules = self._get_default_rules()
        self.medcat_df = medcat_df
        self.use_condition_labels = use_condition_labels
    
    def _get_default_rules(self) -> Dict:
        """默认诊断规则"""
        return {
            "infection_evidence": {
                "text_keywords": [
                    "infection", "sepsis", "septic", "bacteremia", "pneumonia",
                    "antibiotic", "culture positive", "gram positive", "gram negative",
                    "abscess", "cellulitis", "uti", "urinary tract infection"
                ],
                "medcat_concepts": ["has_sepsis", "has_pneumonia", "has_infection", "has_antibiotic"]
            },
            "aki_kdigo": {
                "medcat_concepts": ["has_aki"]
            },
            "ards_simplified": {
                "text_keywords": [
                    "ards",
                    "acute respiratory distress",
                    "bilateral infiltrates",
                    "pulmonary edema",
                    "respiratory failure"
                ],
                "medcat_concepts": ["has_ards"]
            }
        }

    def _get_medcat_evidence(self, stay_id: Optional[int], concepts: List[str]) -> List[str]:
        """从 MedCAT 特征中提取证据"""
        if self.medcat_df is None or not stay_id or not concepts:
            return []
        row = self.medcat_df[self.medcat_df['stay_id'] == stay_id]
        if len(row) == 0:
            return []
        row = row.iloc[0]
        return [concept for concept in concepts if row.get(concept, 0) > 0]
    
    def detect_sirs(self, vitals: List[dict], hour: int) -> Dict:
        """检测 SIRS (≥2 条同时满足)"""
        window_start = max(0, hour - 6)
        window_data = [v for v in vitals if window_start <= v.get('hour', 0) <= hour]
        
        if not window_data:
            return {'detected': False, 'criteria_met': [], 'count': 0}
        
        criteria_met = set()
        for v in window_data:
            temp = v.get('temperature')
            hr = v.get('heart_rate')
            rr = v.get('resp_rate')
            
            if temp is not None and temp > 38.3:
                criteria_met.add('fever')
            if temp is not None and temp < 36.0:
                criteria_met.add('hypothermia')
            if hr is not None and hr > 90:
                criteria_met.add('tachycardia')
            if rr is not None and rr > 20:
                criteria_met.add('tachypnea')
        
        return {
            'detected': len(criteria_met) >= 2,
            'criteria_met': list(criteria_met),
            'count': len(criteria_met),
            'detection_hour': hour,
            'confidence': min(len(criteria_met) / 4, 1.0)
        }
    
    def _keyword_negated(self, text: str, keyword: str) -> bool:
        """Simple negation check within a short window before keyword."""
        kw = re.escape(keyword.lower()).replace(r'\ ', r'\s+')
        pattern = r'(no|without|denies|denied|negative for|rule out)\s+(?:\w+\s+){0,3}' + kw
        return re.search(pattern, text) is not None

    def detect_infection_from_text(self, 
                                    pattern_annotations: List[dict],
                                    stay_id: int = None) -> Dict:
        """从 pattern_annotations 和 MedCAT 检测感染证据"""
        
        infection_keywords = self.rules.get('infection_evidence', {}).get('text_keywords', [])
        text_evidence = []
        
        for ann in pattern_annotations:
            category = str(ann.get('annotation_category', '')).upper()
            if category != 'SUPPORTIVE':
                continue
            aligned_text = str(ann.get('aligned_text', '')).lower()
            annotation_reasoning = str(ann.get('annotation_reasoning', '')).lower()
            combined_text = aligned_text + ' ' + annotation_reasoning
            
            for kw in infection_keywords:
                if kw.lower() in combined_text and not self._keyword_negated(combined_text, kw):
                    text_evidence.append({
                        'keyword': kw,
                        'source': 'aligned_text',
                        'note_type': ann.get('note_type'),
                        'hour': ann.get('note_hour')
                    })
                    break
        
        # 从 MedCAT 概念检测
        medcat_concepts = self.rules.get('infection_evidence', {}).get('medcat_concepts', [])
        medcat_evidence = self._get_medcat_evidence(stay_id, medcat_concepts)
        
        has_evidence = len(text_evidence) > 0 or len(medcat_evidence) > 0
        
        return {
            'detected': has_evidence,
            'text_evidence': text_evidence[:5],
            'medcat_evidence': medcat_evidence,
            'n_text_matches': len(text_evidence),
            'confidence': min((len(text_evidence) + len(medcat_evidence)) / 3, 1.0)
        }
    
    def detect_sepsis(self, episode: dict) -> Dict:
        """检测 Sepsis = SIRS + 感染证据 (优化版：放宽条件提高 Recall)"""
        vitals = episode.get('timeseries', {}).get('vitals', [])
        annotations = episode.get('reasoning', {}).get('pattern_annotations', [])
        stay_id = episode.get('stay_id')
        conditions = episode.get('conditions', []) if self.use_condition_labels else []
        
        # 检测 SIRS (放宽：任意1条SIRS也算部分满足)
        sirs_results = []
        partial_sirs = False
        for hour in range(0, 24, 2):
            sirs = self.detect_sirs(vitals, hour)
            if sirs['detected']:
                sirs_results.append(sirs)
            elif sirs['count'] >= 1:
                partial_sirs = True
        
        has_sirs = len(sirs_results) > 0
        first_sirs_hour = sirs_results[0]['detection_hour'] if sirs_results else None
        best_sirs = max(sirs_results, key=lambda x: x['count']) if sirs_results else None
        
        # 检测感染证据 (放宽：增加更多关键词)
        infection = self.detect_infection_from_text(annotations, stay_id)
        
        # 额外检查：conditions 中是否已有 sepsis
        has_sepsis_condition = self.use_condition_labels and ('sepsis' in conditions)
        
        # 综合判断 (放宽：SIRS + 感染 或 已有sepsis诊断 + 部分SIRS)
        sepsis_detected = (has_sirs and infection['detected']) or \
                          (has_sepsis_condition and (has_sirs or partial_sirs)) or \
                          (has_sirs and has_sepsis_condition)
        
        return {
            'detected': sepsis_detected,
            'sirs': {
                'detected': has_sirs,
                'partial': partial_sirs,
                'first_hour': first_sirs_hour,
                'criteria': best_sirs['criteria_met'] if best_sirs else [],
                'max_count': best_sirs['count'] if best_sirs else 0
            },
            'infection': {
                'detected': infection['detected'],
                'n_evidence': infection['n_text_matches'] + len(infection['medcat_evidence']),
                'sources': infection['text_evidence'][:2]
            },
            'condition_based': has_sepsis_condition,
            'confidence': (0.4 if has_sirs else 0) + (0.3 if infection['detected'] else 0) + (0.3 if has_sepsis_condition else 0)
        }
    
    def detect_aki(self, episode: dict) -> Dict:
        """检测 AKI KDIGO 分期 (优化版：增加 conditions 检查)"""
        labs = episode.get('timeseries', {}).get('labs', [])
        conditions = episode.get('conditions', []) if self.use_condition_labels else []
        stay_id = episode.get('stay_id')
        
        # 检查 conditions 中是否已有 AKI
        has_aki_condition = self.use_condition_labels and ('aki' in conditions)
        
        # 获取肌酐值
        creatinine_values = []
        for l in labs:
            cr = l.get('creatinine')
            if cr is not None:
                creatinine_values.append((l.get('hour', 0), cr))
        
        medcat_concepts = self.rules.get('aki_kdigo', {}).get('medcat_concepts', [])
        medcat_evidence = self._get_medcat_evidence(stay_id, medcat_concepts)
        has_medcat = len(medcat_evidence) > 0

        # 如果有 AKI condition 但缺少肌酐数据，仍然标记为检测到
        if len(creatinine_values) < 2:
            detected = has_aki_condition or has_medcat
            return {
                'detected': detected,
                'stage': 1 if detected else 0,
                'condition_based': has_aki_condition,
                'medcat_evidence': medcat_evidence,
                'reason': 'insufficient_creatinine_data'
            }
        
        # 排序
        creatinine_values.sort(key=lambda x: x[0])
        
        # 基线：前24小时最低值
        early_values = [v for h, v in creatinine_values if h < 24]
        baseline_cr = min(early_values) if early_values else creatinine_values[0][1]
        
        # 检测各阶段
        max_stage = 0
        detection_hour = None
        max_ratio = 1.0
        max_delta = 0.0
        
        for hour, cr in creatinine_values:
            if hour < 6:
                continue
            
            if baseline_cr > 0:
                ratio = cr / baseline_cr
                delta = cr - baseline_cr
                
                max_ratio = max(max_ratio, ratio)
                max_delta = max(max_delta, delta)
                
                # 放宽阈值
                if ratio >= 3.0 or cr >= 4.0:
                    if max_stage < 3:
                        max_stage = 3
                        detection_hour = hour
                elif ratio >= 2.0:
                    if max_stage < 2:
                        max_stage = 2
                        detection_hour = hour
                elif delta >= 0.3 or ratio >= 1.3:  # 放宽：1.5 -> 1.3
                    if max_stage < 1:
                        max_stage = 1
                        detection_hour = hour
        
        # 如果检测到任何上升趋势，结合 condition 判断
        detected = max_stage > 0 or (has_aki_condition and max_ratio >= 1.2) or has_medcat
        
        return {
            'detected': detected,
            'stage': max_stage if max_stage > 0 else (1 if has_aki_condition else 0),
            'baseline_creatinine': round(baseline_cr, 2),
            'max_ratio': round(max_ratio, 2),
            'max_delta': round(max_delta, 2),
            'detection_hour': detection_hour,
            'condition_based': has_aki_condition,
            'medcat_evidence': medcat_evidence,
            'confidence': min(
                (max_stage / 3 if max_stage > 0 else (0.5 if has_aki_condition else 0)) + (0.2 if has_medcat else 0),
                1.0
            )
        }
    
    def detect_ards(self, episode: dict) -> Dict:
        """检测 ARDS (简化标准)"""
        vitals = episode.get('timeseries', {}).get('vitals', [])
        annotations = episode.get('reasoning', {}).get('pattern_annotations', [])
        stay_id = episode.get('stay_id')
        
        # 低氧血症
        hypoxemia_hours = []
        for v in vitals:
            spo2 = v.get('spo2')
            if spo2 is not None and spo2 < 90:
                hypoxemia_hours.append(v.get('hour', 0))
        has_hypoxemia = len(hypoxemia_hours) > 0
        
        # 呼吸衰竭
        resp_failure_hours = []
        for v in vitals:
            rr = v.get('resp_rate')
            if rr is not None and rr > 30:
                resp_failure_hours.append(v.get('hour', 0))
        has_resp_failure = len(resp_failure_hours) > 0
        
        # 文本证据
        ards_keywords = self.rules.get('ards_simplified', {}).get('text_keywords', [])
        text_evidence = []
        for ann in annotations:
            aligned_text = str(ann.get('aligned_text', '')).lower()
            for kw in ards_keywords:
                if kw.lower() in aligned_text:
                    text_evidence.append(kw)
                    break
        
        has_text_evidence = len(text_evidence) > 0

        medcat_concepts = self.rules.get('ards_simplified', {}).get('medcat_concepts', [])
        medcat_evidence = self._get_medcat_evidence(stay_id, medcat_concepts)
        has_medcat = len(medcat_evidence) > 0
        
        # 组合判断: 必须有低氧血症，再结合呼吸衰竭/文本/MedCAT证据
        detected = has_hypoxemia and (has_resp_failure or has_text_evidence or has_medcat)
        
        return {
            'detected': detected,
            'hypoxemia': {
                'detected': has_hypoxemia,
                'n_hours': len(hypoxemia_hours)
            },
            'resp_failure': {
                'detected': has_resp_failure,
                'n_hours': len(resp_failure_hours)
            },
            'text_evidence': text_evidence[:3],
            'medcat_evidence': medcat_evidence,
            'confidence': min(
                (0.4 if has_hypoxemia else 0) +
                (0.3 if has_resp_failure else 0) +
                (0.2 if has_text_evidence else 0) +
                (0.3 if has_medcat else 0),
                1.0
            )
        }
    
    def detect_all(self, episode: dict) -> Dict:
        """检测所有综合征"""
        return {
            'sepsis': self.detect_sepsis(episode),
            'aki': self.detect_aki(episode),
            'ards': self.detect_ards(episode)
        }


def get_accuracy(detected: bool, label: bool) -> str:
    """计算诊断准确性分类"""
    if detected and label:
        return 'TP'
    elif detected and not label:
        return 'FP'
    elif not detected and label:
        return 'FN'
    else:
        return 'TN'
