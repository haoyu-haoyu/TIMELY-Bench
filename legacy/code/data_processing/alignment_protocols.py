"""
TIMELY-Bench Alignment Protocol Definitions
定义多种时间对齐协议，用于评估不同对齐策略的效果

对齐协议类型:
1. D0 Daily - 同一天的笔记
2. ±6h - 模式发生前后6小时窗口
3. ±12h - 模式发生前后12小时窗口
4. ±24h - 模式发生前后24小时窗口
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class AlignmentProtocol(Enum):
    """对齐协议类型"""
    D0_DAILY = "D0_daily"       # 同一天
    WINDOW_6H = "±6h"           # ±6小时
    WINDOW_12H = "±12h"         # ±12小时
    WINDOW_24H = "±24h"         # ±24小时
    ASYMMETRIC = "asymmetric"   # 非对称窗口（默认：-6h,+2h）


@dataclass
class AlignmentWindow:
    """对齐窗口定义"""
    protocol: AlignmentProtocol
    before_hours: int       # 模式发生前的时间窗口（小时）
    after_hours: int        # 模式发生后的时间窗口（小时）
    description: str        # 协议描述
    use_case: str          # 适用场景


# 预定义的对齐窗口
ALIGNMENT_WINDOWS: Dict[AlignmentProtocol, AlignmentWindow] = {
    AlignmentProtocol.D0_DAILY: AlignmentWindow(
        protocol=AlignmentProtocol.D0_DAILY,
        before_hours=24,
        after_hours=24,
        description="Same calendar day alignment (D0)",
        use_case="Aggregated daily features, daily summaries"
    ),
    AlignmentProtocol.WINDOW_6H: AlignmentWindow(
        protocol=AlignmentProtocol.WINDOW_6H,
        before_hours=6,
        after_hours=6,
        description="±6 hour window around pattern detection",
        use_case="Tight temporal alignment for acute changes"
    ),
    AlignmentProtocol.WINDOW_12H: AlignmentWindow(
        protocol=AlignmentProtocol.WINDOW_12H,
        before_hours=12,
        after_hours=12,
        description="±12 hour window around pattern detection",
        use_case="Medium temporal alignment for shift-based notes"
    ),
    AlignmentProtocol.WINDOW_24H: AlignmentWindow(
        protocol=AlignmentProtocol.WINDOW_24H,
        before_hours=24,
        after_hours=24,
        description="±24 hour window around pattern detection",
        use_case="Broad alignment for comprehensive context"
    ),
    AlignmentProtocol.ASYMMETRIC: AlignmentWindow(
        protocol=AlignmentProtocol.ASYMMETRIC,
        before_hours=6,
        after_hours=2,
        description="Asymmetric window (-6h, +2h) - default",
        use_case="Predictive modeling, causal alignment"
    ),
}


def get_alignment_window(protocol: AlignmentProtocol) -> AlignmentWindow:
    """获取指定协议的对齐窗口"""
    return ALIGNMENT_WINDOWS.get(protocol)


def calculate_time_delta_quality(
    time_delta_hours: float,
    protocol: AlignmentProtocol
) -> str:
    """
    根据时间差和协议计算对齐质量

    Returns:
        exact: 时间差<=1小时
        close: 时间差<=3小时
        moderate: 时间差<=12小时
        distant: 时间差>12小时
    """
    abs_delta = abs(time_delta_hours)

    if abs_delta <= 1:
        return "exact"
    elif abs_delta <= 3:
        return "close"
    elif abs_delta <= 12:
        return "moderate"
    else:
        return "distant"


def is_within_window(
    pattern_hour: float,
    note_hour: float,
    protocol: AlignmentProtocol
) -> bool:
    """检查笔记是否在指定窗口内"""
    window = ALIGNMENT_WINDOWS.get(protocol)
    if window is None:
        return False

    time_delta = note_hour - pattern_hour
    return -window.before_hours <= time_delta <= window.after_hours


def get_protocol_card() -> Dict:
    """生成Alignment Protocol Card（用于文档）"""
    card = {
        "name": "TIMELY-Bench Alignment Protocols",
        "version": "1.0",
        "protocols": []
    }

    for protocol, window in ALIGNMENT_WINDOWS.items():
        card["protocols"].append({
            "id": protocol.value,
            "window_before_hours": window.before_hours,
            "window_after_hours": window.after_hours,
            "description": window.description,
            "use_case": window.use_case
        })

    return card


def print_protocol_summary():
    """打印协议摘要"""
    print("\n" + "=" * 70)
    print("TIMELY-Bench Alignment Protocols")
    print("=" * 70)

    for protocol, window in ALIGNMENT_WINDOWS.items():
        print(f"\n{protocol.value}")
        print(f"   Window: -{window.before_hours}h to +{window.after_hours}h")
        print(f"   Description: {window.description}")
        print(f"   Use case: {window.use_case}")


if __name__ == "__main__":
    import json

    # 打印协议摘要
    print_protocol_summary()

    # 导出Protocol Card
    card = get_protocol_card()

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import ROOT_DIR

    doc_dir = ROOT_DIR / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)
    output_path = doc_dir / 'alignment_protocols.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(card, f, indent=2, ensure_ascii=False)

    print(f"\n\nProtocol Card saved to {output_path}")
