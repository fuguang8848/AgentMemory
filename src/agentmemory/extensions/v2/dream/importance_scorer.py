"""
重要性评分器 — ImportanceScorer
===================================

参考：openclaw-auto-dream v4.0 评分算法
公式：importance = (base_weight × recency × reference_boost) / 8.0

评分维度：
- base_weight：基础权重（由标记决定：PERMANENT/HIGH/NORMAL/LOW）
- recency_factor：时间衰减（180天半衰期）
- reference_boost：引用次数对数加成
- completeness：记忆完整性（是否有完整叙事）
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import math


@dataclass
class EntryScore:
    """单项评分结果"""
    entry_id: str
    total_score: float
    base_weight: float      # 0.5~4.0
    recency_score: float    # 0.1~1.0
    reference_boost: float  # 1.0~+
    completeness: float     # 0.0~1.0
    final: float            # 归一化到 0.0~1.0
    breakdown: dict = field(default_factory=dict)

    def is_important(self) -> bool:
        return self.final >= 0.5

    def should_archive(self) -> bool:
        """是否应该归档（极低分数+超过90天）"""
        return self.final < 0.15 and self.recency_score < 0.5

    def grade(self) -> str:
        """评分等级"""
        if self.final >= 0.8: return "S"
        if self.final >= 0.65: return "A"
        if self.final >= 0.5: return "B"
        if self.final >= 0.35: return "C"
        if self.final >= 0.15: return "D"
        return "F"


@dataclass
class MarkerWeight:
    """标记权重表"""
    PERMANENT: float = 4.0   # 永久记忆
    HIGH: float = 2.5         # 高优先级
    PIN: float = 2.0          # 固定
    NORMAL: float = 1.0      # 普通
    LOW: float = 0.5          # 低优先级
    ARCHIVE: float = 0.1      # 归档候选


class ImportanceScorer:
    """
    重要性评分器

    使用示例：
        scorer = ImportanceScorer()
        score = scorer.score(entry={
            "id": "mem_001",
            "content": "关于项目X的决定...",
            "marker": "HIGH",
            "created_at": "2026-06-01T10:00:00Z",
            "reference_count": 5,
            "has_narrative": True,
            "has_people": True,
            "has_project": True,
        })
        print(f"评分: {score.grade()} ({score.final:.2f})")
    """

    # 半衰期（天）
    HALF_LIFE_DAYS = 180

    # 参考计数对数底的底数
    REFERENCE_LOG_BASE = 2

    def __init__(self):
        self.weights = MarkerWeight()

    def score(self, entry: dict) -> EntryScore:
        """
        对单条记忆进行评分

        Args:
            entry: 记忆条目，包含以下字段：
                - id: 记忆唯一ID
                - content: 记忆内容
                - marker: 标记（PERMANENT/HIGH/PIN/NORMAL/LOW/ARCHIVE）
                - created_at: 创建时间 (ISO格式)
                - reference_count: 被引用次数
                - has_narrative: 是否有叙事
                - has_people: 是否涉及人物
                - has_project: 是否涉及项目
                - tags: 标签列表
        """
        entry_id = entry.get("id", "unknown")

        # 1. 基础权重
        marker = entry.get("marker", "NORMAL").upper()
        base_weight = getattr(self.weights, marker, 1.0)

        # 2. 时间衰减（recency_factor）
        created_at = entry.get("created_at", "")
        recency_score = self._calc_recency(created_at)

        # 3. 引用次数加成
        ref_count = entry.get("reference_count", 0)
        reference_boost = self._calc_reference_boost(ref_count)

        # 4. 完整性加成（叙事 + 人物 + 项目 多维度）
        completeness = self._calc_completeness(entry)

        # 5. 综合评分
        raw = base_weight * recency_score * reference_boost
        total_score = raw
        final = min(raw / 8.0, 1.0)  # 归一化到 [0, 1]

        # 6. 完整性加权
        final = final * (0.6 + 0.4 * completeness)

        breakdown = {
            "marker": marker,
            "days_old": self._days_since(created_at),
            "reference_count": ref_count,
        }

        return EntryScore(
            entry_id=entry_id,
            total_score=total_score,
            base_weight=base_weight,
            recency_score=recency_score,
            reference_boost=reference_boost,
            completeness=completeness,
            final=final,
            breakdown=breakdown,
        )

    def _calc_recency(self, created_at: str) -> float:
        """计算时间衰减因子"""
        if not created_at:
            return 0.5  # 无时间戳，默认中等
        days = self._days_since(created_at)
        # Ebbinghaus 遗忘曲线近似：指数衰减
        return max(0.1, 1.0 - (days / self.HALF_LIFE_DAYS))

    def _days_since(self, iso_time: str) -> float:
        """计算距离今天的天数"""
        try:
            dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - dt).total_seconds() / 86400
        except Exception:
            return 0.0

    def _calc_reference_boost(self, count: int) -> float:
        """引用次数对数加成（边际递减）"""
        if count <= 0:
            return 1.0
        # log₂(n+1)，引用次数越多，加成越少
        return 1.0 + math.log(count + 1, self.REFERENCE_LOG_BASE)

    def _calc_completeness(self, entry: dict) -> float:
        """计算记忆完整性"""
        factors = [
            bool(entry.get("content")),           # 有内容
            bool(entry.get("has_narrative")),     # 有叙事
            bool(entry.get("has_people")),        # 有涉及人物
            bool(entry.get("has_project")),       # 有涉及项目
            bool(entry.get("tags")),              # 有标签
            bool(entry.get("created_at")),        # 有时间戳
        ]
        return sum(factors) / len(factors)

    def batch_score(self, entries: list) -> list[EntryScore]:
        """批量评分"""
        return [self.score(e) for e in entries]

    def get_health_metrics(self, scores: list[EntryScore]) -> dict:
        """
        从评分列表计算健康指标

        参考 openclaw-auto-dream 健康仪表盘指标：
        1. 记忆总量
        2. S/A级记忆占比（重要记忆率）
        3. 平均年龄
        4. 低分记忆占比（需要关注）
        5. 引用集中度（是否有记忆被过度引用）
        """
        if not scores:
            return {
                "total": 0,
                "important_ratio": 0.0,
                "avg_age_days": 0.0,
                "low_score_ratio": 0.0,
                "reference_concentration": 0.0,
            }

        total = len(scores)
        important = sum(1 for s in scores if s.is_important())
        low_score = sum(1 for s in scores if s.final < 0.2)
        avg_age = sum(s.breakdown.get("days_old", 0) for s in scores) / total

        # 引用集中度：标准差，越大说明分布越不均匀
        refs = [s.breakdown.get("reference_count", 0) for s in scores]
        avg_ref = sum(refs) / total if total else 0
        variance = sum((r - avg_ref) ** 2 for r in refs) / total if total > 1 else 0
        ref_std = math.sqrt(variance)

        return {
            "total": total,
            "important_ratio": important / total,
            "avg_age_days": avg_age,
            "low_score_ratio": low_score / total,
            "reference_concentration": ref_std,
        }
