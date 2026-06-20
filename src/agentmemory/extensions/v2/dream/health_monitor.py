"""
健康监控 — HealthMonitor
=========================

参考：openclaw-auto-dream 健康仪表盘 (5项指标)

5项健康指标：
1. 记忆总量
2. S/A级记忆占比（重要记忆率）
3. 平均年龄
4. 低分记忆占比（需要关注）
5. 引用集中度
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HealthReport:
    """健康报告"""
    total_memories: int
    important_ratio: float           # S/A级占比
    avg_age_days: float
    low_score_ratio: float          # D/F级占比
    reference_concentration: float   # 引用标准差
    archive_count: int
    grade: str                      # 综合评级 A/B/C/D/F

    def summary(self) -> str:
        lines = [
            f"记忆总量: {self.total_memories}",
            f"重要记忆率: {self.important_ratio:.1%}",
            f"平均年龄: {self.avg_age_days:.0f}天",
            f"低分记忆率: {self.low_score_ratio:.1%}",
            f"引用集中度: {self.reference_concentration:.2f}",
            f"归档数量: {self.archive_count}",
            f"综合评级: {self.grade}",
        ]
        return "\n".join(lines)


class HealthMonitor:
    """
    健康监控器

    使用示例：
        monitor = HealthMonitor()
        report = monitor.evaluate(
            total_memories=100,
            important_ratio=0.35,
            avg_age_days=45,
            low_score_ratio=0.1,
            reference_concentration=2.5,
            archive_count=5,
        )
        print(report.summary())
    """

    def evaluate(
        self,
        total_memories: int,
        important_ratio: float,
        avg_age_days: float,
        low_score_ratio: float,
        reference_concentration: float,
        archive_count: int = 0,
    ) -> HealthReport:
        """
        综合评估健康度，返回评级

        评级标准：
        - A: 重要记忆率 >= 30%，低分率 < 15%，引用分布均匀
        - B: 重要记忆率 >= 20%，低分率 < 25%
        - C: 重要记忆率 >= 10%，低分率 < 40%
        - D: 重要记忆率 >= 5%
        - F: 其他
        """
        # 综合评分
        score = 0.0

        # 重要记忆率贡献（40%权重）
        if important_ratio >= 0.30:
            score += 0.40
        elif important_ratio >= 0.20:
            score += 0.30
        elif important_ratio >= 0.10:
            score += 0.20
        elif important_ratio >= 0.05:
            score += 0.10

        # 低分率贡献（30%权重，越低越好）
        if low_score_ratio < 0.10:
            score += 0.30
        elif low_score_ratio < 0.20:
            score += 0.20
        elif low_score_ratio < 0.30:
            score += 0.10

        # 引用集中度（15%权重，越低越均匀）
        if reference_concentration < 3.0:
            score += 0.15
        elif reference_concentration < 5.0:
            score += 0.10
        elif reference_concentration < 8.0:
            score += 0.05

        # 活跃度（15%权重）
        if avg_age_days < 180:
            score += 0.15
        elif avg_age_days < 365:
            score += 0.10
        elif avg_age_days < 730:
            score += 0.05

        # 评级
        if score >= 0.85:
            grade = "A"
        elif score >= 0.65:
            grade = "B"
        elif score >= 0.45:
            grade = "C"
        elif score >= 0.25:
            grade = "D"
        else:
            grade = "F"

        return HealthReport(
            total_memories=total_memories,
            important_ratio=important_ratio,
            avg_age_days=avg_age_days,
            low_score_ratio=low_score_ratio,
            reference_concentration=reference_concentration,
            archive_count=archive_count,
            grade=grade,
        )
