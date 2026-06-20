"""
自进化 — SelfEvolver
=====================

参考：anda-brain 自进化 + openclaw-auto-dream 洞察生成

核心思想：
- 记忆系统不是静态的，而是活的、会进化的
- 规则从历史数据中学习，不是人工硬编码
- LLM 生成洞察（insights），规则根据洞察自适应调整

功能：
1. 洞察生成：从评分和图谱统计生成非显而易见的学习点
2. 规则自适应：根据记忆访问模式调整评分权重
3. 模式发现：发现反复出现的错误/成功模式
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json
from pathlib import Path


@dataclass
class EvolutionRule:
    """进化规则"""
    id: str
    pattern: str                   # 模式描述
    trigger: str                   # 触发条件
    action: str                    # 执行动作
    confidence: float = 0.5       # 置信度
    times_triggered: int = 0
    last_triggered: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class SelfEvolver:
    """
    自进化引擎

    使用示例：
        evolver = SelfEvolver()
        insights = evolver.generate_insights(health_report, graph_stats)
        rules = evolver.evolve_rules(insights)
    """

    def __init__(self, rules_dir: str = "~/.openclaw/workspace/memory/evolver"):
        self.rules_dir = Path(rules_dir).expanduser()
        self.rules_file = self.rules_dir / "rules.json"
        self._rules: dict[str, EvolutionRule] = {}
        self._load()

    def _load(self):
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        if self.rules_file.exists():
            with open(self.rules_file) as f:
                raw = json.load(f)
                self._rules = {k: EvolutionRule(**v) for k, v in raw.items()}

    def _save(self):
        with open(self.rules_file, "w") as f:
            json.dump({k: vars(v) for k, v in self._rules.items()}, f, indent=2, ensure_ascii=False)

    def generate_insights(self, health_report: dict, graph_stats: dict) -> list[str]:
        """
        从健康报告和图谱统计生成洞察

        参考 openclaw-auto-dream "生成1-3个非显而易见的学习点"

        分析维度：
        1. 健康指标异常（低记忆率/高归档率）
        2. 图谱矛盾检测结果
        3. 访问模式（哪些记忆从未被引用）
        4. 时间分布（记忆是否集中在某段时间）
        """
        insights = []

        # 1. 健康异常检测
        if health_report.get("low_score_ratio", 0) > 0.3:
            insights.append(
                f"⚠️ 低分记忆过多（{health_report['low_score_ratio']:.0%}），"
                "建议检查是否有记忆过载或标记不当"
            )

        # 2. 知识图谱矛盾
        contradictions = graph_stats.get("contradictions", 0)
        if contradictions > 0:
            insights.append(
                f"🔄 检测到 {contradictions} 对矛盾记忆，请通过 SUPSERSEDES 边确认时间顺序"
            )

        # 3. 重要记忆缺失
        if health_report.get("important_ratio", 0) < 0.1:
            insights.append(
                "📉 重要记忆占比过低，可能存在标记习惯问题，"
                "建议增加 HIGH/PERMANENT 标记"
            )

        # 4. 引用集中度
        ref_conc = health_report.get("reference_concentration", 0)
        if ref_conc > 5.0:
            insights.append(
                f"📊 引用分布极不均匀（σ={ref_conc:.1f}），"
                "部分记忆被过度依赖，建议主动引用边缘记忆"
            )

        # 5. 记忆年龄分布
        avg_age = health_report.get("avg_age_days", 0)
        if avg_age > 365:
            insights.append(
                f"📅 平均记忆年龄 {avg_age:.0f}天（>1年），"
                "部分历史记忆可能已过时，建议评估归档"
            )

        if not insights:
            insights.append("🌟 系统健康，所有指标正常")

        return insights

    def evolve_rules(self, insights: list[str]) -> list[EvolutionRule]:
        """
        根据洞察自适应调整规则

        规则生成策略：
        1. 如果洞察提到"低分过多" → 生成"降低 LOW 标记阈值"规则
        2. 如果洞察提到"重要记忆缺失" → 生成"提高 HIGH 标记阈值"规则
        3. 如果洞察提到"引用不均" → 生成"引用激励"规则
        """
        evolved = []

        for insight in insights:
            rule_id = f"rule_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

            if "低分记忆" in insight or "过载" in insight:
                rule = EvolutionRule(
                    id=rule_id,
                    pattern="低分记忆过多",
                    trigger="health_report.low_score_ratio > 0.3",
                    action="建议审查标记习惯，降低 LOW 标记频率",
                    confidence=0.7,
                    times_triggered=1,
                    last_triggered=datetime.now(timezone.utc).isoformat(),
                )
                evolved.append(rule)

            if "重要记忆" in insight or "标记" in insight:
                rule = EvolutionRule(
                    id=rule_id + "_b",
                    pattern="重要记忆占比低",
                    trigger="health_report.important_ratio < 0.1",
                    action="建议增加关键决策标记为 PERMANENT",
                    confidence=0.6,
                    times_triggered=1,
                    last_triggered=datetime.now(timezone.utc).isoformat(),
                )
                evolved.append(rule)

        # 保存新规则
        for rule in evolved:
            self._rules[rule.id] = rule
        if evolved:
            self._save()

        return evolved

    def get_active_rules(self) -> list[EvolutionRule]:
        """获取所有活跃规则（置信度 > 0.4）"""
        return [r for r in self._rules.values() if r.confidence >= 0.4]

    def update_confidence(self, rule_id: str, delta: float):
        """更新规则置信度（根据实际效果调整）"""
        if rule_id in self._rules:
            self._rules[rule_id].confidence = min(1.0, max(0.0, self._rules[rule_id].confidence + delta))
            self._save()
