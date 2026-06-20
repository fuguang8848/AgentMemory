"""
梦境周期 — DreamCycle
=======================

参考：
- LeoYeAI/openclaw-auto-dream v4.0 (⭐556) — 三相：Collect → Consolidate → Evaluate
- ldclabs/anda-brain (⭐68) — 神经符号记忆 + 知识图谱自进化

DreamNet 架构：
  ┌───────────┐   ┌──────────────┐   ┌────────────────────┐
  │  COLLECT   │→ │ CONSOLIDATE  │→  │      EVALUATE       │
  │  扫描日志  │   │ 路由到层级   │   │ 重要性评分          │
  │  提取洞察  │   │ 语义去重     │   │ 遗忘曲线评估        │
  │  检测标记  │   │ 建立图谱关系 │   │ 生成洞察报告        │
  └───────────┘   └──────────────┘   └────────────────────┘
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .importance_scorer import ImportanceScorer, EntryScore
from .forgetting_curve import ForgettingCurve
from .knowledge_graphger import KnowledgeGrapher, NodeType, EdgeType
from .self_evolver import SelfEvolver
from .health_monitor import HealthMonitor, HealthReport
from .lucid_generator import LucidDreamGenerator, LucidDream, Inspiration

logger = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """梦境周期执行结果"""
    success: bool
    phase: str
    entries_collected: int = 0
    entries_consolidated: int = 0
    entries_archived: int = 0
    entries_processed: int = 0
    lucid_dreams_generated: int = 0
    lucid_inspirations_count: int = 0
    insights: list[str] = field(default_factory=list)
    health_report: Optional[HealthReport] = None
    graph_stats: dict = field(default_factory=dict)
    error: str = ""

    def summary(self) -> str:
        status = "✅" if self.success else "❌"
        return (
            f"{status} DreamCycle [{self.phase}]\n"
            f"  收集: {self.entries_collected} | "
            f"整合: {self.entries_consolidated} | "
            f"归档: {self.entries_archived}\n"
            f"  清醒梦: {self.lucid_dreams_generated} | "
            f"灵感: {self.lucid_inspirations_count} | "
            f"洞察: {len(self.insights)}\n"
            f"  健康: {self.health_report.grade if self.health_report else 'N/A'}"
        )


@dataclass
class CollectedEntry:
    """收集阶段提取的单条记忆"""
    id: str
    content: str
    marker: str = "NORMAL"    # PERMANENT / HIGH / PIN / NORMAL / LOW
    layer: str = "long_term" # working / episodic / long_term / procedural / index
    tags: list[str] = field(default_factory=list)
    source_file: str = ""
    extracted_people: list[str] = field(default_factory=list)
    extracted_projects: list[str] = field(default_factory=list)
    created_at: str = ""


class DreamNet:
    """
    DreamNet — AI 梦境记忆系统 orchestrator

    五层记忆架构 + 三相梦境周期 + 遗忘曲线 + 知识图谱 + 自进化

    使用示例：
        dream = DreamNet(workspace_dir="~/.openclaw/workspace")
        result = dream.run_dream_cycle()
        print(result.summary())
    """

    MARKER_PATTERNS = {
        "PERMANENT": re.compile(r"⚠️\s*PERMANENT|PERMANENT", re.IGNORECASE),
        "HIGH": re.compile(r"🔥\s*HIGH|HIGH\s*PRIORITY", re.IGNORECASE),
        "PIN": re.compile(r"📌\s*PIN|PIN", re.IGNORECASE),
        "LOW": re.compile(r"💤\s*LOW|LOW\s*PRIORITY", re.IGNORECASE),
    }

    def __init__(
        self,
        workspace_dir: str = "~/.openclaw/workspace",
        memory_dir: str = "~/.openclaw/workspace/memory",
        scan_days: int = 7,
        scorer: Optional[ImportanceScorer] = None,
        forgetting: Optional[ForgettingCurve] = None,
        grapher: Optional[KnowledgeGrapher] = None,
        evolver: Optional[SelfEvolver] = None,
        monitor: Optional[HealthMonitor] = None,
        lucid: Optional[LucidDreamGenerator] = None,
    ):
        self.workspace_dir = Path(workspace_dir).expanduser()
        self.memory_dir = Path(memory_dir).expanduser()
        self.scan_days = scan_days

        self.scorer = scorer or ImportanceScorer()
        self.forgetting = forgetting or ForgettingCurve(str(self.memory_dir / "archive"))
        self.grapher = grapher or KnowledgeGrapher(str(self.memory_dir / "graph"))
        self.evolver = evolver or SelfEvolver(str(self.memory_dir / "evolver"))
        self.monitor = monitor or HealthMonitor()
        self.lucid = lucid or LucidDreamGenerator(str(self.memory_dir))

        self.grapher.load()

        self.output_dir = self.memory_dir / "dream"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._entries: list[CollectedEntry] = []

    def run_dream_cycle(self) -> DreamResult:
        """执行完整梦境周期（三相）"""
        logger.info("☽ DreamCycle 开始")

        try:
            result = DreamResult(success=True, phase="collect")
            self._entries = self._collect_phase()
            result.entries_collected = len(self._entries)
            logger.info(f"  [Collect] 收集到 {result.entries_collected} 条")

            result.phase = "consolidate"
            self._consolidate_phase()
            result.entries_consolidated = len(self._entries)

            result.phase = "evaluate"
            self._evaluate_phase(result)

            # ── Phase 4: Lucid Dreams（清醒梦生成）─
            result.phase = "lucidity"
            lucid_dreams = self.lucid.generate(graph_stats=self.grapher.stats())
            result.lucid_dreams_generated = len(lucid_dreams)
            inspirations = self.lucid.get_unread_inspirations()
            result.lucid_inspirations_count = len(inspirations)
            logger.info(f"  [Lucid] 生成了 {len(lucid_dreams)} 个清醒梦，{len(inspirations)} 条灵感待推送")

            result.insights = self.evolver.generate_insights(
                health_report={
                    "total": result.entries_processed,
                    "important_ratio": result.health_report.important_ratio if result.health_report else 0,
                    "avg_age_days": result.health_report.avg_age_days if result.health_report else 0,
                    "low_score_ratio": result.health_report.low_score_ratio if result.health_report else 0,
                    "reference_concentration": result.health_report.reference_concentration if result.health_report else 0,
                },
                graph_stats=self.grapher.stats(),
            )

            self.evolver.evolve_rules(result.insights)
            self.grapher.save()
            self._write_dream_report(result)

            logger.info(f"  [Insights] {len(result.insights)} 条")
            logger.info(f"  [Health] {result.health_report.grade if result.health_report else 'N/A'}")
            logger.info("☾ DreamCycle 完成")

            return result

        except Exception as e:
            logger.exception("DreamCycle 失败")
            return DreamResult(success=False, phase="collect", error=str(e))

    def _collect_phase(self) -> list[CollectedEntry]:
        """Phase 1: 扫描日志，检测标记，提取记忆"""
        entries = []
        oc_log_dir = Path.home() / ".openclaw" / "logs"

        if not oc_log_dir.exists():
            return entries

        for log_file in sorted(oc_log_dir.glob("*.log")):
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - mtime).total_seconds() / 86400
            if age_days > self.scan_days:
                continue
            try:
                content = log_file.read_text(encoding="utf-8", errors="replace")
                entries.extend(self._extract_entries(content, str(log_file)))
            except Exception as e:
                logger.warning(f"Failed: {log_file}: {e}")

        return entries

    def _extract_entries(self, text: str, source: str) -> list[CollectedEntry]:
        entries = []
        current_id = 0

        for line in text.split("\n"):
            marker = self._detect_marker(line)
            if not line.strip():
                continue

            if marker != "NORMAL":
                current_id += 1
                entries.append(CollectedEntry(
                    id=f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{current_id:04d}",
                    content=line.strip(),
                    marker=marker,
                    source_file=source,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))

        return entries

    def _detect_marker(self, line: str) -> str:
        for marker, pattern in self.MARKER_PATTERNS.items():
            if pattern.search(line):
                return marker
        return "NORMAL"

    def _consolidate_phase(self):
        """Phase 2: 路由到层级，建立知识图谱"""
        for entry in self._entries:
            entry.layer = self._route_to_layer(entry)
            node = self.grapher.add_node(
                node_type=self._layer_to_node_type(entry.layer),
                label=entry.content[:80],
                content=entry.content,
                properties={
                    "marker": entry.marker,
                    "layer": entry.layer,
                    "source": entry.source_file,
                }
            )
            for person in entry.extracted_people:
                pn = self.grapher.add_node(NodeType.CONCEPT, person, properties={"type": "person"})
                self.grapher.add_edge(node.id, pn.id, EdgeType.LINKED_TO)
            for proj in entry.extracted_projects:
                pn = self.grapher.add_node(NodeType.CONCEPT, proj, properties={"type": "project"})
                self.grapher.add_edge(node.id, pn.id, EdgeType.PART_OF)

    def _route_to_layer(self, entry: CollectedEntry) -> str:
        c = entry.content.lower()
        if any(kw in c for kw in ["工作流", "workflow", "步骤", "step", "如何", "how to"]):
            return "procedural"
        if any(kw in c for kw in ["项目", "project", "决定", "decision", "事件", "event"]):
            return "episodic"
        return "long_term"

    def _layer_to_node_type(self, layer: str) -> NodeType:
        return {"episodic": NodeType.EPISODE, "long_term": NodeType.FACT,
                "procedural": NodeType.PROCEDURE, "working": NodeType.FACT,
                "index": NodeType.CONCEPT}.get(layer, NodeType.FACT)

    def _evaluate_phase(self, result: DreamResult):
        """Phase 3: 评分 + 遗忘评估 + 健康指标"""
        scored = []
        archived = 0

        for entry in self._entries:
            se = {
                "id": entry.id, "content": entry.content, "marker": entry.marker,
                "created_at": entry.created_at, "reference_count": 1,
                "has_narrative": len(entry.content) > 50,
                "has_people": bool(entry.extracted_people),
                "has_project": bool(entry.extracted_projects), "tags": entry.tags,
            }
            score = self.scorer.score(se)
            scored.append(score)

            fr = self.forgetting.evaluate(se)
            if fr["action"] == "archive":
                self.forgetting.archive({**se, "score": score.final})
                archived += 1

        result.entries_processed = len(scored)
        result.entries_archived = archived

        metrics = self.scorer.get_health_metrics(scored)
        arch_stats = self.forgetting.get_archive_stats()

        result.health_report = self.monitor.evaluate(
            total_memories=metrics["total"],
            important_ratio=metrics["important_ratio"],
            avg_age_days=metrics["avg_age_days"],
            low_score_ratio=metrics["low_score_ratio"],
            reference_concentration=metrics["reference_concentration"],
            archive_count=arch_stats["total_archived"],
        )
        result.graph_stats = self.grapher.stats()

    def _write_dream_report(self, result: DreamResult):
        report_file = self.output_dir / f"dream_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}.json"
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": result.success,
            "entries_collected": result.entries_collected,
            "entries_archived": result.entries_archived,
            "insights": result.insights,
            "health": ({
                "grade": result.health_report.grade,
                "total": result.health_report.total_memories,
                "important_ratio": result.health_report.important_ratio,
            } if result.health_report else None),
            "graph_stats": result.graph_stats,
        }
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"  [Report] {report_file}")
