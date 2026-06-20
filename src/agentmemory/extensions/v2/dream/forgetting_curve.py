"""
遗忘曲线 — ForgettingCurve
===========================

参考：Ebbinghaus 遗忘曲线 + openclaw-auto-dream 遗忘策略

核心思想：
- 记忆不会突然消失，而是逐渐衰减
- 高重要性记忆衰减慢，低重要性记忆衰减快
- 超过90天且分数极低的记忆应被归档（而非删除）

策略：
1. 活跃记忆（>0.5分）：保持原样
2. 边缘记忆（0.15~0.5分）：每30天评估一次
3. 归档候选（<0.15分 + >90天）：移动到 archive/ 而非删除
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json
from pathlib import Path


@dataclass
class ForgetPolicy:
    """遗忘策略配置"""
    active_threshold: float = 0.5    # 高于这个分数，保持活跃
    archive_threshold: float = 0.15  # 低于这个分数，归档候选
    archive_days: int = 90           # 超过多少天开始考虑归档
    eval_interval_days: int = 30     # 边缘记忆评估间隔


@dataclass
class MemoryEntry:
    """可遗忘的记忆条目"""
    id: str
    content: str
    score: float                     # 当前重要性分数
    created_at: str                 # 创建时间
    last_accessed: str = ""         # 最后访问时间
    last_scored: str = ""           # 最后评分时间
    is_archived: bool = False
    archive_reason: str = ""
    tags: list[str] = field(default_factory=list)


class ForgettingCurve:
    """
    遗忘曲线管理器

    使用示例：
        fc = ForgettingCurve()
        policy = fc.evaluate(entry={
            "id": "mem_001",
            "content": "项目X决策...",
            "score": 0.45,
            "created_at": "2026-01-01T00:00:00Z"
        })
        if policy.action == "archive":
            fc.archive(entry_id)
    """

    def __init__(self, archive_dir: str = "~/.openclaw/workspace/memory/archive"):
        self.archive_dir = Path(archive_dir).expanduser()
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.policy = ForgetPolicy()
        self._archive_index_file = self.archive_dir / ".archive_index.json"
        self._archive_index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        if self._archive_index_file.exists():
            with open(self._archive_index_file) as f:
                self._archive_index = json.load(f)

    def _save_index(self):
        with open(self._archive_index_file, "w") as f:
            json.dump(self._archive_index, f, indent=2)

    def _days_since(self, iso_time: str) -> float:
        try:
            dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - dt).total_seconds() / 86400
        except Exception:
            return 0.0

    def evaluate(self, entry: dict) -> dict:
        """
        评估遗忘策略

        Returns:
            action: "keep" | "monitor" | "archive"
            reason: str
            next_eval_days: int or None
        """
        score = entry.get("score", 0.5)
        created_at = entry.get("created_at", "")
        days = self._days_since(created_at)
        is_archived = entry.get("is_archived", False)

        p = self.policy

        # 已归档的不再处理
        if is_archived:
            return {"action": "archived", "reason": "Already archived", "next_eval_days": None}

        # 1. 高分记忆 → 保持活跃
        if score >= p.active_threshold:
            return {"action": "keep", "reason": f"Score {score:.2f} >= {p.active_threshold}", "next_eval_days": None}

        # 2. 超90天 + 极低分 → 归档候选
        if days >= p.archive_days and score < p.archive_threshold:
            return {
                "action": "archive",
                "reason": f"Days={days:.0f} >= {p.archive_days} and score={score:.2f} < {p.archive_threshold}",
                "next_eval_days": None
            }

        # 3. 边缘记忆 → 继续观察
        return {
            "action": "monitor",
            "reason": f"Edge case: score={score:.2f}, days={days:.0f}",
            "next_eval_days": p.eval_interval_days
        }

    def archive(self, entry: dict) -> str:
        """
        归档记忆（移动到 archive/ 目录，不删除）
        返回归档文件路径
        """
        entry_id = entry["id"]
        archive_file = self.archive_dir / f"{entry_id}.json"

        archived_entry = {
            **entry,
            "is_archived": True,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(archive_file, "w") as f:
            json.dump(archived_entry, f, indent=2, ensure_ascii=False)

        # 更新索引
        self._archive_index[entry_id] = {
            "file": str(archive_file),
            "archived_at": archived_entry["archived_at"],
            "original_score": entry.get("score", 0),
            "reason": "forgetting_curve",
        }
        self._save_index()

        return str(archive_file)

    def restore(self, entry_id: str) -> Optional[dict]:
        """恢复归档的记忆"""
        if entry_id not in self._archive_index:
            return None

        info = self._archive_index[entry_id]
        archive_file = Path(info["file"])

        if not archive_file.exists():
            return None

        with open(archive_file) as f:
            entry = json.load(f)

        entry["is_archived"] = False
        del entry["archived_at"]

        # 从索引移除
        del self._archive_index[entry_id]
        self._save_index()

        return entry

    def get_archive_stats(self) -> dict:
        """归档统计"""
        return {
            "total_archived": len(self._archive_index),
            "archive_size_mb": sum(
                Path(info["file"]).stat().st_size
                for info in self._archive_index.values()
                if Path(info["file"]).exists()
            ) / (1024 * 1024),
        }
