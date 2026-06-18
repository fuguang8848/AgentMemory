"""L4_file_persist - 文件持久化层 (AgentMemory 2.0)

将记忆内容持久化到文件系统，支持按日期归档。

接口（MemoryHermes 依赖）:
    FilePersistStore(workspace: str)
    .store_fact(content: str, metadata: dict) -> str  # 返回 fact_id
    .get_fact(fact_id: str) -> dict | None
    .list_facts(limit: int = 100) -> list[dict]
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FilePersistStore:
    """L4 文件持久化存储。

    存储格式：JSON Lines（每行一条记录）
    路径结构：{workspace}/{YYYY-MM-DD}.jsonl
    """

    def __init__(self, workspace: str = "memory"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, date: Optional[datetime] = None) -> Path:
        """获取指定日期的存储文件路径。"""
        d = date or datetime.now(timezone.utc)
        date_str = d.strftime("%Y-%m-%d")
        return self.workspace / f"{date_str}.jsonl"

    def store_fact(self, content: str, metadata: dict) -> str:
        """写入一条记忆到文件。

        Args:
            content: 记忆文本内容
            metadata: 元数据字典

        Returns:
            fact_id: 生成的唯一 ID
        """
        fact_id = str(uuid.uuid4())
        record = {
            "id": fact_id,
            "content": content,
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        file_path = self._get_file_path()
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return fact_id

    def get_fact(self, fact_id: str) -> Optional[dict]:
        """根据 ID 获取单条记忆。"""
        # 搜索所有 .jsonl 文件（按时间倒序）
        files = sorted(self.workspace.glob("*.jsonl"), reverse=True)
        for fp in files:
            results = []
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("id") == fact_id:
                        return record
        return None

    def list_facts(self, limit: int = 100) -> list[dict]:
        """列出最近的记忆（按时间倒序）。"""
        files = sorted(self.workspace.glob("*.jsonl"), reverse=True)
        facts = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    facts.append(record)
                    if len(facts) >= limit:
                        return facts
        return facts

    def search_facts(self, query: str, limit: int = 20) -> list[dict]:
        """简单全文搜索（逐文件扫描）。"""
        files = sorted(self.workspace.glob("*.jsonl"), reverse=True)
        results = []
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if query.lower() in record.get("content", "").lower():
                        results.append(record)
                        if len(results) >= limit:
                            return results
        return results

    def count(self) -> int:
        """返回总记录数。"""
        total = 0
        for fp in self.workspace.glob("*.jsonl"):
            with open(fp, encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        return total

    def reset(self) -> int:
        """清空所有存储文件，返回删除的文件数。"""
        count = 0
        for fp in self.workspace.glob("*.jsonl"):
            fp.unlink()
            count += 1
        return count
