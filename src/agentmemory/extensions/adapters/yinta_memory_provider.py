"""YintaMemoryProvider — 远端 DataLake + Library + DecayEngine 组合 → 本地 MemoryProvider ABC

远端 v2.0.1 的 MemoryHermes 缺 L3_vector_store / L4_file_persist / providers.*，
无法直接 import。所以这里**自己组合**远端 Phase 1 可用模块：
- DataLake (data/datalake.py) → 物理持久化
- Library (data/library.py) → 分类索引
- TagIndex (data/tag_index.py) → 标签查询
- DecayEngine (decay_engine.py) → 遗忘策略

本地 MemoryProvider ABC 要求 7 动词:
    add / search / get / update / delete / reset / history

实现策略:
- add: DataLake.write + Library.classify + TagIndex.add
- search: Library.search_by_tag / TagIndex.query → 简单关键词匹配
- get: DataLake.read
- update: DataLake.update (原子写)
- delete: DataLake.delete
- reset: 清空 DataLake + Library + TagIndex
- history: TagIndex.history（tag 加 timestamp）
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from agentmemory.core.types import MemoryItem, SearchQuery, SearchResult, MemoryLayer
from agentmemory.core.memory import MemoryProvider

# 远端 Phase 1 模块
try:
    from agentmemory.extensions.v2.data.datalake import DataLake
    from agentmemory.extensions.v2.data.library import Library
    from agentmemory.extensions.v2.data.tag_index import TagIndex
    from agentmemory.extensions.v2.data.embedding_state import EmbeddingStateMachine
    from agentmemory.extensions.v2.decay_engine import DecayEngine
    YINTA_OK = True
    _IMPORT_ERROR: str | None = None
except Exception as e:
    YINTA_OK = False
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"
    DataLake = Library = TagIndex = EmbeddingStateMachine = DecayEngine = None  # type: ignore


class YintaMemoryProvider(MemoryProvider):
    """远端数据基础设施组合的 MemoryProvider 实现 (Phase 1)。

    用法:
        provider = YintaMemoryProvider(root_dir="~/am_yinta_data")
        await provider.add("hello", tags=["greeting"], importance=0.7)
        results = await provider.search("hello")
    """

    def __init__(
        self,
        root_dir: str | Path = "~/am_yinta_data",
        library_name: str = "memory_library",
        whitelist: list[str] | None = None,
    ):
        if not YINTA_OK:
            raise ImportError(
                f"远端 data/ 模块不可用: {_IMPORT_ERROR}。"
                "检查 agentmemory.extensions.v2.data.* 是否正确移植。"
            )
        self._root = Path(root_dir).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

        # 初始化远端组件
        self._datalake = DataLake(root_dir=self._root, memory_library_name=library_name)
        # Library 是分类层级管理, 用其确保 whitelist 中的 top-level 存在
        self._library = Library(root_dir=self._root, whitelist=whitelist or ["default", "work", "personal"])
        self._tag_index = TagIndex(root_dir=self._root)
        self._embed_sm = EmbeddingStateMachine(root_dir=self._root)
        self._decay = DecayEngine()

    # ---- 7 动词接口 ----
    async def add(
        self,
        content: str | list[str],
        **kw,
    ) -> list[str]:
        if isinstance(content, str):
            content = [content]

        tags = kw.get("tags", [])
        importance = kw.get("importance", 0.5)
        category = kw.get("category", "default")
        ids: list[str] = []

        for txt in content:
            mem_id = str(uuid.uuid4())
            entry = {
                "id": mem_id,
                "content": txt,
                "importance": importance,
                "tags": tags,
                "category": category,
                "created_at": datetime.utcnow().isoformat(),
                "access_count": 0,
            }
            # 远端 DataLake.write 签名: write(content, category, metadata, importance) -> memory_id
            # 该方法是 async, 需直接 await
            new_id = await self._datalake.write(
                content=txt,
                category=[category] if isinstance(category, str) else category,
                metadata=entry,
                importance=importance,
            )
            # Library 是分类管理, 这里是 "记忆 id → 分类" 映射, 不属于 Library
            # DataLake.write 内部已处理 category 字段
            # TagIndex.add_tags(tags, memory_id, category_path) - 批量
            cat_path = category if isinstance(category, str) else "/".join(category or [])
            await self._tag_index.add_tags(tags or [], new_id, cat_path)
            ids.append(new_id)
        return ids

    async def search(
        self,
        query: str | SearchQuery,
        **kw,
    ) -> list[SearchResult]:
        if isinstance(query, SearchQuery):
            q = query.text
            limit = query.limit
        else:
            q = str(query)
            limit = kw.get("limit", 5)
        # 简单按空格分词成 tags
        tags = [t for t in q.split() if t]
        if not tags:
            return []
        # TagIndex.query_with_cooccurrence 接受 tags 列表
        candidates = await self._tag_index.query_with_cooccurrence(tags, include_related=False)
        # 去重保序
        seen: set = set()
        unique: list = []
        for cid in candidates:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)
        results: list[SearchResult] = []
        for cid in unique[:limit]:
            try:
                content_obj = await self._datalake.read(cid)
                content = getattr(content_obj, "content", "") or "" if content_obj else ""
            except Exception:
                content = ""
            results.append(SearchResult(
                item=MemoryItem(id=cid, content=content),
                score=1.0,
                layer=MemoryLayer.L3_VECTOR,
                sources=["yinta_tag_index"],
                explanation={"yinta_tag_match": 1.0},
            ))
        return results

    async def get(self, memory_id: str) -> MemoryItem | None:
        # 远端 DataLake.read 返回 MemoryContent dataclass (memory_id, content, metadata)
        entry = await self._datalake.read(memory_id)
        if not entry:
            return None
        meta = getattr(entry, "metadata", {}) or {}
        content = getattr(entry, "content", "") or ""
        return MemoryItem(
            id=getattr(entry, "memory_id", memory_id),
            content=content,
            importance=meta.get("importance", 0.5) if isinstance(meta, dict) else 0.5,
            tags=meta.get("tags", []) if isinstance(meta, dict) else [],
            created_at=meta.get("created_at", datetime.utcnow().isoformat()) if isinstance(meta, dict) else datetime.utcnow().isoformat(),
        )

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        **kw,
    ) -> bool:
        # entry 是 MemoryContent dataclass
        entry = await self._datalake.read(memory_id)
        if not entry:
            return False
        meta = dict(entry.metadata or {})
        new_content = content if content is not None else entry.content
        for k, v in kw.items():
            if k in ("importance", "tags", "category"):
                meta[k] = v
        meta["updated_at"] = datetime.utcnow().isoformat()
        cat = meta.get("category", "default")
        try:
            await self._datalake.write(
                content=new_content,
                category=[cat] if isinstance(cat, str) else cat,
                metadata=meta,
                importance=meta.get("importance", 0.5),
            )
            return True
        except Exception:
            return False

    async def delete(self, memory_id: str) -> bool:
        return await self._datalake.delete(memory_id)

    async def reset(self, scope: str = "all") -> int:
        """清空数据. 返回删除条目数."""
        # DataLake 没有 clear, 但有 list_memories + delete_memory
        try:
            memories = await self._datalake.list_memories(category_path="", recursive=True)
        except Exception:
            return 0
        count = 0
        for m in memories or []:
            mem_id = getattr(m, "memory_id", None) or (m.get("memory_id") if isinstance(m, dict) else None)
            if mem_id:
                try:
                    await self._datalake.delete_memory(mem_id)
                    count += 1
                except Exception:
                    pass
        return count

    async def history(
        self,
        memory_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """TagIndex 没有 history 方法, 用 get_all_tags + get_tags_for_memory 组合代替.
        返回最近的标签统计 (按 tag 频次)."""
        try:
            all_tags = await self._tag_index.get_all_tags()
        except Exception:
            return []
        result = []
        for tag in (all_tags or [])[:limit]:
            try:
                stats = await self._tag_index.get_tag_stats(tag)
                result.append({
                    "tag": tag,
                    "count": getattr(stats, "count", 0),
                    "categories": getattr(stats, "categories", []) if stats else [],
                })
            except Exception:
                continue
        return result
