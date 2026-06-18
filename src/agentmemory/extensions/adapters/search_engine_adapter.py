"""YintaSearchStrategy — 远端 SearchEngine 适配到本地 RetrievalStrategy ABC

⚠️ Phase 2 适配：远端 SearchEngine 缺 providers.protocols/rrf_fusion 时不可用。
导入失败时回退到 None 标签，调用方应捕获 ImportError。
"""
from __future__ import annotations

import asyncio
from typing import Any

# 本地
from agentmemory.core.retriever import RetrievalStrategy
from agentmemory.core.types import SearchQuery, SearchResult

# 远端 Phase 2 (可能缺)
try:
    from agentmemory.extensions.v2.search.search_engine import SearchEngine
    YINTA_OK = True
    _IMPORT_ERROR: str | None = None
except Exception as e:
    YINTA_OK = False
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"
    SearchEngine = None  # type: ignore


class YintaSearchStrategy(RetrievalStrategy):
    """远端 SearchEngine 包装为本地 RetrievalStrategy。

    注: Phase 2 实际未跑通（远端 search_engine.py 依赖未移植的
    providers.protocols 和 rrf_fusion 模块），但适配层已就位，
    后续移植 L3/L4/providers 后即可激活。
    """

    name = "yinta_search_v2"

    def __init__(self, *args, **kwargs):
        if not YINTA_OK:
            raise ImportError(
                f"远端 SearchEngine 不可用: {_IMPORT_ERROR}。"
                "需先移植 providers.protocols / rrf_fusion / L3_vector_store。"
            )
        self._engine = SearchEngine(*args, **kwargs)

    async def retrieve(
        self,
        query: SearchQuery,
        ctx: dict[str, Any],
    ) -> list[SearchResult]:
        # 远端是 async iterator，转 list
        results: list[SearchResult] = []
        async for r in self._engine.search_hybrid(query.text, category=None):
            results.append(SearchResult(
                item=None,  # 远端 MemoryEntry 不直接兼容本地 MemoryItem
                score=getattr(r, "score", 0.0),
                source="yinta_search",
                metadata={"id": getattr(r, "id", None)},
            ))
        return results
