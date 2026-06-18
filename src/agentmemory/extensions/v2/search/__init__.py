"""
AgentMemory v2.0 - Search 模块

检索引擎：
- SearchEngine: 双轨检索引擎（语义 + 分类）
- HybridRetriever: 混合检索器（向量 + Tag + 重要性）
"""

from .search_engine import (
    SearchEngine,
    SearchOptions,
    MemoryEntry,
    create_search_engine,
)
from .hybrid_retriever import (
    HybridRetriever,
    HybridWeights,
    HybridSearchOptions,
    ScoredEntry,
    create_hybrid_retriever,
)

__all__ = [
    # SearchEngine
    "SearchEngine",
    "SearchOptions",
    "MemoryEntry",
    "create_search_engine",
    # HybridRetriever
    "HybridRetriever",
    "HybridWeights",
    "HybridSearchOptions",
    "ScoredEntry",
    "create_hybrid_retriever",
]
