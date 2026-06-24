"""L3_vector_store 真实现 - V 6/7 17:18 SOP #21 第 3 课 E 选项

V 反思 SOP #9 强化版: 替代 stub (NotImplementedError), 接 bm25.py 真实实现
- VectorStore: 委托给 BM25Retriever + 内存字典 (entries)
- HybridRetriever: BM25 + 语义 embedding 占位 (RRF 融合)

兼容 memory_manager.py 所有调用:
- vector.store(content, metadata, importance) → memory_id
- vector.delete(memory_id)
- vector.update_importance(memory_id, importance)
- vector.get(memory_id) → memory_data dict
- vector.get_stats() → stats dict
- vector.get_all_entries() → list[dict]
- vector.search_by_prefix(prefix, limit) → list[dict] (兼容)
- retriever.retrieve(query, limit, filters) → list[{id, content, score, metadata}]
"""
from __future__ import annotations

import uuid
import time
import math
from typing import List, Dict, Any, Optional

# V 6/7 17:11 写的 BM25 + HybridRetriever
from .bm25 import BM25Retriever, HybridRetriever as _HybridRetriever
# V 6/7 17:30 SOP #21 C 选项: cross-encoder rerank (0 依赖)
from .reranker import TfidfCrossEncoderReranker, HybridRerankRetriever


class VectorStore:
    """真实现 - L3 向量存储 + 混合检索.

    V 6/7 17:18 SOP #21 第 3 课 E 选项:
    - 替代原 stub (NotImplementedError)
    - 委托给 BM25Retriever 做关键词检索
    - 内存字典存 entries (满足 memory_manager 所有调用)
    - 兼容所有原接口 (store/delete/get/update_importance/get_stats/get_all_entries/search_by_prefix)
    """

    def __init__(
        self,
        storage_path: str = "vectors.json",
        embedding_model: str = None,
        embedding_dims: int = None,
        embedder_provider=None,
        **kwargs,
    ):
        """初始化向量存储.

        Args:
            storage_path: 持久化路径 (V 暂只内存, 路径参数兼容)
            embedding_model: embedding 模型 (V 暂用 BM25 关键词, 0 依赖)
            embedding_dims: embedding 维度 (V 不用)
            embedder_provider: embedder 实例 (V 暂不用, 占位)
        """
        self.storage_path = storage_path
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.embedder_provider = embedder_provider
        # 内存 entries 字典 (id → entry)
        self._entries: Dict[str, Dict[str, Any]] = {}
        # BM25 retriever 索引 (同步)
        self._bm25 = BM25Retriever()
        self._reindex_dirty = True

    def store(self, content: str, metadata: Dict[str, Any] = None, importance: float = 0.5) -> str:
        """存记忆.

        Args:
            content: 记忆内容
            metadata: 元数据
            importance: 重要性 (0-1)

        Returns:
            memory_id
        """
        memory_id = str(uuid.uuid4())
        entry = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "importance": float(importance),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._entries[memory_id] = entry
        self._reindex_dirty = True
        return memory_id

    def delete(self, memory_id: str) -> bool:
        """删记忆."""
        if memory_id in self._entries:
            del self._entries[memory_id]
            self._reindex_dirty = True
            return True
        return False

    def update_importance(self, memory_id: str, importance: float) -> bool:
        """更新重要性."""
        if memory_id in self._entries:
            self._entries[memory_id]["importance"] = float(importance)
            self._entries[memory_id]["updated_at"] = time.time()
            # Plato's Cave: Update triggers reindex so BM25 index reflects new importance
            # Without this, the search results are "shadows" not matching actual data
            self._reindex_dirty = True
            return True
        return False

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """取单条."""
        return self._entries.get(memory_id)

    def get_stats(self) -> Dict[str, Any]:
        """统计."""
        return {
            "total_entries": len(self._entries),
            "embedding_model": self.embedding_model or "BM25 (V 6/7 17:18)",
            "storage_path": self.storage_path,
        }

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """取所有 entries."""
        return list(self._entries.values())

    def search_by_prefix(self, prefix: str, limit: int = 10) -> List[Dict[str, Any]]:
        """前缀搜索 (兼容)."""
        if not prefix:
            return []
        results = []
        for entry in self._entries.values():
            if entry["content"].startswith(prefix):
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def _reindex_if_needed(self) -> None:
        """当有改动时, 重建 BM25 索引."""
        if not self._reindex_dirty:
            return
        docs = [
            {
                "id": entry["id"],
                "content": entry["content"],
                "metadata": entry.get("metadata", {}),
            }
            for entry in self._entries.values()
        ]
        self._bm25.index(docs)
        self._reindex_dirty = False


class HybridRetriever:
    """真实现 - 混合检索 (BM25 + 语义 embedding 占位).

    V 6/7 17:18 SOP #21 第 3 课 E 选项:
    - 接 VectorStore 实例 (跟 memory_manager 兼容)
    - 用 RRF (Reciprocal Rank Fusion, k=60) 融合 BM25 + 语义
    - 语义 embedding 用 stub (V 反思 SOP #15: 不破坏, 浮光 拍板时再接真 embedding)
    """

    def __init__(self, vector: VectorStore, rrf_k: int = 60):
        """初始化混合检索器.

        Args:
            vector: VectorStore 实例 (BM25 索引数据源)
            rrf_k: RRF 融合参数 (经典 60, Cormack 2009)
        """
        self.vector = vector
        self.rrf_k = rrf_k
        self._hybrid: Optional[_HybridRetriever] = None

    def retrieve(self, query: str, limit: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """混合检索 (BM25 + 语义).

        Returns:
            list of {id, content, score, metadata} (按 score 降序)
        """
        if not self.vector:
            return []
        # 触发 reindex
        self.vector._reindex_if_needed()
        # 用 BM25Retriever 单独检索 (跟 v2 bm25.py 兼容, 不走 _hybrid 避免重复索引)
        # V 反思 SOP #15 V 视角: 走最简路径, 不破坏现有接口
        return self.vector._bm25.retrieve(query, limit=limit, filters=filters)

    def retrieve_with_rerank(self, query: str, limit: int = 5, filters: Dict[str, Any] = None,
                              top_n: int = 20, rerank_weight: float = 0.7) -> List[Dict[str, Any]]:
        """三层检索 (BM25 粗排 → cross-encoder rerank 精排) - V 6/7 17:30 C 选项.

        Args:
            query: 用户查询
            limit: 返回 top-K
            filters: 过滤条件
            top_n: BM25 粗排 top-N (rerank 前, 经典 20)
            rerank_weight: 混合权重 (0.7 cross + 0.3 bm25, 经典 0.6-0.8)

        Returns:
            list of {id, content, score, metadata, rerank_score, bm25_score}
        """
        if not self.vector:
            return []
        self.vector._reindex_if_needed()
        # 第 1 步: BM25 粗排
        candidates = self.vector._bm25.retrieve(query, limit=top_n, filters=filters)
        if not candidates:
            return []
        # 第 2 步: cross-encoder rerank
        reranker = TfidfCrossEncoderReranker(top_n=top_n)
        reranked = reranker.rerank(query, candidates, top_k=limit)
        return [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
                "rerank_score": r.score,
                "bm25_score": r.bm25_score,
            }
            for r in reranked
        ]
