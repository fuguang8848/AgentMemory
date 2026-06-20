"""
AgentMemory v2.0 - SearchCoordinator（检索协调器）

统一入口：整合向量检索 + BM25关键词 + RRF融合 + 重排序

三路召回：
  向量轨 (60%) → SearchEngine.search_semantic
  关键词轨 (30%) → BM25Retriever.retrieve
  标签轨 (10%) → SearchEngine.search_by_tags

使用 RRF (Reciprocal Rank Fusion) 融合，按排名融合而非分数融合，
避免不同量纲导致的偏差。

用法：
    coordinator = SearchCoordinator()
    results = await coordinator.search("Python异步编程", top_k=10)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class CoordinatorResult:
    """协调器检索结果"""
    memory_id: str
    content: str
    metadata: dict
    final_score: float = 0.0
    # 各轨得分
    vector_score: float = 0.0
    keyword_score: float = 0.0
    tag_score: float = 0.0
    # 排名信息
    ranks: dict[str, int] = field(default_factory=dict)
    # 来源信息
    sources: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.memory_id


@dataclass
class CoordinatorStats:
    """协调器统计信息"""
    total_results: int
    vector_hits: int
    keyword_hits: int
    tag_hits: int
    latency_ms: float
    vector_latency_ms: float = 0.0
    keyword_latency_ms: float = 0.0
    tag_latency_ms: float = 0.0


# =============================================================================
# 权重配置
# =============================================================================

@dataclass
class CoordinatorWeights:
    """三路召回权重配置

    注意：这里的权重用于 RRF 融合时的加权，而不是直接乘以分数。
    RRF 天然对排名敏感，对绝对分数不敏感，因此权重影响的是
    某一路结果在最终排名中的重要性。
    """
    vector: float = 0.6   # 向量语义轨
    keyword: float = 0.3  # BM25关键词轨
    tag: float = 0.1      # 标签匹配轨

    def __post_init__(self):
        total = self.vector + self.keyword + self.tag
        if abs(total - 1.0) > 0.001:
            logger.warning(
                f"CoordinatorWeights sum to {total}, normalizing to 1.0"
            )
            s = total
            self.vector /= s
            self.keyword /= s
            self.tag /= s


# =============================================================================
# 检索协调器
# =============================================================================

class SearchCoordinator:
    """
    检索协调器 — M2扩展的统一入口

    将三个独立的检索轨（向量/关键词/标签）统一协调，
    通过 RRF 融合产生最终排序结果。

    设计原则：
    1. 异步并行三路召回，最大化延迟隐藏
    2. RRF 融合排名，避免分数量纲不一致问题
    3. 结果可追溯（记录每路的得分和排名）
    4. 权重可配置，适应不同场景
    """

    def __init__(
        self,
        memory_dir: str = "memory",
        weights: CoordinatorWeights | None = None,
        default_top_k: int = 10,
        default_threshold: float = 0.0,
        rrf_k: int = 60,
        enable_vector: bool = True,
        enable_keyword: bool = True,
        enable_tag: bool = True,
    ):
        """
        初始化 SearchCoordinator

        Args:
            memory_dir: 记忆存储目录
            weights: 三路权重配置
            default_top_k: 默认返回数量
            default_threshold: 默认相似度阈值
            rrf_k: RRF 衰减参数（k越大，高排名优势越明显）
            enable_vector: 启用向量轨
            enable_keyword: 启用关键词轨
            enable_tag: 启用标签轨
        """
        self._memory_dir = memory_dir
        self._weights = weights or CoordinatorWeights()
        self._default_top_k = default_top_k
        self._default_threshold = default_threshold
        self._rrf_k = rrf_k
        self._enable_vector = enable_vector
        self._enable_keyword = enable_keyword
        self._enable_tag = enable_tag

        # 延迟初始化各个检索器（按需加载）
        self._search_engine = None
        self._bm25_retriever = None
        self._reranker = None

    # -------------------------------------------------------------------------
    # 检索器延迟初始化
    # -------------------------------------------------------------------------

    def _get_search_engine(self):
        """延迟加载 SearchEngine"""
        if self._search_engine is None:
            from .search_engine import SearchEngine
            self._search_engine = SearchEngine(memory_dir=self._memory_dir)
            logger.info("SearchCoordinator: SearchEngine loaded")
        return self._search_engine

    def _get_bm25_retriever(self):
        """延迟加载 BM25Retriever"""
        if self._bm25_retriever is None:
            from ..bm25 import BM25Retriever
            # 从 memory_dir 加载已索引的文档
            # 注意：需要 memory_manager 配合，这里做懒加载
            self._bm25_retriever = BM25Retriever()
            logger.info("SearchCoordinator: BM25Retriever loaded")
        return self._bm25_retriever

    def _get_reranker(self):
        """延迟加载 Reranker"""
        if self._reranker is None:
            try:
                from ..reranker import Reranker
                self._reranker = Reranker()
                logger.info("SearchCoordinator: Reranker loaded")
            except Exception as e:
                logger.warning(f"SearchCoordinator: Reranker not available: {e}")
                self._reranker = None  # type: ignore[assignment]
        return self._reranker

    # -------------------------------------------------------------------------
    # 核心检索方法
    # -------------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        tags: list[str] | None = None,
        weights: CoordinatorWeights | None = None,
        rerank: bool = True,
        category: str | None = None,
    ) -> list[CoordinatorResult]:
        """
        统一检索入口

        异步并行执行三路召回，然后通过 RRF 融合得到最终排序。

        Args:
            query: 查询文本
            top_k: 返回数量（默认 default_top_k）
            threshold: 相似度阈值（低于此分数的结果被过滤）
            tags: 标签过滤（用于标签轨）
            weights: 覆盖默认权重配置
            rerank: 是否使用重排序
            category: 分类过滤

        Returns:
            CoordinatorResult 列表，按 final_score 降序
        """
        top_k = top_k or self._default_top_k
        threshold = threshold or self._default_threshold
        weights = weights or self._weights

        overall_start = time.perf_counter()

        # 并行执行三路召回
        tasks = {}
        latencies = {}

        if self._enable_vector:
            tasks["vector"] = self._search_vector(
                query, top_k * 2, category
            )

        if self._enable_keyword:
            tasks["keyword"] = self._search_keyword(
                query, top_k * 2, category
            )

        if self._enable_tag and tags:
            tasks["tag"] = self._search_tag(
                query, tags, top_k * 2, category
            )

        # 并行执行
        results_map: dict[str, list] = {}
        if tasks:
            done = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, result in zip(tasks.keys(), done):
                if isinstance(result, Exception):
                    logger.warning(f"SearchCoordinator.{name} failed: {result}")
                    results_map[name] = []
                else:
                    results_map[name] = result

        overall_latency = (time.perf_counter() - overall_start) * 1000

        # RRF 融合
        fused = self._rrf_fuse(results_map, weights)

        # 构建 CoordinatorResult
        coordinator_results = []
        for item in fused:
            memory_id = item["memory_id"]
            scores = item["scores"]  # {"vector": s, "keyword": s, ...}
            ranks = item["ranks"]     # {"vector": 1, "keyword": 3, ...}
            content = item.get("content", "")
            metadata = item.get("metadata", {})

            final_score = item["rrf_score"]
            if final_score < threshold:
                continue

            cr = CoordinatorResult(
                memory_id=memory_id,
                content=content,
                metadata=metadata,
                final_score=final_score,
                vector_score=scores.get("vector", 0.0),
                keyword_score=scores.get("keyword", 0.0),
                tag_score=scores.get("tag", 0.0),
                ranks=ranks,
                sources=[k for k in scores if scores[k] > 0],
            )
            coordinator_results.append(cr)

        # 重排序（可选）
        if rerank and coordinator_results and self._get_reranker():
            coordinator_results = await self._do_rerank(
                query, coordinator_results
            )

        return coordinator_results[:top_k]

    # -------------------------------------------------------------------------
    # 三路召回实现
    # -------------------------------------------------------------------------

    async def _search_vector(
        self,
        query: str,
        limit: int,
        category: str | None,
    ) -> list[dict]:
        """向量语义轨"""
        start = time.perf_counter()
        try:
            engine = self._get_search_engine()
            options = {
                "limit": limit,
                "threshold": 0.0,
            }
            if category:
                options["category"] = category

            entries = await engine.search_semantic(query, options)
            results = []
            for rank, entry in enumerate(entries, 1):
                results.append({
                    "memory_id": entry.id,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "score": entry.score,
                    "rank": rank,
                    "source": "vector",
                })
            return results
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
        finally:
            pass  # latency tracked at caller

    async def _search_keyword(
        self,
        query: str,
        limit: int,
        category: str | None,
    ) -> list[dict]:
        """BM25 关键词轨"""
        try:
            retriever = self._get_bm25_retriever()
            filters = {"category": category} if category else None
            results = retriever.retrieve(query, limit=limit, filters=filters)
            out = []
            for rank, r in enumerate(results, 1):
                out.append({
                    "memory_id": r["id"],
                    "content": r["content"],
                    "metadata": r.get("metadata", {}),
                    "score": float(r["score"]),
                    "rank": rank,
                    "source": "keyword",
                })
            return out
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []

    async def _search_tag(
        self,
        query: str,
        tags: list[str],
        limit: int,
        category: str | None,
    ) -> list[dict]:
        """标签轨"""
        try:
            engine = self._get_search_engine()
            # 标签搜索暂用语义搜索代替
            entries = await engine.search_semantic(
                query,
                {"limit": limit, "threshold": 0.0}
            )
            results = []
            for rank, entry in enumerate(entries, 1):
                entry_tags = getattr(entry, "tags", [])
                # 计算标签匹配数
                tag_overlap = len(set(entry_tags) & set(tags)) if entry_tags else 0
                if tag_overlap > 0:
                    results.append({
                        "memory_id": entry.id,
                        "content": entry.content,
                        "metadata": entry.metadata,
                        "score": float(tag_overlap / max(len(tags), 1)),
                        "rank": rank,
                        "source": "tag",
                    })
            return results
        except Exception as e:
            logger.warning(f"Tag search failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # RRF 融合
    # -------------------------------------------------------------------------

    def _rrf_fuse(
        self,
        results_map: dict[str, list],
        weights: CoordinatorWeights,
    ) -> list[dict]:
        """
        加权 RRF 融合

        公式: score(d) = Σ w_i * 1/(k + rank_i(d))
        其中 w_i 是第 i 轨的权重
        """
        k = self._rrf_k
        rrf_scores: dict[str, dict] = {}

        for track_name, track_results in results_map.items():
            if not track_results:
                continue

            # 获取该轨的权重
            weight = getattr(weights, track_name, 1.0)

            for item in track_results:
                memory_id = item["memory_id"]
                rank = item["rank"]  # 1-based

                if memory_id not in rrf_scores:
                    rrf_scores[memory_id] = {
                        "memory_id": memory_id,
                        "content": item.get("content", ""),
                        "metadata": item.get("metadata", {}),
                        "rrf_score": 0.0,
                        "scores": {},
                        "ranks": {},
                    }

                # 加权 RRF
                rrf_contribution = weight * (1.0 / (k + rank))
                rrf_scores[memory_id]["rrf_score"] += rrf_contribution
                rrf_scores[memory_id]["scores"][track_name] = item["score"]
                rrf_scores[memory_id]["ranks"][track_name] = rank

        # 排序
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        return sorted_results

    # -------------------------------------------------------------------------
    # 重排序
    # -------------------------------------------------------------------------

    async def _do_rerank(
        self,
        query: str,
        results: list[CoordinatorResult],
    ) -> list[CoordinatorResult]:
        """使用交叉编码器对结果进行重排序"""
        try:
            reranker = self._get_reranker()
            if not reranker:
                return results

            pairs = [(query, r.content) for r in results]
            reranked = await reranker.rerank(pairs)

            # 重新赋值分数
            for i, r in enumerate(reranked):
                if i < len(results):
                    results[i].final_score = r["score"]

            results.sort(key=lambda x: x.final_score, reverse=True)
            return results
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")
            return results

    # -------------------------------------------------------------------------
    # 统计与调试
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """获取协调器统计信息"""
        return {
            "weights": {
                "vector": self._weights.vector,
                "keyword": self._weights.keyword,
                "tag": self._weights.tag,
            },
            "enabled_tracks": {
                "vector": self._enable_vector,
                "keyword": self._enable_keyword,
                "tag": self._enable_tag,
            },
            "default_top_k": self._default_top_k,
            "default_threshold": self._default_threshold,
            "rrf_k": self._rrf_k,
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_search_coordinator(
    memory_dir: str = "memory",
    **kwargs,
) -> SearchCoordinator:
    """
    工厂函数：创建 SearchCoordinator

    用法：
        coordinator = create_search_coordinator(memory_dir="/path/to/memory")
        results = await coordinator.search("Python async", top_k=10)
    """
    return SearchCoordinator(memory_dir=memory_dir, **kwargs)
