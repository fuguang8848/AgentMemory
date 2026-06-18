"""
AgentMemory v2.0 - HybridRetriever（混合检索器）

混合打分策略：
- 向量相似度 60%
- Tag 命中 30%
- 重要性 10%

支持 limit / threshold / filter
"""

import asyncio
import logging
from typing import Callable, Awaitable
from dataclasses import dataclass, field

from .search_engine import SearchEngine, MemoryEntry, SearchOptions


logger = logging.getLogger(__name__)


@dataclass
class HybridWeights:
    """混合权重配置"""
    vector_similarity: float = 0.6
    tag_match: float = 0.3
    importance: float = 0.1
    
    def __post_init__(self):
        """验证权重和为 1"""
        total = self.vector_similarity + self.tag_match + self.importance
        if abs(total - 1.0) > 0.001:
            logger.warning(
                f"Hybrid weights sum to {total}, normalizing to 1.0"
            )
            self.vector_similarity /= total
            self.tag_match /= total
            self.importance /= total


@dataclass
class HybridSearchOptions(SearchOptions):
    """混合搜索选项"""
    weights: HybridWeights | None = None
    tag_match_boost: float = 1.0  # Tag 匹配加分倍率
    importance_boost: float = 1.0  # 重要性加分倍率


@dataclass
class ScoredEntry:
    """带分的记忆条目"""
    entry: MemoryEntry
    vector_score: float = 0.0
    tag_score: float = 0.0
    importance_score: float = 0.0
    final_score: float = 0.0
    
    @property
    def id(self) -> str:
        return self.entry.id
    
    @property
    def content(self) -> str:
        return self.entry.content
    
    @property
    def metadata(self) -> dict:
        return self.entry.metadata


class HybridRetriever:
    """
    混合检索器
    
    将多种信号融合为统一的相似度得分：
    - 向量相似度（默认 60%）
    - Tag 命中（默认 30%）
    - 重要性分数（默认 10%）
    
    支持自定义权重、阈值过滤、排序策略。
    """
    
    def __init__(
        self,
        search_engine: SearchEngine,
        weights: HybridWeights | None = None,
        default_limit: int = 10,
        default_threshold: float = 0.0,
    ):
        """
        初始化 HybridRetriever
        
        Args:
            search_engine: 底层 SearchEngine
            weights: 混合权重配置
            default_limit: 默认返回数量
            default_threshold: 默认相似度阈值
        """
        self._search_engine = search_engine
        self._weights = weights or HybridWeights()
        self._default_limit = default_limit
        self._default_threshold = default_threshold
    
    @property
    def weights(self) -> HybridWeights:
        """获取权重配置"""
        return self._weights
    
    @weights.setter
    def weights(self, value: HybridWeights) -> None:
        """设置权重配置"""
        self._weights = value
    
    def _calculate_tag_score(
        self,
        entry: MemoryEntry,
        query_tags: list[str] | None = None,
    ) -> float:
        """
        计算 Tag 命中得分
        
        Args:
            entry: 记忆条目
            query_tags: 查询 Tag 列表
            
        Returns:
            Tag 命中得分 (0-1)
        """
        if not query_tags:
            return 1.0  # 无查询 Tag 时返回满分
        
        entry_tags = set(entry.tags)
        query_tags_set = set(query_tags)
        
        if not entry_tags:
            return 0.0
        
        # Jaccard 相似度
        intersection = len(entry_tags & query_tags_set)
        union = len(entry_tags | query_tags_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_importance_score(self, entry: MemoryEntry) -> float:
        """
        计算重要性得分
        
        Args:
            entry: 记忆条目
            
        Returns:
            重要性得分 (0-1)
        """
        importance = entry.importance
        
        # 确保在 0-1 范围内
        return max(0.0, min(1.0, importance))
    
    def _calculate_final_score(
        self,
        vector_score: float,
        tag_score: float,
        importance_score: float,
        weights: HybridWeights,
    ) -> float:
        """
        计算最终混合得分
        
        Args:
            vector_score: 向量相似度得分
            tag_score: Tag 命中得分
            importance_score: 重要性得分
            weights: 权重配置
            
        Returns:
            最终得分 (0-1)
        """
        return (
            vector_score * weights.vector_similarity +
            tag_score * weights.tag_match +
            importance_score * weights.importance
        )
    
    def _score_entry(
        self,
        entry: MemoryEntry,
        query: str,
        query_tags: list[str] | None = None,
        weights: HybridWeights | None = None,
    ) -> ScoredEntry:
        """
        对单个条目打分
        
        Args:
            entry: 记忆条目
            query: 查询文本
            query_tags: 查询 Tag 列表
            weights: 权重配置
            
        Returns:
            ScoredEntry
        """
        weights = weights or self._weights
        
        vector_score = entry.score  # SearchEngine 已经计算了向量得分
        tag_score = self._calculate_tag_score(entry, query_tags)
        importance_score = self._calculate_importance_score(entry)
        
        final_score = self._calculate_final_score(
            vector_score, tag_score, importance_score, weights
        )
        
        return ScoredEntry(
            entry=entry,
            vector_score=vector_score,
            tag_score=tag_score,
            importance_score=importance_score,
            final_score=final_score,
        )
    
    async def search(
        self,
        query: str,
        limit: int | None = None,
        threshold: float | None = None,
        tags: list[str] | None = None,
        weights: HybridWeights | None = None,
        filter_metadata: dict | None = None,
    ) -> list[ScoredEntry]:
        """
        混合搜索
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            threshold: 相似度阈值
            tags: 查询 Tag 列表（用于 Tag 匹配）
            weights: 自定义权重（覆盖默认）
            filter_metadata: 元数据过滤条件
            
        Returns:
            ScoredEntry 列表，按混合得分降序
        """
        limit = limit or self._default_limit
        threshold = threshold or self._default_threshold
        weights = weights or self._weights
        
        # 调用底层 SearchEngine
        options = SearchOptions(
            limit=limit * 2,  # 多取一些用于过滤
            threshold=0.0,  # 先不过滤，由混合得分决定
            filter_metadata=filter_metadata,
        )
        
        entries = await self._search_engine.search_semantic(query, options)
        
        if not entries:
            return []
        
        # 对每个条目打分
        scored_entries = [
            self._score_entry(entry, query, tags, weights)
            for entry in entries
        ]
        
        # 过滤低于阈值的条目
        scored_entries = [
            se for se in scored_entries
            if se.final_score >= threshold
        ]
        
        # 排序
        scored_entries.sort(key=lambda se: se.final_score, reverse=True)
        
        # 限制数量
        return scored_entries[:limit]
    
    async def search_with_category(
        self,
        query: str,
        category: str,
        limit: int | None = None,
        threshold: float | None = None,
        tags: list[str] | None = None,
        weights: HybridWeights | None = None,
    ) -> list[ScoredEntry]:
        """
        带分类的混合搜索
        
        Args:
            query: 查询文本
            category: 分类路径
            limit: 返回结果数量
            threshold: 相似度阈值
            tags: 查询 Tag 列表
            weights: 自定义权重
            
        Returns:
            ScoredEntry 列表
        """
        limit = limit or self._default_limit
        threshold = threshold or self._default_threshold
        weights = weights or self._weights
        
        # 使用 SearchEngine 的混合搜索
        options = SearchOptions(
            limit=limit * 2,
            threshold=0.0,
        )
        
        entries = await self._search_engine.search_hybrid(
            query, category, options
        )
        
        if not entries:
            return []
        
        # 对每个条目打分
        scored_entries = [
            self._score_entry(entry, query, tags, weights)
            for entry in entries
        ]
        
        # 过滤和排序
        scored_entries = [
            se for se in scored_entries
            if se.final_score >= threshold
        ]
        scored_entries.sort(key=lambda se: se.final_score, reverse=True)
        
        return scored_entries[:limit]
    
    async def search_batch(
        self,
        queries: list[str],
        limit_per_query: int | None = None,
        threshold: float | None = None,
    ) -> list[list[ScoredEntry]]:
        """
        批量混合搜索
        
        并行执行多个查询。
        
        Args:
            queries: 查询列表
            limit_per_query: 每个查询的返回数量
            threshold: 相似度阈值
            
        Returns:
            每个查询的 ScoredEntry 列表
        """
        tasks = [
            self.search(query, limit=limit_per_query, threshold=threshold)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "weights": {
                "vector_similarity": self._weights.vector_similarity,
                "tag_match": self._weights.tag_match,
                "importance": self._weights.importance,
            },
            "search_engine": self._search_engine.get_stats(),
        }


def create_hybrid_retriever(
    memory_dir: str = "memory",
    **kwargs,
) -> HybridRetriever:
    """
    工厂函数：创建 HybridRetriever
    
    Args:
        memory_dir: 记忆存储目录
        **kwargs: 传递给 HybridRetriever 的参数
        
    Returns:
        HybridRetriever 实例
    """
    from .search_engine import create_search_engine
    
    search_engine = create_search_engine(memory_dir=memory_dir)
    
    return HybridRetriever(
        search_engine=search_engine,
        **kwargs,
    )
