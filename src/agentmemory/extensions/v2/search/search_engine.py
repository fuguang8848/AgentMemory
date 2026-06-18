"""
AgentMemory v2.0 - SearchEngine（双轨检索引擎）

提供三种检索模式：
1. search_semantic(query) → 向量检索（USearch + Rerank 可选）
2. search_by_category(category_path) → 图书馆分类检索
3. search_hybrid(query, category) → 双轨融合
"""

import os
import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator
from dataclasses import dataclass, field

from ..providers.protocols import (
    BaseEmbedderProvider,
    BaseVectorStoreProvider,
    SearchResult,
)
from ..providers.embedder import get_embedder
from ..providers.vectorstore import get_vectorstore
from .rrf_fusion import RRFusion, RankedResult, FusionResult
from .rrf_fusion import RRFusion, RankedResult, FusionResult

try:
    from ..data import Library
except ImportError:
    Library = None
try:
    from ..data import TagIndex
except ImportError:
    TagIndex = None


logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    vector: list[float] | None = None
    score: float = 0.0
    
    @property
    def category(self) -> str | None:
        """获取分类路径"""
        return self.metadata.get("category")
    
    @property
    def tags(self) -> list[str]:
        """获取标签列表"""
        tags = self.metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        return tags
    
    @property
    def importance(self) -> float:
        """获取重要性分数"""
        return self.metadata.get("importance", 0.5)


@dataclass
class SearchOptions:
    """搜索选项"""
    limit: int = 10
    threshold: float = 0.0
    rerank: bool = False
    category: str | None = None
    tags: list[str] | None = None
    filter_metadata: dict | None = None


class SearchEngine:
    """
    双轨检索引擎
    
    提供语义检索和分类检索两种模式：
    - 语义检索：基于向量相似度
    - 分类检索：基于图书馆分类体系
    - 混合检索：两者融合
    """
    
    def __init__(
        self,
        embedder: BaseEmbedderProvider | None = None,
        vectorstore: BaseVectorStoreProvider | None = None,
        memory_dir: str | Path = "memory",
        library_index_path: str | Path | None = None,
    ):
        """
        初始化 SearchEngine
        
        Args:
            embedder: Embedder Provider（None 则自动检测）
            vectorstore: VectorStore Provider（None 则自动检测）
            memory_dir: 记忆存储目录
            library_index_path: 图书馆索引文件路径
        """
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._memory_dir = Path(memory_dir)
        
        if library_index_path:
            self._library_index_path = Path(library_index_path)
        else:
            self._library_index_path = self._memory_dir / ".library_index.json"
        
        self._library_index: dict | None = None
        self._lock = asyncio.Lock()
        self._datalake = None
        self._tag_index = None
        self._hybrid_config = None
        self._rrf = RRFusion(k=60)    
    @property
    def embedder(self) -> BaseEmbedderProvider:
        """获取 Embedder（懒加载）"""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder
    
    @property
    def vectorstore(self) -> BaseVectorStoreProvider:
        """获取 VectorStore（懒加载）"""
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore(
                dimensions=self.embedder.dimensions
            )
        return self._vectorstore
    
    async def _load_library_index(self) -> dict:
        """加载图书馆索引"""
        if self._library_index is not None:
            return self._library_index
        
        if self._library_index_path.exists():
            with open(self._library_index_path, "r", encoding="utf-8") as f:
                self._library_index = json.load(f)
        else:
            self._library_index = {"categories": {}, "entries": {}}
        
        return self._library_index
    
    async def _save_library_index(self) -> None:
        """保存图书馆索引"""
        if self._library_index is not None:
            with open(self._library_index_path, "w", encoding="utf-8") as f:
                json.dump(self._library_index, f, ensure_ascii=False, indent=2)
    
    async def search_semantic(
        self,
        query: str,
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """
        语义检索（轨一）
        
        使用向量相似度检索相关内容。
        
        Args:
            query: 查询文本
            options: 搜索选项
            
        Returns:
            MemoryEntry 列表，按相关性降序
        """
        options = options or SearchOptions()
        
        # 生成查询向量
        query_vector = self.embedder.embed(query)
        
        # 构建过滤条件
        filter_metadata = options.filter_metadata or {}
        if options.category:
            filter_metadata["category"] = options.category
        if options.tags:
            filter_metadata["tags"] = {"$in": options.tags}
        
        # 搜索
        results = await self.vectorstore.search_async(
            query=query_vector,
            limit=options.limit,
            threshold=options.threshold,
            filter_metadata=filter_metadata if filter_metadata else None,
        )
        
        # 转换为 MemoryEntry
        entries = []
        for result in results:
            entry = MemoryEntry(
                id=result.id,
                content=result.metadata.get("content", ""),
                metadata=result.metadata,
                vector=result.vector,
                score=result.score,
            )
            entries.append(entry)
        
        # 可选的 Rerank
        if options.rerank and entries:
            entries = await self._rerank(query, entries)
        
        return entries
    
    async def _rerank(
        self,
        query: str,
        entries: list[MemoryEntry],
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """
        Rerank 排序
        
        基于查询和文档的交叉编码器评分重新排序。
        目前使用简单的关键词匹配作为替代方案。
        
        Args:
            query: 查询文本
            entries: 待排序条目
            top_k: 返回数量
            
        Returns:
            重新排序后的条目
        """
        # 简单的关键词匹配 rerank
        query_words = set(query.lower().split())
        
        def keyword_score(entry: MemoryEntry) -> float:
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            return overlap / max(len(query_words), 1)
        
        # 合并原始向量相似度和关键词得分
        reranked = []
        for entry in entries:
            combined_score = entry.score * 0.7 + keyword_score(entry) * 0.3
            entry.score = combined_score
            reranked.append(entry)
        
        # 重新排序
        reranked.sort(key=lambda e: e.score, reverse=True)
        
        return reranked[:top_k]
    
    async def search_by_category(
        self,
        category_path: str,
        recursive: bool = True,
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """
        分类检索（轨二）
        
        基于图书馆分类体系检索指定分类下的所有记忆。
        
        Args:
            category_path: 分类路径，如 "A.项目/石榴籽/语料"
            recursive: 是否递归子分类
            options: 搜索选项
            
        Returns:
            MemoryEntry 列表
        """
        options = options or SearchOptions()
        
        library_index = await self._load_library_index()
        entries_map = library_index.get("entries", {})
        
        results = []
        
        # 构建分类前缀
        category_prefix = category_path.strip("/")
        
        for entry_id, entry_data in entries_map.items():
            entry_category = entry_data.get("metadata", {}).get("category", "")
            
            # 检查分类匹配
            if recursive:
                match = entry_category.startswith(category_prefix)
            else:
                match = entry_category == category_prefix
            
            if not match:
                continue
            
            # 检查标签过滤
            if options.tags:
                entry_tags = entry_data.get("metadata", {}).get("tags", [])
                if isinstance(entry_tags, str):
                    entry_tags = [entry_tags]
                if not any(tag in entry_tags for tag in options.tags):
                    continue
            
            entry = MemoryEntry(
                id=entry_id,
                content=entry_data.get("content", ""),
                metadata=entry_data.get("metadata", {}),
                score=1.0,  # 分类检索默认满分
            )
            results.append(entry)
        
        # 限制数量
        return results[:options.limit]
    
    async def search_hybrid(
        self,
        query: str,
        category: str | None = None,
        options: SearchOptions | None = None,
    ) -> list[MemoryEntry]:
        """
        混合检索（双轨融合）
        
        同时执行语义检索和分类检索，然后融合结果。
        融合权重：向量相似度 60% + 分类命中 40%
        
        Args:
            query: 查询文本
            category: 可选的分类路径
            options: 搜索选项
            
        Returns:
            MemoryEntry 列表，按混合得分降序
        """
        options = options or SearchOptions()
        
        # 并行执行两种检索
        semantic_task = self.search_semantic(query, options)
        
        if category:
            category_options = SearchOptions(
                limit=options.limit * 2,  # 多取一些用于融合
            )
            category_task = self.search_by_category(
                category, 
                options=category_options
            )
            semantic_results, category_results = await asyncio.gather(
                semantic_task, category_task
            )
        else:
            semantic_results = await semantic_task
            category_results = []
        
        # 构建分类结果映射
        category_scores: dict[str, float] = {}
        for entry in category_results:
            category_scores[entry.id] = entry.score
        
        # 融合得分
        hybrid_scores: dict[str, tuple[MemoryEntry, float]] = {}
        
        # 添加语义检索结果
        for entry in semantic_results:
            hybrid_scores[entry.id] = (entry, entry.score * 0.6)
        
        # 添加分类检索结果
        for entry in category_results:
            if entry.id in hybrid_scores:
                # 已存在，融合
                existing_entry, existing_score = hybrid_scores[entry.id]
                hybrid_scores[entry.id] = (
                    existing_entry,
                    existing_score + entry.score * 0.4
                )
            else:
                # 不存在，只用分类得分
                hybrid_scores[entry.id] = (entry, entry.score * 0.4)
        
        # 排序
        sorted_entries = sorted(
            hybrid_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 提取条目并更新得分
        results = []
        for entry, final_score in sorted_entries[:options.limit]:
            entry.score = final_score
            results.append(entry)
        
        return results

    # ============================================================================
    # RRF 融合检索方法
    # ============================================================================

    async def _search_vector(self, query: str, limit: int) -> list:
        """
        向量轨检索
        
        Args:
            query: 查询文本
            limit: 返回数量
            
        Returns:
            RankedResult 列表
        """
        try:
            options = SearchOptions(limit=limit)
            entries = await self.search_semantic(query, options)
            return [
                RankedResult(
                    memory_id=entry.id,
                    score=entry.score,
                    rank=i,
                    source="vector"
                )
                for i, entry in enumerate(entries)
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _search_library(
        self, 
        category_path: str = None, 
        limit: int = 10
    ) -> list:
        """
        图书馆轨检索
        
        Args:
            category_path: 分类路径
            limit: 返回数量
            
        Returns:
            RankedResult 列表
        """
        if not category_path:
            return []
        
        try:
            options = SearchOptions(limit=limit)
            entries = await self.search_by_category(
                category_path, 
                options=options
            )
            return [
                RankedResult(
                    memory_id=entry.id,
                    score=entry.score,
                    rank=i,
                    source="library"
                )
                for i, entry in enumerate(entries)
            ]
        except Exception as e:
            logger.warning(f"Library search failed: {e}")
            return []

    async def _search_tags(
        self, 
        tags: list = None, 
        limit: int = 10
    ) -> list:
        """
        Tag 轨检索（基于共现图谱扩展）
        
        Args:
            tags: 标签列表
            limit: 返回数量
            
        Returns:
            RankedResult 列表
        """
        if not tags:
            return []
        
        if self._tag_index is None:
            return []
        
        try:
            matched_ids = set()
            for tag in tags:
                ids = await self._tag_index.query(tag)
                matched_ids.update(ids)
            
            results = []
            for i, memory_id in enumerate(list(matched_ids)[:limit]):
                if self._datalake:
                    content_obj = await self._datalake.get_memory(memory_id)
                    if content_obj:
                        content_tags = content_obj.metadata.get("tags", [])
                        if isinstance(content_tags, str):
                            content_tags = [content_tags]
                        match_count = sum(1 for t in tags if t in content_tags)
                        score = match_count / max(len(tags), 1)
                        results.append(
                            RankedResult(
                                memory_id=memory_id,
                                score=score,
                                rank=i,
                                source="tag"
                            )
                        )
            
            return results
        except Exception as e:
            logger.warning(f"Tag search failed: {e}")
            return []

    async def search_hybrid_rrf(
        self,
        query: str,
        limit: int = 10,
        category_path: str = None,
        tags: list = None,
        fusion_k: int = 60,
        vector_weight: float = 0.5,
        library_weight: float = 0.3,
        tag_weight: float = 0.2,
    ) -> list[dict]:
        """
        双轨混合搜索（RRF 融合）。
        
        同时查询：
        1. 向量轨：semantic search
        2. 图书馆轨：category + tag 精确匹配
        3. Tag 轨：共现图谱扩展搜索
        
        融合策略：RRF + 加权组合
        
        Args:
            query: 查询文本
            limit: 返回数量
            category_path: 分类路径过滤
            tags: 标签过滤
            fusion_k: RRF 衰减参数
            vector_weight: 向量轨权重
            library_weight: 图书馆轨权重
            tag_weight: Tag 轨权重
            
        Returns:
            list[dict]: [{
                "id": "memory_id",
                "content": "...",
                "score": 0.xx,
                "rrf_score": 0.xx,
                "sources": ["vector", "library"],
                "metadata": {...}
            }]
        """
        vector_results = await self._search_vector(query, limit * 2)
        library_results = await self._search_library(category_path, limit * 2) if category_path else []
        tag_results = await self._search_tags(tags, limit * 2) if tags else []
        
        fusion = RRFusion(k=fusion_k)
        fusion_results = fusion.fuse(
            vector_results=vector_results if vector_results else None,
            library_results=library_results if library_results else None,
            tag_results=tag_results if tag_results else None,
        )
        
        final_results = []
        for fr in fusion_results[:limit]:
            content_text = ""
            metadata = {}
            if self._datalake:
                memory_content = await self._datalake.get_memory(fr.memory_id)
                if memory_content:
                    content_text = memory_content.content
                    metadata = memory_content.metadata
            
            final_results.append({
                "id": fr.memory_id,
                "content": content_text,
                "score": fr.rrf_score,
                "rrf_score": fr.rrf_score,
                "sources": list(fr.ranks.keys()),
                "ranks": fr.ranks,
                "details": fr.details,
                "metadata": metadata,
            })
        
        return final_results

    async def index_entry(
        self,
        id: str,
        content: str,
        metadata: dict,
    ) -> None:
        """
        索引记忆条目
        
        将记忆写入向量存储和图书馆索引。
        
        Args:
            id: 记忆 ID
            content: 记忆内容
            metadata: 记忆元数据
        """
        async with self._lock:
            # 生成向量
            vector = self.embedder.embed(content)
            
            # 写入向量存储
            from ..providers.protocols import VectorEntry
            entry = VectorEntry(
                id=id,
                vector=vector,
                metadata={**metadata, "content": content},
            )
            await self.vectorstore.upsert_async([entry])
            await self.vectorstore.persist_async()
            
            # 更新图书馆索引
            library_index = await self._load_library_index()
            if "entries" not in library_index:
                library_index["entries"] = {}
            
            library_index["entries"][id] = {
                "content": content,
                "metadata": metadata,
            }
            await self._save_library_index()
    
    async def delete_entry(self, id: str) -> None:
        """
        删除记忆条目
        
        Args:
            id: 记忆 ID
        """
        async with self._lock:
            # 从向量存储删除
            await self.vectorstore.delete_async([id])
            await self.vectorstore.persist_async()
            
            # 从图书馆索引删除
            library_index = await self._load_library_index()
            if "entries" in library_index and id in library_index["entries"]:
                del library_index["entries"][id]
                await self._save_library_index()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = {
            "vectorstore_count": self.vectorstore.count,
            "vectorstore_path": self.vectorstore.path,
            "embedder_model": self.embedder.model,
            "embedder_dimensions": self.embedder.dimensions,
        }

        try:
            library_index = asyncio.get_event_loop().run_until_complete(
                self._load_library_index()
            )
            stats["library_entry_count"] = len(
                library_index.get("entries", {})
            )
            stats["library_category_count"] = len(
                library_index.get("categories", {})
            )
        except Exception:
            stats["library_entry_count"] = 0
            stats["library_category_count"] = 0

        return stats

    # ============================================================================
    # §5.7 SearchEngine 接口契约统一入口
    # ============================================================================

    RRF_K = 60
    VEC_WEIGHT = 0.6
    BM25_WEIGHT = 0.4

    async def search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        tags: list[str] | None = None,
        mode: str = "hybrid",
    ) -> list[MemoryEntry]:
        """§5.7 search — 统一入口，内部路由到 search_hybrid

        Args:
            query: 查询文本
            limit: 返回数量
            category: 可选的分类路径
            tags: 可选的标签过滤
            mode: "hybrid" | "vector" | "category"

        Returns:
            MemoryEntry 列表
        """
        options = SearchOptions(
            limit=limit,
            category=category,
            tags=tags,
        )
        if mode == "hybrid":
            return await self.search_hybrid(query, category=category, options=options)
        elif mode == "vector":
            return await self.search_semantic(query, options=options)
        elif mode == "category":
            if category:
                return await self.search_by_category(category, options=options)
            else:
                return []
        else:
            return await self.search_hybrid(query, category=category, options=options)

    async def prefetch(
        self,
        query: str,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """§5.7 prefetch — 预取语义相近的 top-K（用于 Agent 上下文注入）"""
        options = SearchOptions(limit=limit)
        return await self.search_semantic(query, options=options)


def create_search_engine(
    memory_dir: str | Path = "memory",
    **kwargs,
) -> SearchEngine:
    """
    工厂函数：创建 SearchEngine
    
    Args:
        memory_dir: 记忆存储目录
        **kwargs: 传递给 SearchEngine 的参数
        
    Returns:
        SearchEngine 实例
    """
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    return SearchEngine(
        memory_dir=str(memory_dir),
        **kwargs,
    )
