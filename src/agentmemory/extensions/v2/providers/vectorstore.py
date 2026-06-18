"""
AgentMemory v2.0 - VectorStore Provider 实现

支持：
- USearchVectorStore：基于 USearch 的高性能向量存储
- MockVectorStore：内存中的简单向量存储（测试用）
"""

import os
import json
import asyncio
from pathlib import Path
from typing import AsyncIterator

from .protocols import (
    BaseVectorStoreProvider,
    VectorStoreConfig,
    VectorEntry,
    SearchResult,
    DistanceMetric,
)


def _get_usearch():
    """延迟导入 USearch，避免强依赖"""
    try:
        import usearch
        return usearch
    except ImportError:
        raise ImportError(
            "usearch not installed. Install with: pip install usearch\n"
            "Or use MockVectorStore for testing without dependencies."
        )


class USearchVectorStore(BaseVectorStoreProvider):
    """
    USearch 向量存储
    
    基于 USearch 库的高性能向量索引，支持：
    - 多种距离度量（cosine、l2、ip）
    - .usearch 文件持久化
    - 增量索引
    - 异步操作
    """
    
    def __init__(
        self,
        path: str = "memory.usearch",
        metric: DistanceMetric = DistanceMetric.COSINE,
        dimensions: int = 384,
        connectivity: int = 0,
        expansion_add: int = 0,
        expansion_search: int = 0,
    ):
        """
        初始化 USearch 向量存储
        
        Args:
            path: 索引文件路径（.usearch）
            metric: 距离度量（cosine/l2/ip）
            dimensions: 向量维度
            connectivity: 连接度参数（0=自动）
            expansion_add: 索引扩展参数
            expansion_search: 搜索扩展参数
        """
        self._path = Path(path)
        self._config = VectorStoreConfig(
            path=str(self._path),
            metric=metric,
            dimensions=dimensions,
            connectivity=connectivity,
            expansion_add=expansion_add,
        )
        self._index = None
        self._metadata: dict[str, dict] = {}  # id -> metadata
        self._metadata_path = self._path.with_suffix(".meta.json")
        self._lock = asyncio.Lock()
    
    @property
    def dimensions(self) -> int:
        return self._config.dimensions
    
    @property
    def count(self) -> int:
        return len(self._metadata) if self._index is None else self._index.size
    
    @property
    def path(self) -> str:
        return str(self._path)
    
    def _get_index(self):
        """获取或创建索引"""
        if self._index is None:
            usearch = _get_usearch()
            
            metric_map = {
                DistanceMetric.COSINE: "cos",
                DistanceMetric.L2: "l2sq",
                DistanceMetric.IP: "ip",
            }
            metric_str = metric_map.get(self._config.metric, "cos")
            
            self._index = usearch.Index(
                ndim=self._config.dimensions,
                metric=metric_str,
                connectivity=self._config.connectivity,
                expansion_add=self._config.expansion_add,
                expansion_search=self._config.expansion_search,
            )
            
            # 如果索引文件存在，加载它
            if self._path.exists():
                self._index.load(str(self._path))
                self._load_metadata()
        
        return self._index
    
    def _load_metadata(self) -> None:
        """从磁盘加载元数据"""
        if self._metadata_path.exists():
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
    
    def _save_metadata(self) -> None:
        """保存元数据到磁盘"""
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
    
    def upsert(self, entries: list[VectorEntry]) -> None:
        """批量写入/更新向量"""
        return asyncio.get_event_loop().run_until_complete(
            self.upsert_async(entries)
        )
    
    async def upsert_async(self, entries: list[VectorEntry]) -> None:
        """异步批量写入/更新向量"""
        if not entries:
            return
        
        async with self._lock:
            index = self._get_index()
            
            # 准备批量数据
            ids = []
            vectors = []
            for entry in entries:
                ids.append(entry.id)
                vectors.append(entry.vector)
                self._metadata[entry.id] = entry.metadata
            
            # 批量添加
            index.add(
                [i for i in range(len(entries))],
                vectors,
            )
            
            # 更新元数据
            self._save_metadata()
    
    def search(
        self,
        query: list[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """向量相似度搜索"""
        return asyncio.get_event_loop().run_until_complete(
            self.search_async(query, limit, threshold, filter_metadata)
        )
    
    async def search_async(
        self,
        query: list[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """异步向量相似度搜索"""
        async with self._lock:
            index = self._get_index()
            
            if index.size == 0:
                return []
            
            # 搜索
            results = index.search(query, count=limit * 2)  # 多搜一些用于过滤
            
            search_results = []
            for key, score in zip(results.keys, results.distance):
                key_str = str(key)
                if key_str not in self._metadata:
                    continue
                
                metadata = self._metadata[key_str]
                
                # 应用元数据过滤
                if filter_metadata:
                    match = all(
                        metadata.get(k) == v
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue
                
                # 转换分数（USearch 的 distance 是距离，越小越相似）
                if self._config.metric == DistanceMetric.COSINE:
                    similarity = 1.0 - min(score, 1.0)  # cos 距离转相似度
                elif self._config.metric == DistanceMetric.L2:
                    similarity = 1.0 / (1.0 + score)  # L2 距离转相似度
                else:
                    similarity = max(0.0, score)  # IP 本身就是相似度
                
                if similarity < threshold:
                    continue
                
                search_results.append(SearchResult(
                    id=key_str,
                    score=similarity,
                    metadata=metadata,
                ))
                
                if len(search_results) >= limit:
                    break
            
            # 按相似度排序
            search_results.sort(key=lambda x: x.score, reverse=True)
            
            return search_results[:limit]
    
    def delete(self, ids: list[str]) -> None:
        """删除向量"""
        return asyncio.get_event_loop().run_until_complete(
            self.delete_async(ids)
        )
    
    async def delete_async(self, ids: list[str]) -> None:
        """异步删除向量"""
        async with self._lock:
            index = self._get_index()
            
            for id_str in ids:
                try:
                    key = int(id_str)
                    index.remove([key])
                except (ValueError, KeyError):
                    pass
                
                if id_str in self._metadata:
                    del self._metadata[id_str]
            
            self._save_metadata()
    
    def persist(self) -> None:
        """持久化索引到磁盘"""
        return asyncio.get_event_loop().run_until_complete(
            self.persist_async()
        )
    
    async def persist_async(self) -> None:
        """异步持久化索引到磁盘"""
        async with self._lock:
            if self._index is not None:
                self._index.save(str(self._path))
                self._save_metadata()
    
    def load(self) -> None:
        """从磁盘加载索引"""
        return asyncio.get_event_loop().run_until_complete(
            self.load_async()
        )
    
    async def load_async(self) -> None:
        """异步从磁盘加载索引"""
        async with self._lock:
            self._get_index()  # 这会自动加载


class MockVectorStore(BaseVectorStoreProvider):
    """
    Mock 向量存储
    
    内存中的简单向量存储，适用于测试和开发。
    不依赖任何外部库。
    """
    
    def __init__(
        self,
        path: str = "memory.mock",
        dimensions: int = 384,
    ):
        """
        初始化 Mock 向量存储
        
        Args:
            path: 路径（仅用于兼容性，实际不使用）
            dimensions: 向量维度
        """
        self._path = path
        self._dimensions = dimensions
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict] = {}
        self._lock = asyncio.Lock()
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def count(self) -> int:
        return len(self._vectors)
    
    @property
    def path(self) -> str:
        return self._path
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def upsert(self, entries: list[VectorEntry]) -> None:
        """批量写入/更新向量"""
        return asyncio.get_event_loop().run_until_complete(
            self.upsert_async(entries)
        )
    
    async def upsert_async(self, entries: list[VectorEntry]) -> None:
        """异步批量写入/更新向量"""
        async with self._lock:
            for entry in entries:
                self._vectors[entry.id] = entry.vector
                self._metadata[entry.id] = entry.metadata
    
    def search(
        self,
        query: list[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """向量相似度搜索"""
        return asyncio.get_event_loop().run_until_complete(
            self.search_async(query, limit, threshold, filter_metadata)
        )
    
    async def search_async(
        self,
        query: list[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """异步向量相似度搜索"""
        async with self._lock:
            results = []
            
            for id_str, vector in self._vectors.items():
                metadata = self._metadata.get(id_str, {})
                
                # 应用元数据过滤
                if filter_metadata:
                    match = all(
                        metadata.get(k) == v
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue
                
                score = self._cosine_similarity(query, vector)
                
                if score >= threshold:
                    results.append(SearchResult(
                        id=id_str,
                        score=score,
                        metadata=metadata,
                        vector=vector,
                    ))
            
            # 按相似度排序
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results[:limit]
    
    def delete(self, ids: list[str]) -> None:
        """删除向量"""
        return asyncio.get_event_loop().run_until_complete(
            self.delete_async(ids)
        )
    
    async def delete_async(self, ids: list[str]) -> None:
        """异步删除向量"""
        async with self._lock:
            for id_str in ids:
                self._vectors.pop(id_str, None)
                self._metadata.pop(id_str, None)
    
    def persist(self) -> None:
        """持久化（Mock 版本无操作）"""
        pass
    
    async def persist_async(self) -> None:
        """异步持久化"""
        pass
    
    def load(self) -> None:
        """加载（Mock 版本无操作）"""
        pass
    
    async def load_async(self) -> None:
        """异步加载"""
        pass


def get_vectorstore(
    provider: str = "usearch",
    dimensions: int = 384,
    **kwargs,
) -> BaseVectorStoreProvider:
    """
    工厂函数：获取 VectorStore Provider
    
    Args:
        provider: Provider 类型（"usearch" | "mock"）
        dimensions: 向量维度
        **kwargs: 传递给具体 Provider 的参数
        
    Returns:
        VectorStore Provider 实例
    """
    provider = provider.lower()
    
    if provider in ("usearch", "us", "vector"):
        try:
            _get_usearch()  # 检查是否可用
            return USearchVectorStore(dimensions=dimensions, **kwargs)
        except ImportError:
            import warnings
            warnings.warn(
                "USearch not installed, falling back to MockVectorStore. "
                "Install with: pip install usearch",
                ImportWarning,
            )
            return MockVectorStore(dimensions=dimensions, **kwargs)
    
    elif provider == "mock":
        return MockVectorStore(dimensions=dimensions, **kwargs)
    
    else:
        raise ValueError(f"Unknown vectorstore provider: {provider}")
