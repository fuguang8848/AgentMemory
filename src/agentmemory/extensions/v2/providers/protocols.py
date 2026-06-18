"""
AgentMemory v2.0 - Provider Protocol 接口定义

定义 LLM、Embedder、VectorStore 的抽象接口契约。
使用 Protocol 实现结构化子类型（static duck typing），避免继承耦合。
"""

from typing import Protocol, AsyncIterator, TypeVar, Generic, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC


# ==================== 配置数据类 ====================


@dataclass(frozen=True)
class EmbedderConfig:
    """Embedder 配置"""
    model: str = "text-embedding-3-small"
    dimensions: int = 384
    batch_size: int = 32
    api_base: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_base: str | None = None
    timeout: float = 60.0


class DistanceMetric(Enum):
    """向量距离度量"""
    COSINE = "cosine"
    L2 = "l2"
    IP = "ip"  # Inner Product


@dataclass(frozen=True)
class VectorStoreConfig:
    """VectorStore 配置"""
    path: str = "memory.vec"
    metric: DistanceMetric = DistanceMetric.COSINE
    dimensions: int = 384
    connectivity: int = 0
    expansion_add: int = 0


@dataclass
class VectorEntry:
    """向量条目"""
    id: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    vector: list[float] | None = None


# ==================== Embedder Protocol ====================


@runtime_checkable
class BaseEmbedderProvider(Protocol):
    """
    Embedder Provider 接口协议
    
    所有 Embedder 实现必须实现以下方法：
    - embed(text: str) -> list[float]: 单文本嵌入
    - embed_batch(texts: list[str]) -> list[list[float]]: 批量嵌入
    - dimensions: int: 向量维度（只读属性）
    """
    
    @property
    def dimensions(self) -> int:
        """返回向量维度"""
        ...
    
    @property
    def model(self) -> str:
        """返回模型名称"""
        ...
    
    def embed(self, text: str) -> list[float]:
        """
        单文本嵌入
        
        Args:
            text: 待嵌入文本
            
        Returns:
            嵌入向量，维度为 self.dimensions
        """
        ...
    
    async def embed_async(self, text: str) -> list[float]:
        """
        单文本异步嵌入（可选实现）
        
        Args:
            text: 待嵌入文本
            
        Returns:
            嵌入向量
        """
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本嵌入
        
        Args:
            texts: 待嵌入文本列表
            
        Returns:
            嵌入向量列表
        """
        ...
    
    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本异步嵌入（可选实现）
        
        Args:
            texts: 待嵌入文本列表
            
        Returns:
            嵌入向量列表
        """
        ...


# ==================== LLM Protocol ====================


@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    usage: dict = field(default_factory=dict)
    raw: dict | None = None
    model: str = ""
    
    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


class BaseLLMProvider(Protocol):
    """
    LLM Provider 接口协议
    
    所有 LLM 实现必须实现以下方法：
    - chat(messages: list[dict]) -> LLMResponse: 同步聊天
    - chat_async(messages: list[dict]) -> LLMResponse: 异步聊天
    - stream_complete(prompt: str) -> AsyncIterator[str]: 流式补全
    - aclose() -> None: 清理资源
    """
    
    @property
    def model(self) -> str:
        """返回模型名称"""
        ...
    
    def chat(self, messages: list[dict]) -> LLMResponse:
        """
        同步聊天
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            
        Returns:
            LLMResponse 响应对象
        """
        ...
    
    async def chat_async(self, messages: list[dict]) -> LLMResponse:
        """
        异步聊天
        
        Args:
            messages: 消息列表
            
        Returns:
            LLMResponse 响应对象
        """
        ...
    
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        流式补全
        
        Args:
            prompt: 提示词
            
        Yields:
            增量文本片段
        """
        ...
    
    async def aclose(self) -> None:
        """清理资源（关闭连接池等）"""
        ...


# ==================== VectorStore Protocol ====================


@runtime_checkable
class BaseVectorStoreProvider(Protocol):
    """
    VectorStore Provider 接口协议
    
    所有 VectorStore 实现必须实现以下方法：
    - upsert(entries: list[VectorEntry]) -> None: 批量写入向量
    - search(query: list[float], limit: int, threshold: float) -> list[SearchResult]: 向量检索
    - delete(ids: list[str]) -> None: 删除向量
    - persist() -> None: 持久化到磁盘
    - load() -> None: 从磁盘加载
    """
    
    @property
    def dimensions(self) -> int:
        """返回向量维度"""
        ...
    
    @property
    def count(self) -> int:
        """返回向量数量"""
        ...
    
    @property
    def path(self) -> str:
        """返回存储路径"""
        ...
    
    def upsert(self, entries: list[VectorEntry]) -> None:
        """
        批量写入/更新向量
        
        Args:
            entries: VectorEntry 列表
        """
        ...
    
    async def upsert_async(self, entries: list[VectorEntry]) -> None:
        """
        异步批量写入/更新向量
        
        Args:
            entries: VectorEntry 列表
        """
        ...
    
    def search(
        self,
        query: list[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """
        向量相似度搜索
        
        Args:
            query: 查询向量
            limit: 返回结果数量上限
            threshold: 相似度阈值（0-1）
            filter_metadata: 元数据过滤条件
            
        Returns:
            SearchResult 列表，按相似度降序
        """
        ...
    
    async def search_async(
        self,
        query: list[float],
        limit: int = 10,
        threshold: float = 0.0,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """
        异步向量相似度搜索
        
        Args:
            query: 查询向量
            limit: 返回结果数量上限
            threshold: 相似度阈值（0-1）
            filter_metadata: 元数据过滤条件
            
        Returns:
            SearchResult 列表，按相似度降序
        """
        ...
    
    def delete(self, ids: list[str]) -> None:
        """
        删除向量
        
        Args:
            ids: 要删除的向量 ID 列表
        """
        ...
    
    async def delete_async(self, ids: list[str]) -> None:
        """
        异步删除向量
        
        Args:
            ids: 要删除的向量 ID 列表
        """
        ...
    
    def persist(self) -> None:
        """持久化索引到磁盘"""
        ...
    
    async def persist_async(self) -> None:
        """异步持久化索引到磁盘"""
        ...
    
    def load(self) -> None:
        """从磁盘加载索引"""
        ...
    
    async def load_async(self) -> None:
        """异步从磁盘加载索引"""
        ...
