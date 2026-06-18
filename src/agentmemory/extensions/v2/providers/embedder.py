"""
AgentMemory v2.0 - Embedder Provider 实现

支持：
- MockEmbedder：确定性 hash 向量，无 API Key 兜底
- OpenAIEmbedder：OpenAI 兼容 API
- MinimaxEmbedder：Minimax Embedding API
- BailianEmbedder：阿里百炼/通义 API
"""

import os
import hashlib
import asyncio
from typing import Annotated, Literal

import httpx

from .protocols import (
    BaseEmbedderProvider,
    EmbedderConfig,
)


def _hash_to_vector(text: str, dimensions: int) -> list[float]:
    """
    将文本确定性 hash 为固定维度的向量
    
    使用 SHA256 哈希生成伪随机但确定性的向量，
    确保相同文本总是产生相同的向量。
    
    Args:
        text: 输入文本
        dimensions: 向量维度
        
    Returns:
        归一化的向量
    """
    # 使用 SHA256 生成确定性的字节序列
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    
    # 将哈希字节转换为浮点数序列
    vector = []
    for i in range(dimensions):
        # 使用循环方式从哈希字节中提取值
        byte_idx = (i * 2) % len(hash_bytes)
        high_byte = hash_bytes[byte_idx]
        low_byte = hash_bytes[(byte_idx + 1) % len(hash_bytes)]
        
        # 组合成 16 位整数，然后转换为 0-1 之间的浮点数
        value = ((high_byte << 8) | low_byte) / 65535.0
        vector.append(value)
    
    # L2 归一化
    magnitude = sum(v * v for v in vector) ** 0.5
    if magnitude > 0:
        vector = [v / magnitude for v in vector]
    
    return vector


class MockEmbedder(BaseEmbedderProvider):
    """
    Mock Embedder - 确定性 hash 向量
    
    无需 API Key，使用文本的 SHA256 哈希生成确定性的嵌入向量。
    适用于：
    - 本地开发测试
    - 无网络环境
    - CI/CD 流水线
    
    向量特点：
    - 相同文本 → 相同向量
    - 相似的文本 → 可能有完全不同的向量（这是 hash 的特性）
    - 可重现的测试结果
    """
    
    def __init__(
        self,
        dimensions: int = 384,
        model: str = "mock-hash-v1",
    ):
        """
        初始化 MockEmbedder
        
        Args:
            dimensions: 向量维度，默认 384
            model: 模型名称标识
        """
        self._dimensions = dimensions
        self._model = model
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model(self) -> str:
        return self._model
    
    def embed(self, text: str) -> list[float]:
        """单文本嵌入"""
        if not text:
            return [0.0] * self._dimensions
        return _hash_to_vector(text, self._dimensions)
    
    async def embed_async(self, text: str) -> list[float]:
        """异步单文本嵌入（Mock 版本即同步）"""
        # 模拟异步调用开销
        await asyncio.sleep(0)
        return self.embed(text)
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本嵌入"""
        if not texts:
            return []
        return [self.embed(text) for text in texts]
    
    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """异步批量文本嵌入"""
        # Mock 版本并行调用
        await asyncio.sleep(0)
        return self.embed_batch(texts)


class OpenAIEmbedder(BaseEmbedderProvider):
    """
    OpenAI 兼容 Embedder
    
    支持 OpenAI API 以及任何兼容 OpenAI 格式的 API（如 LocalAI、Ollama 等）
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536
    DEFAULT_API_BASE = "https://api.openai.com/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        api_base: str | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
    ):
        """
        初始化 OpenAI Embedder
        
        Args:
            api_key: API Key，优先从环境变量 OPENAI_API_KEY 读取
            model: 模型名称
            dimensions: 向量维度（None 则使用模型默认值）
            api_base: API 基础 URL
            batch_size: 批量处理大小
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model = model
        self._dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self._api_base = api_base or os.getenv("OPENAI_API_BASE", self.DEFAULT_API_BASE)
        self._batch_size = batch_size
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model(self) -> str:
        return self._model
    
    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._client
    
    def embed(self, text: str) -> list[float]:
        """单文本嵌入"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.embed_async(text))
    
    async def embed_async(self, text: str) -> list[float]:
        """异步单文本嵌入"""
        results = await self.embed_batch_async([text])
        return results[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本嵌入"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.embed_batch_async(texts)
        )
    
    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """异步批量文本嵌入"""
        if not texts:
            return []
        
        client = self._get_client()
        url = f"{self._api_base}/embeddings"
        
        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            
            payload = {
                "input": batch,
                "model": self._model,
                "dimensions": self._dimensions,
            }
            
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def aclose(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


class MinimaxEmbedder(BaseEmbedderProvider):
    """
    Minimax Embedder
    
    使用 Minimax Embedding API
    """
    
    DEFAULT_MODEL = "embo-01"
    DEFAULT_DIMENSIONS = 1024
    DEFAULT_API_BASE = "https://api.minimax.chat/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        group_id: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        api_base: str | None = None,
        batch_size: int = 8,
        timeout: float = 60.0,
    ):
        """
        初始化 Minimax Embedder
        
        Args:
            api_key: API Key，优先从环境变量 MINIMAX_API_KEY 读取
            group_id: Group ID，从环境变量 MINIMAX_GROUP_ID 读取
            model: 模型名称
            dimensions: 向量维度
            api_base: API 基础 URL
            batch_size: 批量处理大小
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.getenv("MINIMAX_GROUP_ID", "")
        self._model = model
        self._dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self._api_base = api_base or self.DEFAULT_API_BASE
        self._batch_size = batch_size
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model(self) -> str:
        return self._model
    
    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client
    
    def embed(self, text: str) -> list[float]:
        """单文本嵌入"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.embed_async(text))
    
    async def embed_async(self, text: str) -> list[float]:
        """异步单文本嵌入"""
        results = await self.embed_batch_async([text])
        return results[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本嵌入"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.embed_batch_async(texts)
        )
    
    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """异步批量文本嵌入"""
        if not texts:
            return []
        
        client = self._get_client()
        url = f"{self._api_base}/embeddings"
        
        # Minimax 批量限制较小
        all_embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            
            payload = {
                "input": batch,
                "model": self._model,
                "group_id": self.group_id,
            }
            
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def aclose(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


class BailianEmbedder(BaseEmbedderProvider):
    """
    阿里百炼（通义）Embedder
    
    使用阿里百炼 Embedding API
    """
    
    DEFAULT_MODEL = "text-embedding-v2"
    DEFAULT_DIMENSIONS = 1536
    DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/api/v1/services/embeddings"
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        api_base: str | None = None,
        batch_size: int = 8,
        timeout: float = 60.0,
    ):
        """
        初始化阿里百炼 Embedder
        
        Args:
            api_key: API Key，优先从环境变量 DASHSCOPE_API_KEY 读取
            model: 模型名称
            dimensions: 向量维度
            api_base: API 基础 URL
            batch_size: 批量处理大小
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._model = model
        self._dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self._api_base = api_base or self.DEFAULT_API_BASE
        self._batch_size = batch_size
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model(self) -> str:
        return self._model
    
    def _get_client(self) -> httpx.AsyncClient:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client
    
    def embed(self, text: str) -> list[float]:
        """单文本嵌入"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.embed_async(text))
    
    async def embed_async(self, text: str) -> list[float]:
        """异步单文本嵌入"""
        results = await self.embed_batch_async([text])
        return results[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本嵌入"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.embed_batch_async(texts)
        )
    
    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """异步批量文本嵌入"""
        if not texts:
            return []
        
        client = self._get_client()
        
        # 百炼 API 不支持批量，需要逐个调用
        embeddings = []
        for text in texts:
            payload = {
                "model": self._model,
                "input": {"texts": [text]},
                "parameters": {
                    "dimension": self._dimensions,
                },
            }
            
            response = await client.post(self._api_base, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embedding = data["data"]["embeddings"][0]["embedding"]
            embeddings.append(embedding)
        
        return embeddings
    
    async def aclose(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


def get_embedder(
    provider: str | None = None,
    dimensions: int = 384,
    **kwargs,
) -> BaseEmbedderProvider:
    """
    工厂函数：获取 Embedder Provider
    
    自动检测环境变量，按优先级选择：
    1. 显式指定 provider
    2. DASHSCOPE_API_KEY → BailianEmbedder
    3. MINIMAX_API_KEY → MinimaxEmbedder
    4. OPENAI_API_KEY → OpenAIEmbedder
    5. 默认 → MockEmbedder
    
    Args:
        provider: 强制指定的 provider（"openai" | "minimax" | "bailian" | "mock"）
        dimensions: 向量维度（MockEmbedder 专用）
        **kwargs: 传递给具体 Provider 的参数
        
    Returns:
        Embedder Provider 实例
    """
    # 显式指定
    if provider:
        provider = provider.lower()
        if provider in ("openai", "openai-compatible", "openai-compat"):
            return OpenAIEmbedder(dimensions=dimensions, **kwargs)
        elif provider in ("minimax", "minimax-embedder"):
            return MinimaxEmbedder(dimensions=dimensions, **kwargs)
        elif provider in ("bailian", "dashscope", "tongyi", "qwen"):
            return BailianEmbedder(dimensions=dimensions, **kwargs)
        elif provider in ("mock", "hash", "fake"):
            return MockEmbedder(dimensions=dimensions, **kwargs)
        else:
            raise ValueError(f"Unknown embedder provider: {provider}")
    
    # 自动检测环境变量
    if os.getenv("DASHSCOPE_API_KEY"):
        return BailianEmbedder(dimensions=dimensions, **kwargs)
    if os.getenv("MINIMAX_API_KEY"):
        return MinimaxEmbedder(dimensions=dimensions, **kwargs)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedder(dimensions=dimensions, **kwargs)
    
    # 默认使用 Mock
    return MockEmbedder(dimensions=dimensions, **kwargs)
