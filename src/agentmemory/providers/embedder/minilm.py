"""
MiniLM Embedder (384 dimensions)
Implements Embedder Protocol
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Embedding result object"""
    embedding: list[float]
    model: str
    dimensions: int
    raw: dict[str, Any] | None = None


class Embedder:
    """Protocol for embedding providers"""

    async def embed(self, texts: str | list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings for text(s) - async (满足 core/embedder.py Protocol).

        V 6/7 08:42 P1 修复: 之前是 sync, 但调用方都用 await (pipeline/embed.py:172 等),
        await sync 函数实际不 work (TypeError). 改 async 后正常 await.
        """
        raise NotImplementedError

    async def aembed(self, texts: str | list[str], **kwargs) -> list[list[float]]:
        """Async embedding - 保留为向后兼容 alias, 内部 await self.embed.

        V 6/7 08:42: 旧 API. 新代码应该用 self.embed (现 async).
        保留 aembed 避免破坏现有调用方.
        """
        return await self.embed(texts, **kwargs)

    @property
    def dimensions(self) -> int:
        """Embedding dimensions"""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        """Model name identifier"""
        raise NotImplementedError


class MiniLMEmbedder(Embedder):
    """
    MiniLM-L6-v2 Embedder (384 dimensions)
    Lightweight, fast embedding model via SentenceTransformers
    """
    DIMENSIONS = 384
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
        **kwargs
    ):
        # V 6/18 10:50 fix: 原本 `self.model_name = ...` 调 base Embedder.model_name property (无 setter) → AttributeError
        # 改用 _model_name 私有 attr, property 返它 (避免递归)
        self._model_name = model_name or self.MODEL_NAME
        self.device = device or "cpu"
        self.normalize = normalize
        self.extra_kwargs = kwargs
        self._model = None

    def _get_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self._model_name, device=self.device, **self.extra_kwargs)
        return self._model

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: str | list[str], **kwargs) -> list[list[float]]:
        """Generate embeddings using MiniLM - async.

        V 6/7 08:42 P1 修复: 改 async. 内部 model.encode 是 sync (CPU bound),
        用 asyncio.to_thread 包成异步避免阻塞 event loop.
        满足 core/embedder.py Protocol 期望 (async def embed).
        """
        import asyncio
        model = self._get_model()
        single = isinstance(texts, str)
        texts_list = [texts] if single else texts

        def _encode():
            return model.encode(
                texts_list,
                normalize_embeddings=self.normalize,
                **kwargs
            )

        embeddings = await asyncio.to_thread(_encode)

        # Convert to list of floats
        result = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

        return result[0] if single else result

    async def embed_query(self, text: str) -> list[float]:
        """V 6/7 08:42 P1 修复: 满足 core/embedder.py Protocol 的 embed_query 接口.

        内部 await self.embed([text]) 取第一条.
        """
        result = await self.embed([text])
        return result[0] if result else []

    async def close(self) -> None:
        """V 6/7 08:42 P1 修复: 满足 core/embedder.py Protocol 的 close 接口.

        MiniLM 无外部连接 (本地 CPU 模型), close 是 noop.
        子类 override 可加实际清理 (e.g. 释放 GPU).
        """
        return None

    async def aembed(self, texts: str | list[str], **kwargs) -> list[list[float]]:
        """V 6/7 08:42: 内部 await self.embed (现 async, 向后兼容)."""
        return await self.embed(texts, **kwargs)
