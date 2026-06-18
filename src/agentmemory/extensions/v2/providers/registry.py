"""
AgentMemory v2.0 - Provider Registry

Provider 注册表，支持：
- 自动检测环境变量选择最佳 Provider
- 延迟初始化（懒加载）
- 全局单例模式
- 自定义 Provider 注册
"""

import os
from typing import TypeVar, Generic, Callable, Any
from dataclasses import dataclass, field

from .protocols import (
    BaseEmbedderProvider,
    BaseLLMProvider,
    BaseVectorStoreProvider,
)
from .embedder import get_embedder
from .llm import get_llm_provider
from .vectorstore import get_vectorstore


T = TypeVar("T")


@dataclass
class ProviderInstance(Generic[T]):
    """Provider 实例包装器"""
    instance: T | None = None
    factory: Callable[[], T] | None = None
    config: dict = field(default_factory=dict)
    
    def get(self) -> T:
        """获取或创建实例"""
        if self.instance is None and self.factory is not None:
            self.instance = self.factory()
        if self.instance is None:
            raise RuntimeError("Provider not initialized")
        return self.instance
    
    def reset(self) -> None:
        """重置实例"""
        if self.instance is not None:
            # 尝试调用 aclose
            if hasattr(self.instance, "aclose"):
                import asyncio
                try:
                    asyncio.get_event_loop().run_until_complete(
                        self.instance.aclose()
                    )
                except Exception:
                    pass
            self.instance = None


class ProviderRegistry:
    """
    Provider 注册表
    
    管理 LLM、Embedder、VectorStore 的注册和获取。
    支持自动检测环境变量和懒加载。
    
    使用示例：
        registry = ProviderRegistry()
        registry.configure(embedder={"dimensions": 512})
        
        embedder = registry.get_embedder()
        llm = registry.get_llm()
        vectorstore = registry.get_vectorstore()
    """
    
    _instance: "ProviderRegistry | None" = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化注册表"""
        if self._initialized:
            return
        
        self._embedder: ProviderInstance[BaseEmbedderProvider] = ProviderInstance()
        self._llm: ProviderInstance[BaseLLMProvider] = ProviderInstance()
        self._vectorstore: ProviderInstance[BaseVectorStoreProvider] = ProviderInstance()
        self._config: dict[str, Any] = {}
        self._initialized = True
    
    def configure(
        self,
        embedder: dict | None = None,
        llm: dict | None = None,
        vectorstore: dict | None = None,
        **kwargs,
    ) -> None:
        """
        配置全局 Provider 设置
        
        Args:
            embedder: Embedder 配置
            llm: LLM 配置
            vectorstore: VectorStore 配置
            **kwargs: 其他配置项
        """
        if embedder:
            self._config["embedder"] = embedder
        if llm:
            self._config["llm"] = llm
        if vectorstore:
            self._config["vectorstore"] = vectorstore
        self._config.update(kwargs)
    
    def get_embedder(
        self,
        provider: str | None = None,
        **kwargs,
    ) -> BaseEmbedderProvider:
        """
        获取 Embedder Provider
        
        Args:
            provider: 强制指定的 provider
            **kwargs: 传递给 Provider 的参数
            
        Returns:
            Embedder Provider 实例
        """
        config = {**self._config.get("embedder", {}), **kwargs}
        
        if provider is None:
            provider = config.pop("provider", None)
        else:
            config.pop("provider", None)
        
        return get_embedder(provider=provider, **config)
    
    def get_llm(
        self,
        provider: str | None = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """
        获取 LLM Provider
        
        Args:
            provider: 强制指定的 provider
            **kwargs: 传递给 Provider 的参数
            
        Returns:
            LLM Provider 实例
        """
        config = {**self._config.get("llm", {}), **kwargs}
        
        if provider is None:
            provider = config.pop("provider", None)
        else:
            config.pop("provider", None)
        
        return get_llm_provider(provider=provider, **config)
    
    def get_vectorstore(
        self,
        provider: str | None = None,
        **kwargs,
    ) -> BaseVectorStoreProvider:
        """
        获取 VectorStore Provider
        
        Args:
            provider: 强制指定的 provider
            **kwargs: 传递给 Provider 的参数
            
        Returns:
            VectorStore Provider 实例
        """
        config = {**self._config.get("vectorstore", {}), **kwargs}
        
        if provider is None:
            provider = config.pop("provider", None)
        else:
            config.pop("provider", None)
        
        return get_vectorstore(provider=provider, **config)
    
    def reset(self) -> None:
        """重置所有 Provider 实例"""
        self._embedder.reset()
        self._llm.reset()
        self._vectorstore.reset()
    
    @classmethod
    def get_instance(cls) -> "ProviderRegistry":
        """获取单例实例"""
        return cls()


# 全局注册表实例
_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """获取全局 Provider 注册表"""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


def register_provider(
    name: str,
    provider_type: str,
    factory: Callable[[], Any],
) -> None:
    """
    注册自定义 Provider
    
    Args:
        name: Provider 名称
        provider_type: Provider 类型（"embedder" | "llm" | "vectorstore"）
        factory: Provider 工厂函数
    """
    registry = get_registry()
    
    if provider_type == "embedder":
        original_get_embedder = get_embedder
        def custom_get_embedder(**kwargs):
            if kwargs.get("provider") == name:
                return factory()
            return original_get_embedder(**kwargs)
        # 替换全局函数
        import sys
        module = sys.modules[__name__]
        module.get_embedder = custom_get_embedder
    
    elif provider_type == "llm":
        original_get_llm_provider = get_llm_provider
        def custom_get_llm_provider(**kwargs):
            if kwargs.get("provider") == name:
                return factory()
            return original_get_llm_provider(**kwargs)
        import sys
        module = sys.modules[__name__]
        module.get_llm_provider = custom_get_llm_provider
    
    elif provider_type == "vectorstore":
        original_get_vectorstore = get_vectorstore
        def custom_get_vectorstore(**kwargs):
            if kwargs.get("provider") == name:
                return factory()
            return original_get_vectorstore(**kwargs)
        import sys
        module = sys.modules[__name__]
        module.get_vectorstore = custom_get_vectorstore
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
