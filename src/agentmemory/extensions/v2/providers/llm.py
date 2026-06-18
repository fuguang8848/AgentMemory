"""
AgentMemory v2.0 - LLM Provider 实现

支持：
- MockLLMProvider：确定性回复，无 API Key 兜底
- BailianProvider：阿里百炼/通义 LLM API
- MinimaxProvider：Minimax LLM API
- OpenAICompatProvider：OpenAI 兼容 API
"""

import os
import asyncio
from typing import AsyncIterator

import httpx

from .protocols import (
    BaseLLMProvider,
    LLMConfig,
    LLMResponse,
)


# Provider 映射表
_PROVIDER_MAP: dict[str, type[BaseLLMProvider]] = {}


def _register_provider(name: str):
    """Provider 注册装饰器"""
    def decorator(cls: type[BaseLLMProvider]):
        _PROVIDER_MAP[name] = cls
        return cls
    return decorator


@_register_provider("mock")
class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM Provider - 确定性回复
    
    无需 API Key，使用模板生成确定性回复。
    适用于：开发测试、CI/CD 流水线
    """
    
    def __init__(
        self,
        model: str = "mock-gpt-4",
        response_template: str = "Mock response to: {query}",
    ):
        """
        初始化 Mock LLM Provider
        
        Args:
            model: 模型名称标识
            response_template: 响应模板，{query} 会被替换为用户查询
        """
        self._model = model
        self._response_template = response_template
    
    @property
    def model(self) -> str:
        return self._model
    
    def chat(self, messages: list[dict]) -> LLMResponse:
        """同步聊天"""
        return asyncio.get_event_loop().run_until_complete(
            self.chat_async(messages)
        )
    
    async def chat_async(self, messages: list[dict]) -> LLMResponse:
        """异步聊天"""
        # 提取最后一条用户消息
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        response_text = self._response_template.format(query=user_message)
        
        return LLMResponse(
            content=response_text,
            usage={
                "prompt_tokens": len(user_message) // 4,
                "completion_tokens": len(response_text) // 4,
            },
            model=self._model,
        )
    
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式补全"""
        response_text = self._response_template.format(query=prompt)
        
        # 模拟流式输出
        for char in response_text:
            await asyncio.sleep(0.01)  # 模拟延迟
            yield char
    
    async def aclose(self) -> None:
        """清理资源（Mock 无需清理）"""
        pass


@_register_provider("bailian")
class BailianProvider(BaseLLMProvider):
    """
    阿里百炼（通义）LLM Provider
    
    使用阿里云百炼 API（DashScope）
    """
    
    DEFAULT_MODEL = "qwen3.6-plus"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        初始化阿里百炼 LLM Provider
        
        Args:
            api_key: API Key，从环境变量 DASHSCOPE_API_KEY 读取
            model: 模型名称，默认 qwen3.6-plus
            base_url: API 基础 URL
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
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
    
    def chat(self, messages: list[dict]) -> LLMResponse:
        """同步聊天"""
        return asyncio.get_event_loop().run_until_complete(
            self.chat_async(messages)
        )
    
    async def chat_async(self, messages: list[dict]) -> LLMResponse:
        """异步聊天"""
        client = self._get_client()
        url = f"{self._base_url}/chat/completions"
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        
        response = await client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage", {}),
            raw=data,
            model=data.get("model", self._model),
        )
    
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式补全"""
        client = self._get_client()
        url = f"{self._base_url}/chat/completions"
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "stream": True,
        }
        
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    
                    import json
                    data = json.loads(data_str)
                    
                    if "choices" in data:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
    
    async def aclose(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


@_register_provider("minimax")
class MinimaxProvider(BaseLLMProvider):
    """
    Minimax LLM Provider
    
    使用 Minimax API
    """
    
    DEFAULT_MODEL = "abab6.5s-chat"
    DEFAULT_BASE_URL = "https://api.minimax.chat/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        group_id: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        初始化 Minimax LLM Provider
        
        Args:
            api_key: API Key，从环境变量 MINIMAX_API_KEY 读取
            group_id: Group ID，从环境变量 MINIMAX_GROUP_ID 读取
            model: 模型名称
            base_url: API 基础 URL
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.group_id = group_id or os.getenv("MINIMAX_GROUP_ID", "")
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
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
    
    def chat(self, messages: list[dict]) -> LLMResponse:
        """同步聊天"""
        return asyncio.get_event_loop().run_until_complete(
            self.chat_async(messages)
        )
    
    async def chat_async(self, messages: list[dict]) -> LLMResponse:
        """异步聊天"""
        client = self._get_client()
        url = f"{self._base_url}/text/chatcompletion_v2"
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "group_id": self.group_id,
        }
        
        response = await client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage", {}),
            raw=data,
            model=data.get("model", self._model),
        )
    
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式补全"""
        client = self._get_client()
        url = f"{self._base_url}/text/chatcompletion_v2"
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "stream": True,
            "group_id": self.group_id,
        }
        
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    import json
                    data = json.loads(data_str)
                    
                    if "choices" in data:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
    
    async def aclose(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


@_register_provider("openai")
class OpenAICompatProvider(BaseLLMProvider):
    """
    OpenAI 兼容 LLM Provider
    
    支持 OpenAI API 以及任何兼容 OpenAI 格式的 API
    """
    
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        初始化 OpenAI 兼容 LLM Provider
        
        Args:
            api_key: API Key，从环境变量 OPENAI_API_KEY 读取
            model: 模型名称
            base_url: API 基础 URL
            temperature: 温度参数
            max_tokens: 最大 token 数
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model = model or self.DEFAULT_MODEL
        self._base_url = base_url or os.getenv("OPENAI_API_BASE", self.DEFAULT_BASE_URL)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
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
    
    def chat(self, messages: list[dict]) -> LLMResponse:
        """同步聊天"""
        return asyncio.get_event_loop().run_until_complete(
            self.chat_async(messages)
        )
    
    async def chat_async(self, messages: list[dict]) -> LLMResponse:
        """异步聊天"""
        client = self._get_client()
        url = f"{self._base_url}/chat/completions"
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        
        response = await client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage", {}),
            raw=data,
            model=data.get("model", self._model),
        )
    
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式补全"""
        client = self._get_client()
        url = f"{self._base_url}/chat/completions"
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "stream": True,
        }
        
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    import json
                    data = json.loads(data_str)
                    
                    if "choices" in data:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
    
    async def aclose(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


def get_llm_provider(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """
    工厂函数：获取 LLM Provider
    
    自动检测环境变量，按优先级选择：
    1. 显式指定 provider
    2. DASHSCOPE_API_KEY → BailianProvider
    3. MINIMAX_API_KEY → MinimaxProvider
    4. OPENAI_API_KEY → OpenAICompatProvider
    5. 默认 → MockLLMProvider
    
    Args:
        provider: 强制指定的 provider（"openai" | "minimax" | "bailian" | "mock"）
        model: 模型名称
        **kwargs: 传递给具体 Provider 的参数
        
    Returns:
        LLM Provider 实例
    """
    # 显式指定
    if provider:
        provider = provider.lower()
        if provider in _PROVIDER_MAP:
            return _PROVIDER_MAP[provider](model=model, **kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    # 自动检测环境变量
    if os.getenv("DASHSCOPE_API_KEY"):
        return BailianProvider(model=model, **kwargs)
    if os.getenv("MINIMAX_API_KEY"):
        return MinimaxProvider(model=model, **kwargs)
    if os.getenv("OPENAI_API_KEY"):
        return OpenAICompatProvider(model=model, **kwargs)
    
    # 默认使用 Mock
    return MockLLMProvider(model=model or "mock-gpt-4", **kwargs)
