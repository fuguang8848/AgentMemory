"""
OpenAI Compatible LLM Provider
Supports: OpenAI / 通义千问 (DashScope) / vLLM / Ollama
Implements LLMProvider Protocol
"""

from typing import Any, AsyncIterator, Iterator
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM response object"""
    content: str
    model: str
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None


class LLMProvider:
    """Protocol for LLM providers - defines interface for all LLM backends"""

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse:
        """Synchronous completion"""
        raise NotImplementedError

    async def acomplete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse:
        """Asynchronous completion"""
        raise NotImplementedError

    def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> Iterator[str]:
        """Synchronous streaming"""
        raise NotImplementedError

    async def astream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Asynchronous streaming"""
        raise NotImplementedError


class OpenAICompatLLM(LLMProvider):
    """
    OpenAI-compatible LLM provider supporting multiple backends:
    - OpenAI (api.openai.com)
    - 通义千问 (DashScope)
    - vLLM (self-hosted)
    - Ollama (local)
    """

    BACKENDS = {
        "openai": "https://api.openai.com/v1",
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "vllm": "http://localhost:8000/v1",
        "ollama": "http://localhost:11434/v1",
    }

    def __init__(
        self,
        api_key: str | None = None,
        backend: str = "openai",
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        **kwargs
    ):
        self.api_key = api_key or kwargs.get("api_key", "")
        self.backend = backend
        self.base_url = base_url or self.BACKENDS.get(backend, self.BACKENDS["openai"])
        self.default_model = default_model
        self.extra_kwargs = kwargs

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse:
        """Send completion request to OpenAI-compatible endpoint"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.extra_kwargs
        )

        response = client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse:
        """Async completion - delegate to sync for simplicity"""
        return self.complete(
            prompt, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> Iterator[str]:
        """Streaming completion"""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.extra_kwargs
        )

        stream = client.chat.completions.create(
            model=model or self.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def astream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Async streaming - simple sync wrapper"""
        for chunk in self.stream(prompt, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs):
            yield chunk


    # ============================================================
    # V 6/7 08:35 P1 修复: 满足 core/llm.py LLMProvider Protocol
    # Protocol 期望: async def chat(messages, ...)/stream(messages, **kw)/close()
    # 旧 API (complete/astream/stream/astream) 保留, 向后兼容
    # ============================================================

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
        **kw,
    ) -> LLMResponse:
        """满足 Protocol 的 chat(messages) 接口.

        V 6/7 08:35: Protocol 期望 chat(messages, ...) 签名, 之前只有
        complete(prompt, ...) 签名 (方法名 + messages vs prompt 都不同),
        ProviderRegistry.isinstance 检查 fail, 互换使用阻塞.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.extra_kwargs
        )

        kwargs = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        kwargs.update(kw)

        response = client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def stream(
        self,
        messages: list[dict],
        **kw,
    ):
        """满足 Protocol 的 stream(messages, **kw) 接口.

        V 6/7 08:35: Protocol 期望 stream(messages, **kw), 之前只有 stream(prompt, ...).
        内部用 OpenAI client 走 messages 模式.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.extra_kwargs
        )

        model = kw.pop("model", None) or self.default_model
        temperature = kw.pop("temperature", 0.7)
        max_tokens = kw.pop("max_tokens", None)

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kw,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def close(self) -> None:
        """满足 Protocol 的 close() 接口.

        V 6/7 08:35: 资源清理. OpenAI Python SDK v1+ 无显式 close
        (httpx client 走连接池), 这里作为 noop 满足 Protocol.
        子类 override 可加实际清理.
        """
        return None
