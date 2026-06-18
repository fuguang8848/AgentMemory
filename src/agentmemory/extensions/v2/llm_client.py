"""
LLM Client - 统一调用层
自动识别当前配置的模型，支持多 provider (minimax/bailian/openai-compatible)
"""

import json
import os
import httpx
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict


class LLMClient:
    """
    统一 LLM 调用接口，自动识别 provider 和模型。
    
    支持：
    - minimax (anthropic-messages API)
    - bailian (openai-completions API)
    - openai-compatible (openai-chat API)
    
    自动从环境变量/配置文件读取 API Key 和 base URL。
    """

    # Provider 优先级（当前 session 的 primary model 优先）
    PROVIDER_PRECEDENCE = ["minimax", "bailian"]

    # 各 provider 的 API 风格
    PROVIDER_APIS = {
        "minimax": "anthropic-messages",
        "bailian": "openai-completions",
        "openai-compatible": "openai-chat",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """
        初始化 LLM Client。
        
        Args:
            model: 模型 ID（如 "minimax/MiniMax-M2.7-highspeed" 或 "qwen3.6-plus"）
                  若为 None，自动从环境变量/配置读取
            api_key: API Key，若为 None 则从环境变量读取
            base_url: API Base URL，若为 None 则使用 provider 默认
            provider: 指定 provider，可选
        """
        self.model = model or self._detect_model()
        self.api_key = api_key or self._detect_api_key()
        # provider 必须先于 base_url 检测（base_url 依赖 provider）
        self.provider = provider or self._detect_provider()
        self.base_url = base_url or self._detect_base_url()
        self._client: Optional[httpx.AsyncClient] = None

    def _detect_model(self) -> str:
        """从环境变量获取模型，默认 minimax（当前 session primary）"""
        # 优先级：显式设置 > MINIMAX_API_KEY > BAILIAN_API_KEY
        if os.environ.get("MINIMAX_API_KEY"):
            return "minimax/MiniMax-M2.7-highspeed"
        if os.environ.get("BAILIAN_API_KEY"):
            return "bailian/qwen3.6-plus"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai/gpt-4o"
        # 默认：当前 session 的 primary model
        return "minimax/MiniMax-M2.7-highspeed"

    def _detect_api_key(self) -> str:
        """检测 API Key"""
        # minimax
        if os.environ.get("MINIMAX_API_KEY"):
            return os.environ["MINIMAX_API_KEY"]
        # bailian (百炼)
        if os.environ.get("BAILIAN_API_KEY"):
            return os.environ["BAILIAN_API_KEY"]
        if os.environ.get("DASHSCOPE_API_KEY"):
            return os.environ["DASHSCOPE_API_KEY"]
        # openai
        if os.environ.get("OPENAI_API_KEY"):
            return os.environ["OPENAI_API_KEY"]
        return ""

    def _detect_base_url(self) -> str:
        """检测 Base URL"""
        if self.provider == "minimax":
            return "https://api.minimaxi.com/anthropic"
        if self.provider == "bailian":
            return "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        if self.provider == "openai-compatible":
            return "https://api.openai.com/v1"
        # 默认 minimax
        return "https://api.minimaxi.com/anthropic"

    def _detect_provider(self) -> str:
        """检测 provider"""
        if self.model.startswith("minimax/"):
            return "minimax"
        if self.model.startswith("bailian/"):
            return "bailian"
        if self.model.startswith("openai/"):
            return "openai-compatible"
        # 从 API key 推断
        if os.environ.get("MINIMAX_API_KEY"):
            return "minimax"
        if os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"):
            return "bailian"
        return "minimax"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        发送对话请求。
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            model: 覆盖默认模型
            temperature: 温度
            max_tokens: 最大 token 数
        
        Returns:
            LLMResponse(content, model, usage)
        """
        target_model = model or self.model
        target_provider = target_model.split("/")[0] if "/" in target_model else self.provider

        if target_provider == "minimax":
            return await self._chat_minimax(messages, target_model, temperature, max_tokens)
        elif target_provider == "bailian":
            return await self._chat_bailian(messages, target_model, temperature, max_tokens)
        else:
            return await self._chat_openai_compat(messages, target_model, temperature, max_tokens)

    async def _chat_minimax(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Minimax (anthropic-messages API)"""
        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model.replace("minimax/", ""),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = await client.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["content"][0]["text"],
            model=model,
            usage=data.get("usage", {}),
        )

    async def _chat_bailian(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """Bailian (dashscope, openai-compatible API)"""
        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model.replace("bailian/", ""),
            "input": {"messages": messages},
            "parameters": {
                "result_format": "message",
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["parameters"]["max_tokens"] = max_tokens

        response = await client.post(
            f"{self.base_url}/services/aigc/text-generation/generation",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["output"]["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
        )

    async def _chat_openai_compat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> LLMResponse:
        """OpenAI-compatible (openai-chat API)"""
        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model.replace("openai/", ""),
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=model,
            usage=data.get("usage", {}),
        )

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Embedding Client - 向量化
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """
    统一 Embedding 接口，支持 dashscope/OpenAI-compatible。
    自动从环境变量读取配置。
    """

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: int = 1024,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.dimensions = dimensions
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        """
        获取文本的 embedding 向量。
        
        Args:
            texts: 单个文本或文本列表
        
        Returns:
            单个向量 list[float] 或多个向量 list[list[float]]
        """
        client = self._get_client()
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts

        payload = {
            "model": self.model,
            "input": {"texts": input_texts},
            "parameters": {"encoding_format": "float"},
        }

        response = await client.post(
            f"{self.base_url}/services/embeddings/text-embedding/embedding",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        outputs = data["output"]["embeddings"]
        vectors = [item["embedding"] for item in outputs]

        return vectors[0] if is_single else vectors

    async def embed_async(self, text: str) -> list[float]:
        """异步获取单个文本的 embedding"""
        result = await self.embed(text)
        return result if isinstance(result, list) else result[0]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "EmbeddingClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()