"""LLM Provider Protocol.

References:
    - ARCHITECTURE.md §5.3.2 (LLMProvider Protocol)
"""

from __future__ import annotations

__all__ = ["LLMProvider", "LLMResponse"]

from typing import TYPE_CHECKING, AsyncIterator, Protocol

if TYPE_CHECKING:
    from datetime import datetime
else:
    from pydantic import BaseModel


class LLMResponse(BaseModel):  # type: ignore[valid-type]
    """LLM chat response model."""
    content: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    raw: dict | None = None


class LLMProvider(Protocol):
    """Protocol for LLM providers.

    All LLM backends (OpenAI, Anthropic, local, etc.) must implement this.
    """
    name: str

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
        """Send a chat completion request."""
        ...

    async def stream(
        self,
        messages: list[dict],
        **kw,
    ) -> AsyncIterator[str]:
        """Stream chat response tokens."""
        ...

    async def close(self) -> None:
        """Clean up provider resources."""
        ...
