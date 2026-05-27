"""
L1 层：LLM 压缩层 (LCM Compressor)
从对话中提取关键事实，不存储原始对话内容。

使用百炼 API (dashscope) 进行 LLM 调用和 embedding。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, TypedDict

import httpx


# ---------------------------------------------------------------------------
# 枚举：事实类型
# ---------------------------------------------------------------------------


class FactType(str, Enum):
    """从对话中提取的事实类型枚举。"""

    PERSON = "person"  # 人物
    PROJECT = "project"  # 项目
    DATE = "date"  # 日期/时间
    DECISION = "decision"  # 决策
    PREFERENCE = "preference"  # 偏好
    FACT = "fact"  # 一般事实
    LOCATION = "location"  # 地点
    EVENT = "event"  # 事件


# ---------------------------------------------------------------------------
# 数据类：提取的事实
# ---------------------------------------------------------------------------


@dataclass
class ExtractedFact:
    """从对话中提取的单个关键事实。"""

    content: str  # 事实内容，如 "用户参加石榴籽省赛"
    fact_type: FactType  # 事实类型
    entities: list[str]  # 实体列表，如 ["石榴籽", "省赛"]
    importance: float  # 重要性 0.0~1.0
    source_turn: int  # 对话轮次（0-based）
    created_at: str  # ISO 8601 时间戳

    def to_dict(self) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "content": self.content,
            "fact_type": self.fact_type.value,
            "entities": self.entities,
            "importance": self.importance,
            "source_turn": self.source_turn,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# 百炼 API 配置
# ---------------------------------------------------------------------------


class BaiLianConfig(TypedDict):
    """百炼 API 配置。"""

    api_key: str
    base_url: str
    model: str
    embedding_model: str


DEFAULT_CONFIG: BaiLianConfig = {
    "api_key": "",
    "base_url": "https://dashscope.aliyuncs.com/api/v1",
    "model": "qwen-plus",
    "embedding_model": "text-embedding-v3",
}


# ---------------------------------------------------------------------------
# FactExtractor：使用 LLM 从对话中提取关键事实
# ---------------------------------------------------------------------------


class FactExtractor:
    """
    使用百炼 LLM 从对话历史中提取关键事实。

    示例：
        extractor = FactExtractor(api_key="your-api-key")
        facts = await extractor.extract_facts([
            "用户说今天要去参加石榴籽省赛",
            "用户提到他喜欢用 VSCode 写代码",
        ])
    """

    SYSTEM_PROMPT = """你是事实提取专家，专门从对话中提取关键信息。

从给定对话中提取所有重要事实，每条事实按以下 JSON 格式返回（不要包裹在 markdown 代码块中，直接输出 JSON 数组）：

[
  {
    "content": "事实内容描述",
    "fact_type": "person|project|date|decision|preference|fact|location|event",
    "entities": ["实体1", "实体2"],
    "importance": 0.0~1.0,
    "source_turn": 对话轮次号（0-based）
  }
]

提取规则：
- person: 涉及具体人物（姓名、称呼）
- project: 项目名称、任务、比赛等
- date: 具体日期、时间、截止日期
- decision: 做出的决定、选择
- preference: 偏好、习惯、喜欢/不喜欢
- fact: 一般性事实信息
- location: 地点、位置
- event: 事件、发生的事情

重要性评分标准：
- 0.9~1.0: 核心事实，不可丢失（项目名称、重要决策、生命事件）
- 0.6~0.8: 重要信息，值得记忆（偏好、项目进展）
- 0.3~0.5: 中等价值（一般事实）
- 0.0~0.2: 低价值，边缘信息

只输出 JSON 数组，不要其他内容。"""

    USER_PROMPT_TEMPLATE = """对话历史（共 {turn_count} 轮）：

{messages}

请提取所有关键事实："""

    def __init__(
        self,
        config: BaiLianConfig | None = None,
        *,
        timeout: float = 60.0,
    ) -> None:
        """
        初始化 FactExtractor。

        Args:
            config: 百炼 API 配置，若为 None 则使用默认配置（需自行设置 api_key）
            timeout: 请求超时时间（秒）
        """
        self._config = config or DEFAULT_CONFIG.copy()
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def api_key(self) -> str:
        return self._config["api_key"]

    @api_key.setter
    def api_key(self, value: str) -> None:
        self._config["api_key"] = value

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Authorization": f"Bearer {self._config['api_key']}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "FactExtractor":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def _call_llm(self, messages: list[str]) -> list[dict]:
        """
        调用百炼 LLM API 提取事实。

        Args:
            messages: 对话消息列表

        Returns:
            LLM 返回的原始事实字典列表
        """
        if not self.api_key:
            raise ValueError("API key 未设置，请先设置 api_key")

        user_content = self.USER_PROMPT_TEMPLATE.format(
            turn_count=len(messages),
            messages="\n".join(
                f"[轮次 {i}] {msg}" for i, msg in enumerate(messages)
            ),
        )

        payload = {
            "model": self._config["model"],
            "input": {
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
            },
            "parameters": {
                "result_format": "message",
                "temperature": 0.1,
            },
        }

        client = self._get_client()
        response = await client.post(
            f"{self._config['base_url']}/services/aigc/text-generation/generation",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        output = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")

        # 尝试解析 LLM 返回的 JSON
        try:
            # LLM 可能返回带 markdown 代码块的情况
            text = output
            if isinstance(output, dict):
                text = output.get("text", "")
            if isinstance(text, str):
                # 去掉可能的 markdown 代码块包装
                text = text.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM 返回格式错误，无法解析 JSON: {e}\n原始输出: {output}") from e

        return []

    async def extract_facts(
        self,
        messages: list[str],
        source_turn_offset: int = 0,
    ) -> list[ExtractedFact]:
        """
        从对话历史中提取关键事实。

        Args:
            messages: 对话消息列表
            source_turn_offset: 轮次起始偏移（用于多批次连续处理）

        Returns:
            ExtractedFact 实例列表
        """
        if not messages:
            return []

        raw_facts = await self._call_llm(messages)

        facts: list[ExtractedFact] = []
        now = datetime.now(timezone.utc).isoformat()

        for item in raw_facts:
            try:
                fact_type_str = item.get("fact_type", "fact")
                fact_type = FactType(fact_type_str)
            except ValueError:
                fact_type = FactType.FACT

            fact = ExtractedFact(
                content=item.get("content", "").strip(),
                fact_type=fact_type,
                entities=item.get("entities", []),
                importance=max(0.0, min(1.0, float(item.get("importance", 0.5)))),
                source_turn=item.get("source_turn", 0) + source_turn_offset,
                created_at=now,
            )
            if fact.content:
                facts.append(fact)

        return facts


# ---------------------------------------------------------------------------
# LCMCompressor：压缩层，聚合与去重
# ---------------------------------------------------------------------------


class CompressionResult(TypedDict):
    """压缩结果。"""

    facts: list[dict]
    duplicate_count: int
    new_count: int
    total_input_turns: int


class LCMCompressor:
    """
    L1 层：LLM 压缩层。

    接收对话历史 → 调用 FactExtractor 提取事实 → 避免重复 → 返回压缩后的事实列表。

    示例：
        compressor = LCMCompressor(api_key="your-api-key")
        result = await compressor.compress([
            {"role": "user", "content": "今天参加了石榴籽省赛"},
            {"role": "assistant", "content": "恭喜！"},
        ])
    """

    SIMILARITY_THRESHOLD = 0.88  # embedding 相似度阈值

    def __init__(
        self,
        config: BaiLianConfig | None = None,
        *,
        similarity_threshold: float = 0.88,
        timeout: float = 60.0,
    ) -> None:
        """
        初始化 LCMCompressor。

        Args:
            config: 百炼 API 配置
            similarity_threshold: 事实去重相似度阈值（0.0~1.0）
            timeout: HTTP 请求超时（秒）
        """
        self._config = config or DEFAULT_CONFIG.copy()
        self._similarity_threshold = similarity_threshold
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._extractor: FactExtractor | None = None

    @property
    def api_key(self) -> str:
        return self._config["api_key"]

    @api_key.setter
    def api_key(self, value: str) -> None:
        self._config["api_key"] = value

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Authorization": f"Bearer {self._config['api_key']}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """关闭所有 HTTP 连接。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._extractor is not None:
            await self._extractor.close()
            self._extractor = None

    async def __aenter__(self) -> "LCMCompressor":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def _get_extractor(self) -> FactExtractor:
        if self._extractor is None:
            self._extractor = FactExtractor(self._config, timeout=self._timeout)
        return self._extractor

    async def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        """
        调用 dashscope embedding API 获取文本向量。

        Args:
            texts: 文本列表

        Returns:
            embedding 向量列表
        """
        if not self.api_key:
            raise ValueError("API key 未设置")

        payload = {
            "model": self._config["embedding_model"],
            "input": {"texts": texts},
        }

        client = self._get_client()
        response = await client.post(
            f"{self._config['base_url']}/services/embeddings/text-embedding/text-embedding",
            json=payload,
        )
        response.raise_for_status()

        data = response.json()
        output = data.get("output", {}).get("embeddings", [])
        return [item.get("embedding", []) for item in output]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _is_duplicate(
        self,
        new_fact: ExtractedFact,
        existing_facts: list[ExtractedFact],
        existing_embeddings: list[list[float]],
    ) -> tuple[bool, float]:
        """
        检查新事实是否与已有记忆高度相似。

        Returns:
            (is_duplicate, similarity_score)
        """
        if not existing_embeddings:
            return False, 0.0

        try:
            new_embeddings = await self._get_embedding([new_fact.content])
            new_vec = new_embeddings[0]
        except Exception:
            return False, 0.0

        for i, existing in enumerate(existing_facts):
            existing_vec = existing_embeddings[i]
            sim = self._cosine_similarity(new_vec, existing_vec)
            if sim >= self._similarity_threshold:
                return True, sim

        return False, 0.0

    async def compress(
        self,
        conversation: list[dict],
        existing_facts: list[ExtractedFact] | None = None,
    ) -> CompressionResult:
        """
        压缩对话历史，提取关键事实。

        Args:
            conversation: 对话历史，每项为 {"role": "user"|"assistant", "content": "..."}
            existing_facts: 已有的记忆事实（用于去重检测），若为 None 则视为空列表

        Returns:
            CompressionResult，包含压缩后的 facts 列表及统计信息
        """
        existing_facts = existing_facts or []

        # 1. 提取消息内容
        messages = [
            str(item.get("content", "")) for item in conversation if item.get("content")
        ]
        if not messages:
            return CompressionResult(
                facts=[],
                duplicate_count=0,
                new_count=0,
                total_input_turns=0,
            )

        # 2. 调用 LLM 提取事实
        extractor = self._get_extractor()
        new_raw_facts = await extractor.extract_facts(messages)

        # 3. 去重检查
        existing_embeddings: list[list[float]] = []
        if existing_facts:
            existing_contents = [f.content for f in existing_facts]
            if existing_contents:
                try:
                    existing_embeddings = await self._get_embedding(existing_contents)
                except Exception:
                    existing_embeddings = []

        final_facts: list[ExtractedFact] = list(existing_facts)
        new_count = 0
        duplicate_count = 0

        for fact in new_raw_facts:
            is_dup, sim = await self._is_duplicate(
                fact, final_facts, existing_embeddings
            )
            if is_dup:
                duplicate_count += 1
                # 更新重要性（取max）
                for existing in final_facts:
                    if self._cosine_similarity(
                        existing_embeddings[final_facts.index(existing)]
                        if existing_embeddings
                        else [],
                        [],
                    ) >= self._similarity_threshold:
                        existing.importance = max(existing.importance, fact.importance)
            else:
                final_facts.append(fact)
                new_count += 1
                # 获取新 embedding 追加
                try:
                    new_emb = await self._get_embedding([fact.content])
                    existing_embeddings.append(new_emb[0])
                except Exception:
                    existing_embeddings.append([])

        return CompressionResult(
            facts=[f.to_dict() for f in final_facts],
            duplicate_count=duplicate_count,
            new_count=new_count,
            total_input_turns=len(messages),
        )


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


async def extract_facts_from_conversation(
    messages: list[str],
    api_key: str,
    model: str = "qwen-plus",
) -> list[ExtractedFact]:
    """
    便捷函数：从消息列表中提取事实。

    Args:
        messages: 对话消息列表
        api_key: 百炼 API key
        model: LLM 模型名称

    Returns:
        ExtractedFact 列表
    """
    config: BaiLianConfig = {
        **DEFAULT_CONFIG,
        "api_key": api_key,
        "model": model,
    }
    async with FactExtractor(config) as extractor:
        return await extractor.extract_facts(messages)


async def compress_conversation(
    conversation: list[dict],
    api_key: str,
    existing_facts: list[ExtractedFact] | None = None,
    model: str = "qwen-plus",
) -> CompressionResult:
    """
    便捷函数：压缩对话历史。

    Args:
        conversation: 对话历史
        api_key: 百炼 API key
        existing_facts: 已有事实
        model: LLM 模型名称

    Returns:
        CompressionResult
    """
    config: BaiLianConfig = {
        **DEFAULT_CONFIG,
        "api_key": api_key,
        "model": model,
    }
    async with LCMCompressor(config) as compressor:
        return await compressor.compress(conversation, existing_facts)
