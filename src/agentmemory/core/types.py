"""Core data types for AgentMemory 2.0.

References:
    - ARCHITECTURE.md §5.2 (lines 483-540)
"""

from __future__ import annotations

__all__ = [
    "MemoryItem",
    "SearchQuery",
    "SearchResult",
    "Episode",
    "MemoryType",
    "MemoryLayer",
]

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    """Memory type taxonomy."""
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    REFLECTIVE = "reflective"
    USER = "user"


class MemoryLayer(str, Enum):
    """Memory layer taxonomy (L0-L5)."""
    L0_CORE = "L0"       # in-context core memory
    L1_COMPRESS = "L1"   # raw extraction output
    L2_GRAPH = "L2"      # entity-relation
    L3_VECTOR = "L3"     # vector + bm25
    L4_FILE = "L4"       # file archive
    L5_AUDIT = "L5"      # audit log (new in 2.0)


class MemoryItem(BaseModel):
    """2.0 统一记忆条目。

    1.x 的 MemoryEntry / ExtractedFact / DiaryEntry 都收敛到这里。
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    layer: MemoryLayer = MemoryLayer.L3_VECTOR
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source: str = "user"           # user / system / inference / reflection
    source_turn: int | None = None
    tenant_id: str = "default"
    namespace: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime | None = None
    access_count: int = 0
    decay_score: float | None = None
    valid_from: datetime | None = None    # M3 时序图
    valid_to: datetime | None = None
    supersedes: list[str] = Field(default_factory=list)  # M3 反思时用


class SearchQuery(BaseModel):
    """Search query model."""
    text: str
    top_k: int = 5
    strategy: list[str] = Field(default_factory=lambda: ["vector", "bm25", "importance"])
    filter_type: list[MemoryType] | None = None
    filter_layer: list[MemoryLayer] | None = None
    filter_tags: list[str] | None = None
    tenant_id: str | None = None
    namespace: str | None = None
    min_score: float = 0.0
    rerank: bool = False
    as_of: datetime | None = None   # M3 时序查询
    graph_hops: int = 0             # 0=不做图遍历


class SearchResult(BaseModel):
    """Search result with scoring metadata."""
    item: MemoryItem
    score: float
    layer: MemoryLayer
    sources: list[str] = Field(default_factory=list)   # ["vector", "bm25", "graph"]
    explanation: dict[str, float] = Field(default_factory=dict)  # 各路分项


class Episode(BaseModel):
    """一段连续对话（M2 引入，参考 Zep）。

    A contiguous conversation session.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[dict[str, str]]
    facts: list[MemoryItem] = Field(default_factory=list)
    summary: str | None = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    tenant_id: str = "default"
