"""GeometricDecayPolicy — 远端 DecayEngine 适配到本地 DecayPolicy ABC

远端 v2.0.1 公式:
    score = (log(1+access))^0.3 × importance^0.4 × recency^0.3
    recency = 0.5 ** (age_days / half_life_days)

本地 DecayPolicy ABC 要求:
    - name 属性
    - async score(item: MemoryItem) -> float [0, 1]
    - async decide(item: MemoryItem) -> DecayAction [KEEP/ARCHIVE/FORGET]

差异:
    - 远端: 同步 score() 返回 DecayScore（含 components + reasons）
    - 本地: 异步 score() 返回 float
    - 阈值: 远端 0.2/0.5 (forget/archive), 本地 0.3/0.5
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

# 本地 ABC
from agentmemory.pipeline.decay import DecayAction, DecayPolicy
from agentmemory.core.types import MemoryItem


# 远端 DecayEngine（Phase 1 独立模块）
try:
    from agentmemory.extensions.v2.decay_engine import (
        DecayEngine,
        DecayPolicy as YintaDecayPolicyCfg,
    )
    YINTA_OK = True
    _IMPORT_ERROR: str | None = None
except Exception as e:
    YINTA_OK = False
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"
    DecayEngine = None  # type: ignore
    YintaDecayPolicyCfg = None  # type: ignore


class GeometricDecayPolicy(DecayPolicy):
    """几何乘积衰减策略（v2.0.1 公式）适配到本地 ABC。

    Args:
        weight_access: log(1+access) 指数，默认 0.3
        weight_importance: importance 指数，默认 0.4
        weight_recency: recency 指数，默认 0.3
        half_life_days: 半衰期，默认 30
        forget_threshold: 低于此值遗忘，默认 0.2
        archive_threshold: 低于此值归档，默认 0.5
    """

    def __init__(
        self,
        weight_access: float = 0.3,
        weight_importance: float = 0.4,
        weight_recency: float = 0.3,
        half_life_days: float = 30.0,
        forget_threshold: float = 0.3,
        archive_threshold: float = 0.5,
        # 远端额外参数
        access_count_attr: str = "access_count",
        last_accessed_attr: str = "last_accessed",
    ):
        if not YINTA_OK:
            raise ImportError(
                f"远端 DecayEngine 不可用: {_IMPORT_ERROR}。"
                "检查 agentmemory.extensions.v2.decay_engine 是否正确移植。"
            )
        # 用远端 DecayPolicyCfg 复用其类型校验
        self._cfg = YintaDecayPolicyCfg(
            weight_access=weight_access,
            weight_importance=weight_importance,
            weight_recency=weight_recency,
            half_life_days=half_life_days,
            forget_threshold=forget_threshold,
            archive_threshold=archive_threshold,
        )
        self._engine = DecayEngine(policy=self._cfg)
        # 用于 score() 适配
        self._access_count_attr = access_count_attr
        self._last_accessed_attr = last_accessed_attr

    @property
    def name(self) -> str:
        return f"geometric_v2.0.1_{self._cfg.half_life_days}d"

    async def score(self, item: MemoryItem) -> float:
        """适配: 远端 score() 同步 → 本地 async + MemoryItem 字段映射"""
        # 本地 MemoryItem → 远端 entry dict
        access_count = getattr(item, self._access_count_attr, 0) or 0
        importance = item.importance
        last_accessed = getattr(item, self._last_accessed_attr, None) or item.created_at
        created_at = item.created_at
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        # 直接复用远端公式 (内联实现，避免远端 entry schema 限制)
        if not isinstance(last_accessed, datetime):
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            else:
                last_accessed = created_at

        age_days = max(0.0, (datetime.utcnow() - last_accessed).total_seconds() / 86400)
        access_factor = math.log1p(access_count) ** self._cfg.weight_access
        importance_factor = importance ** self._cfg.weight_importance
        recency_factor = (0.5 ** (age_days / self._cfg.half_life_days)) ** self._cfg.weight_recency
        score = access_factor * importance_factor * recency_factor
        return min(1.0, max(0.0, score))

    async def decide(self, item: MemoryItem) -> DecayAction:
        """根据 score() 决定 KEEP / ARCHIVE / FORGET"""
        s = await self.score(item)
        if s < self._cfg.forget_threshold:
            return DecayAction.FORGET
        if s < self._cfg.archive_threshold:
            return DecayAction.ARCHIVE
        return DecayAction.KEEP
