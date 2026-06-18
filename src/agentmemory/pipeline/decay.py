"""Decay Pipeline - half-life forgetting with DecayPolicy.

References:
    - ARCHITECTURE.md §10.6 (lines 1596-1614)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..core.types import MemoryItem, MemoryLayer

_BATCH_SIZE = 1000


class DecayAction(str, Enum):
    """Decay pipeline action decision."""
    KEEP = "keep"
    ARCHIVE = "archive"  # Move to L4 archive
    FORGET = "forget"    # Delete


class DecayPolicy(ABC):
    """Abstract base class for decay scoring policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name."""
        ...

    @abstractmethod
    async def score(self, item: MemoryItem) -> float:
        """Calculate decay score for an item.

        Args:
            item: MemoryItem to score

        Returns:
            Score from 0.0 to 1.0
            - < 0.3: forget (delete)
            - 0.3-0.5: archive (move to L4)
            - >= 0.5: keep
        """
        ...

    @abstractmethod
    async def decide(self, item: MemoryItem) -> DecayAction:
        """Decide action based on score.

        Args:
            item: MemoryItem to evaluate

        Returns:
            DecayAction: KEEP, ARCHIVE, or FORGET
        """
        ...


class HalfLifePolicy(DecayPolicy):
    """Classic half-life decay policy (1.x style).

    Score = importance * (0.5 ^ (age_in_days / half_life_days))
    """

    def __init__(self, half_life_days: float = 30.0, base_importance: float = 0.5):
        """Initialize HalfLifePolicy.

        Args:
            half_life_days: Number of days for importance to halve
            base_importance: Default importance if item.importance is very low
        """
        self._half_life_days = half_life_days
        self._base_importance = base_importance

    @property
    def name(self) -> str:
        return f"half_life_{self._half_life_days}d"

    async def score(self, item: MemoryItem) -> float:
        """Calculate half-life decay score."""
        import math

        # Base importance from item
        importance = item.importance if item.importance > 0 else self._base_importance

        # Calculate age in days
        created = item.created_at
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        age = datetime.utcnow() - created
        age_days = age.total_seconds() / 86400

        # Half-life decay formula
        score = importance * math.pow(0.5, age_days / self._half_life_days)

        # Factor in access count (recent access boosts score)
        if item.last_accessed_at is not None:
            last_access = item.last_accessed_at
            if isinstance(last_access, str):
                last_access = datetime.fromisoformat(last_access)
            access_age = datetime.utcnow() - last_access
            access_age_days = access_age.total_seconds() / 86400
            # More recent access = higher boost
            access_boost = math.exp(-access_age_days / self._half_life_days) * item.access_count * 0.1
            score = min(1.0, score + access_boost)

        return max(0.0, min(1.0, score))

    async def decide(self, item: MemoryItem) -> DecayAction:
        """Decide based on score thresholds."""
        score = await self.score(item)
        if score < 0.3:
            return DecayAction.FORGET
        elif score < 0.5:
            return DecayAction.ARCHIVE
        return DecayAction.KEEP


class ImportanceOnlyPolicy(DecayPolicy):
    """Mem0-style importance-only policy.

    Score based purely on importance, ignores age.
    """

    def __init__(self, forget_threshold: float = 0.2, archive_threshold: float = 0.35):
        self._forget_threshold = forget_threshold
        self._archive_threshold = archive_threshold

    @property
    def name(self) -> str:
        return "importance_only"

    async def score(self, item: MemoryItem) -> float:
        """Return item importance as score."""
        return max(0.0, min(1.0, item.importance))

    async def decide(self, item: MemoryItem) -> DecayAction:
        """Decide based on importance thresholds."""
        score = await self.score(item)
        if score < self._forget_threshold:
            return DecayAction.FORGET
        elif score < self._archive_threshold:
            return DecayAction.ARCHIVE
        return DecayAction.KEEP


class DecayPipeline:
    """Half-life forgetting pipeline.

    Scans all items in batches, applies DecayPolicy to decide
    keep/archive/forget, and executes the action.
    """

    def __init__(
        self,
        policy: DecayPolicy | None = None,
        batch_size: int = _BATCH_SIZE,
        archive_layer: MemoryLayer = MemoryLayer.L4_FILE,
    ):
        """Initialize DecayPipeline.

        Args:
            policy: DecayPolicy instance (default: HalfLifePolicy)
            batch_size: Items to process per batch
            archive_layer: Target layer for archived items
        """
        self.policy = policy or HalfLifePolicy()
        self.batch_size = batch_size
        self.archive_layer = archive_layer

    async def scan_and_decay(
        self,
        items: list[MemoryItem],
        delete_fn: callable | None = None,
        archive_fn: callable | None = None,
    ) -> dict[str, int]:
        """Scan items and apply decay policy.

        Args:
            items: List of all MemoryItem to evaluate
            delete_fn: Optional async function(item) to delete
            archive_fn: Optional async function(item, target_layer) to archive

        Returns:
            Dict with counts: {"kept": N, "archived": N, "forgotten": N}
        """
        stats = {"kept": 0, "archived": 0, "forgotten": 0}

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]

            # Process each item in batch with error isolation
            for item in batch:
                try:
                    action = await self.policy.decide(item)

                    if action == DecayAction.FORGET:
                        if delete_fn is not None:
                            await delete_fn(item)
                        stats["forgotten"] += 1
                    elif action == DecayAction.ARCHIVE:
                        if archive_fn is not None:
                            await archive_fn(item, self.archive_layer)
                        stats["archived"] += 1
                    else:
                        stats["kept"] += 1

                except Exception:
                    # Best effort per-item, don't fail entire batch
                    stats["kept"] += 1

        return stats

    async def score_item(self, item: MemoryItem) -> float:
        """Score a single item.

        Args:
            item: MemoryItem to score

        Returns:
            Decay score 0.0 to 1.0
        """
        return await self.policy.score(item)

    async def decide_action(self, item: MemoryItem) -> DecayAction:
        """Decide action for a single item.

        Args:
            item: MemoryItem to evaluate

        Returns:
            DecayAction to take
        """
        return await self.policy.decide(item)
