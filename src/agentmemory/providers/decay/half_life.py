"""
Half-Life Decay Policy
Implements DecayPolicy Protocol
"""

import math
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DecayResult:
    """Decay calculation result"""
    original_score: float
    decayed_score: float
    half_lives_elapsed: float
    metadata: dict[str, Any] | None = None


class DecayPolicy(ABC):
    """Abstract base class for decay policies"""

    @abstractmethod
    def calculate(self, score: float, timestamp: float, current_time: float | None = None, **kwargs) -> DecayResult:
        """Calculate decayed score"""
        raise NotImplementedError

    @abstractmethod
    def get_half_life(self) -> float:
        """Get half-life in seconds"""
        raise NotImplementedError


class HalfLifeDecay(DecayPolicy):
    """
    Half-life based exponential decay policy.

    Decay formula: score(t) = score_0 * (0.5 ^ (elapsed_time / half_life))

    The score halves every half_life period.
    """

    def __init__(
        self,
        half_life_seconds: float = 86400 * 7,  # Default: 1 week
        min_score: float = 0.0,
        max_score: float = 1.0,
        **kwargs
    ):
        """
        Args:
            half_life_seconds: Time for score to decay to half (default: 1 week)
            min_score: Minimum possible score (floor)
            max_score: Maximum possible score (ceiling)
        """
        self.half_life_seconds = half_life_seconds
        self.min_score = min_score
        self.max_score = max_score
        self.kwargs = kwargs

    def get_half_life(self) -> float:
        """Get half-life in seconds"""
        return self.half_life_seconds

    def calculate(
        self,
        score: float,
        timestamp: float,
        current_time: float | None = None,
        **kwargs
    ) -> DecayResult:
        """
        Calculate decayed score using exponential decay.

        Args:
            score: Original score (0-1 typically)
            timestamp: Original timestamp (Unix epoch)
            current_time: Current timestamp (default: now via time.time())

        Returns:
            DecayResult with original and decayed scores
        """
        import time

        if current_time is None:
            current_time = time.time()

        elapsed = current_time - timestamp

        # Calculate half-lives elapsed
        if self.half_life_seconds <= 0:
            half_lives_elapsed = 0.0
        else:
            half_lives_elapsed = elapsed / self.half_life_seconds

        # Exponential decay: score * (0.5 ^ half_lives)
        decay_factor = math.pow(0.5, half_lives_elapsed)
        decayed_score = score * decay_factor

        # Clamp to min/max
        decayed_score = max(self.min_score, min(self.max_score, decayed_score))

        return DecayResult(
            original_score=score,
            decayed_score=decayed_score,
            half_lives_elapsed=half_lives_elapsed,
            metadata={
                "elapsed_seconds": elapsed,
                "half_life_seconds": self.half_life_seconds,
                "decay_factor": decay_factor
            }
        )

    def get_score_at_time(
        self,
        score: float,
        timestamp: float,
        target_time: float
    ) -> float:
        """
        Get the decayed score at a specific target time.

        Args:
            score: Original score
            timestamp: Original timestamp
            target_time: Target time to calculate score at

        Returns:
            Decayed score at target time
        """
        elapsed = target_time - timestamp

        if self.half_life_seconds <= 0:
            return score

        half_lives_elapsed = elapsed / self.half_life_seconds
        decay_factor = math.pow(0.5, half_lives_elapsed)
        decayed_score = score * decay_factor

        return max(self.min_score, min(self.max_score, decayed_score))
