"""
AgentMemory v2.0 - RRF Fusion（Reciprocal Rank Fusion）

双轨检索融合器：向量轨 × 图书馆轨 × Tag轨

RRF 公式：
    RRF_score(d) = Σ 1/(k + rank_i(d))

其中：
- d = 文档（memory_id）
- k = 衰减参数（通常 k=60）
- rank_i(d) = 文档在第 i 个检索结果中的排名
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """单条检索结果"""
    memory_id: str
    score: float
    rank: int
    source: str  # "vector" | "library" | "tag"


@dataclass
class FusionResult:
    """融合后的检索结果"""
    memory_id: str
    rrf_score: float
    ranks: dict[str, int]  # {"vector": 1, "library": 3}
    details: dict[str, float]  # {"vector_score": 0.85, "library_score": 0.72}


class RRFusion:
    """
    Reciprocal Rank Fusion 融合器

    将多路检索结果按 RRF 公式融合。
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF 衰减参数。k 越大，高排名结果优势越明显。
               k=60 是 standard RRF（参考 BM25S 论文）
        """
        self.k = k

    def fuse(
        self,
        vector_results: list[RankedResult] = None,
        library_results: list[RankedResult] = None,
        tag_results: list[RankedResult] = None,
    ) -> list[FusionResult]:
        """
        融合多路检索结果。

        任一路可为空（表示该路没有结果）。

        Args:
            vector_results: 向量轨检索结果
            library_results: 图书馆轨检索结果
            tag_results: Tag 轨检索结果

        Returns:
            FusionResult 列表，按 RRF 分数降序
        """
        # 构建各轨的 ranked lists
        ranked_lists: list[list[RankedResult]] = []
        track_names: list[str] = []

        if vector_results:
            ranked_lists.append(vector_results)
            track_names.append("vector")
        if library_results:
            ranked_lists.append(library_results)
            track_names.append("library")
        if tag_results:
            ranked_lists.append(tag_results)
            track_names.append("tag")

        if not ranked_lists:
            return []

        # 计算 RRF 分数
        rrf_scores = self._compute_rrf(ranked_lists)

        # 构建融合结果
        fusion_results: dict[str, FusionResult] = {}

        # 处理每一轨的结果
        for track_idx, track_name in enumerate(track_names):
            results = ranked_lists[track_idx]
            for result in results:
                memory_id = result.memory_id

                if memory_id not in fusion_results:
                    fusion_results[memory_id] = FusionResult(
                        memory_id=memory_id,
                        rrf_score=rrf_scores.get(memory_id, 0.0),
                        ranks={},
                        details={},
                    )

                fusion_results[memory_id].ranks[track_name] = result.rank
                fusion_results[memory_id].details[f"{track_name}_score"] = result.score

        # 排序并返回
        sorted_results = sorted(
            fusion_results.values(),
            key=lambda x: x.rrf_score,
            reverse=True
        )

        return sorted_results

    def _compute_rrf(self, ranked_lists: list[list[RankedResult]]) -> dict[str, float]:
        """
        计算所有文档的 RRF 分数。

        标准 RRF: score(d) = Σ 1/(k + rank_i(d))

        Args:
            ranked_lists: 各轨的检索结果列表

        Returns:
            dict[memory_id, rrf_score]
        """
        rrf_scores: dict[str, float] = {}

        for ranked_list in ranked_lists:
            for result in ranked_list:
                memory_id = result.memory_id
                # RRF 公式: score(d) = Σ 1/(k + rank_i(d)), rank_i 从 1 开始
                # RankedResult.rank 是 1-based 排名，直接使用
                rank_position = result.rank

                rrf = 1.0 / (self.k + rank_position)

                if memory_id not in rrf_scores:
                    rrf_scores[memory_id] = 0.0
                rrf_scores[memory_id] += rrf

        return rrf_scores

    def _normalize_scores(self, results: list[RankedResult]) -> list[RankedResult]:
        """
        将原始分数归一化到 [0, 1]

        Args:
            results: 待归一化的结果列表

        Returns:
            归一化后的结果列表
        """
        if not results:
            return results

        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)

        if max_score == min_score:
            # 所有分数相同，保持原样
            return results

        range_score = max_score - min_score
        normalized = []
        for r in results:
            normalized_score = (r.score - min_score) / range_score
            normalized.append(
                RankedResult(
                    memory_id=r.memory_id,
                    score=normalized_score,
                    rank=r.rank,
                    source=r.source
                )
            )

        return normalized


def rrf_weighted(
    results_by_track: dict[str, list[RankedResult]],
    k: int = 60,
    weights: dict[str, float] = None,
) -> dict[str, float]:
    """
    加权 RRF 融合。

    score(d) = Σ w_i * 1/(k + rank_i(d))
    其中 w_i 是第 i 轨的权重

    Args:
        results_by_track: 各轨的检索结果字典
        k: RRF 衰减参数
        weights: 各轨的权重字典，默认为 1.0

    Returns:
        dict[memory_id, weighted_rrf_score]
    """
    if weights is None:
        weights = {track: 1.0 for track in results_by_track}

    rrf_scores: dict[str, float] = {}

    for track, results in results_by_track.items():
        weight = weights.get(track, 1.0)

        for i, result in enumerate(results):
            # rank 从 1 开始
            rank_position = i + 1
            rrf = weight * (1.0 / (k + rank_position))

            if result.memory_id not in rrf_scores:
                rrf_scores[result.memory_id] = 0.0
            rrf_scores[result.memory_id] += rrf

    return rrf_scores
