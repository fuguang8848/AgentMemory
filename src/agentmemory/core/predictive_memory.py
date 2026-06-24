"""
Predictive Memory — 基于访问模式的预测性记忆

梁文峰"问题定义"视角:
  传统记忆系统解决"如何检索过去"(搜索问题)
  预测性记忆解决"未来需要什么"(预测问题)

LeCun JEPA 视角:
  不是生成过去记忆的完整内容(生成式)
  而是学习"记忆状态转移"的表征(对比式)

本质问题: 记忆的价值不在于"存储"而在于"提前准备"
"""

from __future__ import annotations

import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class AccessPattern:
    """访问模式 — 学习记忆被使用的规律"""
    memory_id: str
    access_count: int = 0
    last_access: float = 0.0
    access_intervals: list[float] = field(default_factory=list)  # 最近访问间隔
    co_access_mems: list[str] = field(default_factory=list)     # 同时被访问的记忆

    # JEPA 预测表征: 不是存储记忆内容，而是存储"访问状态转移"
    # 状态 = (time_bucket, recent_accesses, context)
    # 转移 = P(next_memory | current_state)
    predicted_next: Optional[str] = None
    confidence: float = 0.0


class PredictiveMemory:
    """
    预测性记忆 — 学习访问模式，预测未来需求

    不是"更好的搜索"，而是"提前知道需要什么"

    JEPA 对比式学习视角:
    - 不是重建记忆内容(生成式)
    - 而是学习状态转移表征(对比式)
    - 表征 = (memory_id, access_time, context) → 预测下一个 memory_id
    """

    # 梁文峰工程极限: 用有限资源做最强性能
    MAX_PATTERNS = 5000          # 最多跟踪5000个记忆的访问模式
    TIME_BUCKETS = 24            # 24小时时间桶
    COACCESS_WINDOW = 3           # 3次访问内算"同时访问"
    PREDICTION_HORIZON = 5       # 预测未来5个可能的记忆

    def __init__(self, max_patterns: int = MAX_PATTERNS):
        self._patterns: Dict[str, AccessPattern] = {}
        self._max_patterns = max_patterns
        self._recent_access_order: list[tuple[str, float]] = []  # (memory_id, timestamp)
        self._coaccess_graph: Dict[str, set[str]] = defaultdict(set)  # 共同访问图

        # JEPA 状态转移矩阵: state_key → {next_mem: count}
        # state_key = (hour, recent_mem_ids)
        self._transition_matrix: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_access(self, memory_id: str, timestamp: float = None) -> None:
        """
        记录一次记忆访问，学习访问模式

        梁文峰本质问题: 访问即数据，不是存储即记忆
        """
        if timestamp is None:
            timestamp = time.time()

        pattern = self._patterns.get(memory_id)
        if pattern is None:
            if len(self._patterns) >= self._max_patterns:
                self._evict_low_value_pattern()
            pattern = AccessPattern(memory_id=memory_id)
            self._patterns[memory_id] = pattern

        # 更新访问计数和时间
        now = time.time()
        if pattern.last_access > 0:
            interval = now - pattern.last_access
            if interval > 0:
                pattern.access_intervals.append(interval)
                if len(pattern.access_intervals) > 10:
                    pattern.access_intervals.pop(0)  # 保留最近10个间隔

        pattern.access_count += 1
        pattern.last_access = now

        # 更新最近访问顺序(用于共同访问图)
        self._recent_access_order.append((memory_id, now))
        if len(self._recent_access_order) > 100:
            self._recent_access_order.pop(0)

        # 更新共同访问图
        self._update_coaccess(memory_id)

        # 更新状态转移矩阵(JEPA核心)
        self._update_transition(memory_id)

    def _update_coaccess(self, memory_id: str) -> None:
        """更新共同访问图 — 识别经常一起被访问的记忆"""
        # 找到最近 COACCESS_WINDOW 次访问范围内的其他记忆
        window_start = max(0, len(self._recent_access_order) - self.COACCESS_WINDOW - 1)
        recent = self._recent_access_order[window_start:]

        for other_mem, _ in recent:
            if other_mem != memory_id:
                self._coaccess_graph[memory_id].add(other_mem)
                self._coaccess_graph[other_mem].add(memory_id)

    def _update_transition(self, memory_id: str) -> None:
        """
        更新状态转移矩阵 — JEPA 对比式学习核心

        状态 = (hour_bucket, recent_mem_1, recent_mem_2)
        转移 = P(current_mem → next_mem | state)
        """
        # 获取当前时间桶
        now = time.time()
        hour_bucket = int((now % 86400) / 3600)  # 0-23

        # 获取最近访问的2个记忆作为上下文
        recent_mems = [
            m for m, t in self._recent_access_order[-3:-1]  # 最近2个(不含当前)
            if t < now - 1  # 排除同一秒内的重复
        ]

        if len(recent_mems) >= 1:
            state_key = (hour_bucket, tuple(recent_mems[-2:]))  # (hour, context_mem_1, context_mem_2)
            self._transition_matrix[state_key][memory_id] += 1

    def predict_next(self, current_context: list[str] = None, limit: int = None) -> list[tuple[str, float]]:
        """
        预测下一个最可能访问的记忆

        Returns:
            list of (memory_id, probability) 按概率降序

        JEPA 对比式预测:
        - 不是生成记忆内容(生成式)
        - 而是预测访问状态转移(对比式)
        """
        if limit is None:
            limit = self.PREDICTION_HORIZON

        now = time.time()
        hour_bucket = int((now % 86400) / 3600)

        # 获取当前上下文记忆
        if current_context is None:
            # 使用最近访问作为上下文
            current_context = [m for m, t in self._recent_access_order[-3:] if t > now - 300]

        # 查询状态转移矩阵
        candidates: Dict[str, float] = defaultdict(float)

        for mem in current_context:
            # 直接转移: P(next | current, hour)
            state_key = (hour_bucket, (mem,))
            for next_mem, count in self._transition_matrix[state_key].items():
                candidates[next_mem] += count * 2.0  # 直接转移权重高

            # 共同访问: 经常一起访问的记忆
            for coaccess_mem in self._coaccess_graph.get(mem, []):
                candidates[coaccess_mem] += 1.0

            # 反向转移: 从当前记忆预测
            for state_key, transitions in self._transition_matrix.items():
                if mem in state_key[1]:  # mem在上下文中
                    for next_mem, count in transitions.items():
                        candidates[next_mem] += count * 0.5

        # 归一化为概率
        total = sum(candidates.values())
        if total == 0:
            return []

        sorted_candidates = sorted(
            [(m, c / total) for m, c in candidates.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_candidates[:limit]

    def get_anticipatory_score(self, memory_id: str) -> tuple[float, str]:
        """
        获取记忆的"前瞻性得分"

        Returns:
            (score, reason) - score越高越可能需要被预取

        梁文峰本质问题: 决定"哪些记忆值得优先保持活跃"
        """
        pattern = self._patterns.get(memory_id)
        if pattern is None:
            return 0.0, "no_access_pattern"

        now = time.time()

        # 因素1: 访问频率
        freq_score = min(pattern.access_count / 10.0, 1.0) * 0.3

        # 因素2: 访问规律性(间隔方差小=规律)
        if len(pattern.access_intervals) >= 2:
            intervals = pattern.access_intervals[-5:]
            mean_interval = sum(intervals) / len(intervals)
            variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
            regularity = 1.0 / (1.0 + math.sqrt(variance) / max(mean_interval, 1))
        else:
            regularity = 0.5
        regularity_score = regularity * 0.3

        # 因素3: 最近访问时间(太旧=可能被遗忘)
        if pattern.last_access > 0:
            hours_since = (now - pattern.last_access) / 3600
            recency_score = math.exp(-hours_since / 24) * 0.2
        else:
            recency_score = 0.0

        # 因素4: 预测置信度
        pred_score = pattern.confidence * 0.2

        total = freq_score + regularity_score + recency_score + pred_score

        reasons = []
        if freq_score > 0.2:
            reasons.append(f"freq={pattern.access_count}")
        if regularity_score > 0.2:
            reasons.append(f"regular={regularity:.2f}")
        if recency_score > 0.1:
            reasons.append(f"recent_h={hours_since:.1f}")

        return total, ",".join(reasons) if reasons else "low_activity"

    def prefetch_hint(self) -> list[str]:
        """
        返回应该被预取的记忆ID列表

        梁文峰工程极限: 用最少计算预测最有价值的预取
        """
        # 获取所有有访问模式的记忆
        scored = []
        for mem_id in self._patterns:
            score, reason = self.get_anticipatory_score(mem_id)
            if score > 0.1:  # 阈值
                scored.append((mem_id, score, reason))

        # 按前瞻性得分排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 返回top 10
        return [mem_id for mem_id, _, _ in scored[:10]]

    def _evict_low_value_pattern(self) -> None:
        """驱逐最低价值的模式(LRU变体)"""
        if not self._patterns:
            return

        now = time.time()
        # 驱逐: 从未被访问 + 最老的
        candidates = [
            (mem_id, pattern.last_access if pattern.last_access > 0 else -1)
            for mem_id, pattern in self._patterns.items()
            if pattern.access_count == 0
        ]

        if candidates:
            candidates.sort(key=lambda x: x[1])
            evict_id = candidates[0][0]
            del self._patterns[evict_id]
        else:
            # 所有都被访问过，找最不活跃的
            candidates = [
                (mem_id, pattern.access_count / max(now - pattern.last_access, 1))
                for mem_id, pattern in self._patterns.items()
            ]
            candidates.sort(key=lambda x: x[1])
            evict_id = candidates[0][0]
            del self._patterns[evict_id]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "tracked_memories": len(self._patterns),
            "coaccess_edges": sum(len(v) for v in self._coaccess_graph.values()) // 2,
            "transition_states": len(self._transition_matrix),
            "recent_accesses": len(self._recent_access_order),
        }
