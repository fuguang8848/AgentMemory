"""
DreamNet: AI 梦境记忆系统
=============================

五层记忆架构 + 三相梦境周期 + 遗忘曲线 + 知识图谱

参考设计：
- LeoYeAI/openclaw-auto-dream (⭐556): 重要性评分 × 遗忘曲线 × 5层记忆
- ldclabs/anda-brain (⭐68): 神经符号记忆 + 知识图谱 + 自进化
- cerebralos-org/cerebralos (⭐9): 潜意识层 + 主动遗忘

核心概念：
- Dream Cycle = AI 睡眠时的记忆整合过程
- 清醒（WAKE）→ 快速眼动（REM-like）→ 慢波睡眠（SWS-like）→ 清醒
- 工作记忆：LCM插件（上下文压缩）
- 情景记忆：项目叙事 + 事件时间线
- 长期记忆：事实 + 决策 + 人物 + 里程碑
- 程序记忆：工作流 + 偏好 + 工具模式
- 索引层：元数据 + 重要性分数 + 关系 + 健康统计

使用示例：
    from agentmemory.extensions.v2.dream import DreamNet, SleepScheduler

    dream = DreamNet(workspace_dir="~/.openclaw/workspace")
    scheduler = SleepScheduler(dream)
    scheduler.start()  # 开始定时梦境周期

    # 手动触发梦境
    result = dream.run_dream_cycle()
"""

from .dream_cycle import DreamNet, DreamResult
from .importance_scorer import ImportanceScorer, EntryScore
from .forgetting_curve import ForgettingCurve
from .knowledge_graphger import KnowledgeGrapher, MemoryNode, MemoryEdge
from .sleep_scheduler import SleepScheduler
from .self_evolver import SelfEvolver, EvolutionRule
from .health_monitor import HealthMonitor, HealthReport
from .lucid_generator import LucidDreamGenerator, LucidDream, Inspiration
from .minimax_client import MiniMaxClient, create_minimax_client
from .multi_model_bridge import MultiModelMemoryBridge, ModelIdentity, MemorySlice

__all__ = [
    # 核心
    "DreamNet",
    "DreamResult",
    # 评分
    "ImportanceScorer",
    "EntryScore",
    # 遗忘
    "ForgettingCurve",
    # 图谱
    "KnowledgeGrapher",
    "MemoryNode",
    "MemoryEdge",
    # 调度
    "SleepScheduler",
    # 自进化
    "SelfEvolver",
    "EvolutionRule",
    # 健康
    "HealthMonitor",
    "HealthReport",
    # 清醒梦
    "LucidDreamGenerator",
    "LucidDream",
    "Inspiration",
    # MiniMax
    "MiniMaxClient",
    "create_minimax_client",
    # 跨模型记忆
    "MultiModelMemoryBridge",
    "ModelIdentity",
    "MemorySlice",
]

__version__ = "1.0.0"
