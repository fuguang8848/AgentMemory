"""v2 包初始化 — 自身只做路径常量声明，alias 注册在 load_yinta_modules 末尾执行

远端 YintaTriss/AgentMemory 内部用 `from agentmemory.xxx import yyy` 写死了路径。
我们在 load_yinta_modules() 末尾, 把 `agentmemory.extensions.v2.*` 映射成 `agentmemory.*`，
这样远端代码 0 修改即可工作。

不在这里注册是因为: v2/__init__.py 加载时, multi_agent_core 自身依赖尚未全部加载,
会触发循环 ImportError. 后期注册则所有模块都已 OK.
"""
from __future__ import annotations

__version__ = "2.0.1"  # 远端版本
SOURCE_REPO = "https://github.com/YintaTriss/AgentMemory"
SOURCE_TAG = "v2.0.1"

# ── OTel 可观测性 ────────────────────────────────────────────────────────────
from agentmemory.extensions.v2.opentelemetry_setup import (
    setup_tracing,
    setup_metrics,
    setup_logging,
    async_trace,
)

# ── Dynamic Team Builder ───────────────────────────────────────────────────────
from agentmemory.extensions.v2.dynamic_team_builder import (
    TaskDecomposer,
    DynamicTeamBuilder,
    TeamOrchestrator,
    AgentRole,
    TaskNode,
    TaskTree,
    Agent,
    Conflict,
    CollaborationResult,
    OrchestratorConfig,
    decompose_task,
    build_team,
    run_collaboration,
    run_collaboration_async,
)

# ── Skill Versioning ───────────────────────────────────────────────────────────
from agentmemory.extensions.v2.skill_versioning import (
    SkillVersion,
    SkillRegistry,
    SkillHotReload,
    SkillDiff,
    skill_diff,
)

__all__ = [
    # OTel
    "setup_tracing",
    "setup_metrics",
    "setup_logging",
    "async_trace",
    # Dynamic Team Builder
    "TaskDecomposer",
    "DynamicTeamBuilder",
    "TeamOrchestrator",
    "AgentRole",
    "TaskNode",
    "TaskTree",
    "Agent",
    "Conflict",
    "CollaborationResult",
    "OrchestratorConfig",
    "decompose_task",
    "build_team",
    "run_collaboration",
    "run_collaboration_async",
    # Skill Versioning
    "SkillVersion",
    "SkillRegistry",
    "SkillHotReload",
    "SkillDiff",
    "skill_diff",
]
