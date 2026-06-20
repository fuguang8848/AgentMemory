"""Dynamic Team Builder — hierarchical task decomposition + dynamic agent role assignment.

类似 Deer-flow 的 hierarchical planning:
1. TaskDecomposer      — 将复杂任务分解为子任务树
2. DynamicTeamBuilder  — 根据子任务动态分配 Agent 角色 (planner/critic/actor/reporter/searcher)
3. TeamOrchestrator    — 协调多 Agent 协作，收集结果，检测冲突，汇总输出

支持同步 / 异步两种协作模式，集成 OTel 可观测性。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────


class AgentRole(str, Enum):
    """角色池."""

    PLANNER = "planner"      # 规划分解
    CRITIC = "critic"        # 审查批评
    ACTOR = "actor"          # 执行操作
    REPORTER = "reporter"    # 汇报总结
    SEARCHER = "searcher"    # 搜索检索


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskNode:
    """子任务树节点."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    role: AgentRole | None = None          # 建议角色
    status: TaskStatus = TaskStatus.PENDING
    parent_id: str | None = None
    children: list[TaskNode] = field(default_factory=list)
    result: Any = None
    error: str | None = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class TaskTree:
    """子任务树 (decompose_task 的返回结果)."""

    root: TaskNode
    total_nodes: int = 0
    leaf_count: int = 0

    def __post_init__(self):
        self.total_nodes = self._count_nodes(self.root)
        self.leaf_count = self._count_leaves(self.root)

    def _count_nodes(self, node: TaskNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _count_leaves(self, node: TaskNode) -> int:
        if node.is_leaf():
            return 1
        return sum(self._count_leaves(c) for c in node.children)


@dataclass
class Agent:
    """虚拟 Agent (不绑定具体 LLM 实现，仅保留角色 + 执行逻辑)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: AgentRole = AgentRole.ACTOR
    tasks: list[TaskNode] = field(default_factory=list)
    busy: bool = False

    def can_handle(self, task: TaskNode) -> bool:
        if self.busy:
            return False
        if task.role is None:
            return True
        return task.role == self.role


@dataclass
class Conflict:
    """检测到的冲突."""

    task_a_id: str
    task_b_id: str
    description: str
    severity: str = "medium"  # low / medium / high


@dataclass
class CollaborationResult:
    """run_collaboration 的返回结果."""

    task_tree: TaskTree
    agent_outputs: dict[str, Any]  # agent_id -> output
    conflicts: list[Conflict]
    summary: str
    duration_seconds: float
    success: bool


# ──────────────────────────────────────────────────────────────────────────────
# TaskDecomposer — hierarchical planning
# ──────────────────────────────────────────────────────────────────────────────


class TaskDecomposer:
    """将复杂任务分解为子任务树.

    使用简单的启发式 + 递归拆分，模拟 Deer-flow 的 hierarchical planning。
    """

    ROLE_KEYWORDS: dict[AgentRole, list[str]] = {
        AgentRole.PLANNER: ["规划", "分解", "拆分", "设计", "计划"],
        AgentRole.CRITIC: ["审查", "批评", "检查", "审核", "质疑"],
        AgentRole.ACTOR: ["执行", "实施", "完成", "做", "写", "生成"],
        AgentRole.REPORTER: ["汇报", "总结", "报告", "输出", "呈现"],
        AgentRole.SEARCHER: ["搜索", "查询", "查找", "检索", "获取"],
    }

    def decompose_task(self, task_text: str, max_depth: int = 3) -> TaskTree:
        """将任务文本递归分解为子任务树.

        参数:
            task_text: 原始任务描述
            max_depth: 最大递归深度

        返回:
            TaskTree — 包含根节点和所有子节点
        """
        with tracer.start_as_current_span("TaskDecomposer.decompose_task") as span:
            span.set_attribute("task_text", task_text[:200])
            span.set_attribute("max_depth", max_depth)

            root = TaskNode(description=task_text)
            self._recursive_decompose(root, depth=0, max_depth=max_depth)

            tree = TaskTree(root=root)
            span.set_attribute("total_nodes", tree.total_nodes)
            span.set_attribute("leaf_count", tree.leaf_count)
            logger.info(
                "Task decomposed: %d nodes, %d leaves",
                tree.total_nodes,
                tree.leaf_count,
            )
            return tree

    def _recursive_decompose(self, node: TaskNode, depth: int, max_depth: int) -> None:
        """递归拆分节点直到达到 max_depth 或不可再分."""
        if depth >= max_depth:
            node.role = self._infer_role(node.description)
            return

        subtasks = self._generate_subtasks(node.description)
        if len(subtasks) <= 1 or depth == max_depth - 1:
            node.role = self._infer_role(node.description)
            return

        for subtask_text in subtasks:
            child = TaskNode(description=subtask_text, parent_id=node.id)
            node.children.append(child)
            self._recursive_decompose(child, depth + 1, max_depth)

    def _generate_subtasks(self, task_text: str) -> list[str]:
        """基于关键词模式匹配生成子任务 (简单启发式)."""
        text = task_text.lower()

        # 复杂任务模式：包含多个动词短语 → 拆分
        separators = [
            "，", "。", "；", "\n",
            "首先", "然后", "接着", "最后",
            "第一步", "第二步", "第三步",
            "并且", "同时", "此外",
        ]

        segments = [task_text]
        for sep in separators:
            new_segments = []
            for seg in segments:
                new_segments.extend(seg.split(sep))
            if len(new_segments) > len(segments):
                segments = new_segments
                break

        # 过滤掉太短的片段
        result = [s.strip() for s in segments if len(s.strip()) > 5]
        return result if len(result) > 1 else [task_text]

    def _infer_role(self, description: str) -> AgentRole:
        """根据描述关键词推断最佳角色."""
        text = description.lower()
        scores: dict[AgentRole, int] = {role: 0 for role in AgentRole}

        for role, keywords in self.ROLE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[role] += 1

        if max(scores.values()) == 0:
            return AgentRole.ACTOR
        return max(scores, key=scores.get)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# DynamicTeamBuilder — role assignment
# ──────────────────────────────────────────────────────────────────────────────


class DynamicTeamBuilder:
    """根据 TaskTree 动态分配 Agent 角色.

    角色池: planner / critic / actor / reporter / searcher
    """

    def __init__(self, max_agents_per_role: int = 3):
        self.max_agents_per_role = max_agents_per_role
        self.decomposer = TaskDecomposer()

    def build_team(self, task_tree: TaskTree) -> list[Agent]:
        """根据子任务树构建 Agent 团队.

        参数:
            task_tree: 由 TaskDecomposer 生成的子任务树

        返回:
            List[Agent] — 分配好角色的 Agent 列表
        """
        with tracer.start_as_current_span("DynamicTeamBuilder.build_team") as span:
            span.set_attribute("total_nodes", task_tree.total_nodes)

            # 统计各角色需要的数量
            role_counts = self._count_roles(task_tree)
            span.set_attribute("role_counts", str(role_counts))

            team: list[Agent] = []
            for role, count in role_counts.items():
                actual = min(count, self.max_agents_per_role)
                for i in range(actual):
                    agent = Agent(id=f"{role.value}-{i+1}", role=role)
                    team.append(agent)

            # 将叶子任务分配给对应的 Agent
            self._assign_tasks_to_agents(team, task_tree)

            logger.info("Team built: %d agents", len(team))
            span.set_attribute("team_size", len(team))
            return team

    def _count_roles(self, tree: TaskTree) -> dict[AgentRole, int]:
        """统计各角色在叶子任务中的需求量."""
        counts: dict[AgentRole, int] = {role: 0 for role in AgentRole}
        self._traverse_and_count(tree.root, counts)
        return counts

    def _traverse_and_count(self, node: TaskNode, counts: dict[AgentRole, int]) -> None:
        if node.is_leaf():
            role = node.role or AgentRole.ACTOR
            counts[role] += 1
        for child in node.children:
            self._traverse_and_count(child, counts)

    def _assign_tasks_to_agents(
        self, team: list[Agent], task_tree: TaskTree
    ) -> None:
        """将叶子任务均匀分配给对应角色的 Agent."""
        role_agents: dict[AgentRole, list[Agent]] = {}
        for agent in team:
            role_agents.setdefault(agent.role, []).append(agent)

        leaves = self._collect_leaves(task_tree.root)
        round_robin: dict[AgentRole, int] = {role: 0 for role in AgentRole}

        for leaf in leaves:
            role = leaf.role or AgentRole.ACTOR
            agents = role_agents.get(role, [])
            if not agents:
                # fallback: 任何空闲的 actor
                actors = [a for a in team if a.role == AgentRole.ACTOR and not a.busy]
                if actors:
                    agents = actors

            if agents:
                idx = round_robin[role] % len(agents)
                agent = agents[idx]
                agent.tasks.append(leaf)
                round_robin[role] += 1
                leaf.role = role  # 确认角色

    def _collect_leaves(self, node: TaskNode) -> list[TaskNode]:
        if node.is_leaf():
            return [node]
        result: list[TaskNode] = []
        for child in node.children:
            result.extend(self._collect_leaves(child))
        return result


# ──────────────────────────────────────────────────────────────────────────────
# TeamOrchestrator — coordination
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class OrchestratorConfig:
    """TeamOrchestrator 配置."""

    mode: str = "sync"           # "sync" | "async"
    timeout_seconds: float = 60.0
    detect_conflicts: bool = True
    conflict_threshold: float = 0.7  # 相似度阈值


T = TypeVar("T")


class TeamOrchestrator:
    """协调多 Agent 协作，收集结果，检测冲突，汇总输出."""

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()

    def run_collaboration(
        self,
        team: list[Agent],
        task: str,
        execute_fn: Callable[[Agent, TaskNode], Any] | None = None,
    ) -> CollaborationResult:
        """运行协作 (同步模式).

        参数:
            team:        Agent 团队
            task:        原始任务描述
            execute_fn:  可选的执行函数 (agent, task_node) -> Any
                         若不提供，使用默认 mock 执行

        返回:
            CollaborationResult
        """
        with tracer.start_as_current_span("TeamOrchestrator.run_collaboration") as span:
            span.set_attribute("mode", self.config.mode)
            span.set_attribute("team_size", len(team))

            start = time.monotonic()
            tree = TaskDecomposer().decompose_task(task)

            # 重新构建团队并分配任务
            builder = DynamicTeamBuilder()
            team = builder.build_team(tree)

            if self.config.mode == "async":
                raise NotImplementedError("Async mode, use run_collaboration_async()")

            # 同步执行
            agent_outputs: dict[str, Any] = {}
            for agent in team:
                for task_node in agent.tasks:
                    task_node.status = TaskStatus.RUNNING
                    try:
                        result = (
                            execute_fn(agent, task_node)
                            if execute_fn
                            else self._default_execute(agent, task_node)
                        )
                        task_node.result = result
                        task_node.status = TaskStatus.DONE
                        agent_outputs[agent.id] = result
                    except Exception as exc:  # noqa: BLE001
                        task_node.status = TaskStatus.FAILED
                        task_node.error = str(exc)
                        logger.error("Agent %s failed on task %s: %s", agent.id, task_node.id, exc)

            # 冲突检测
            conflicts = []
            if self.config.detect_conflicts:
                conflicts = self._detect_conflicts(tree)

            # 汇总
            summary = self._summarize(tree, team, conflicts)
            duration = time.monotonic() - start
            success = all(
                t.status == TaskStatus.DONE
                for t in self._collect_all_nodes(tree.root)
            )

            span.set_attribute("duration_seconds", duration)
            span.set_attribute("success", success)
            span.set_attribute("conflict_count", len(conflicts))

            return CollaborationResult(
                task_tree=tree,
                agent_outputs=agent_outputs,
                conflicts=conflicts,
                summary=summary,
                duration_seconds=duration,
                success=success,
            )

    async def run_collaboration_async(
        self,
        team: list[Agent],
        task: str,
        execute_fn: Callable[[Agent, TaskNode], Any] | None = None,
    ) -> CollaborationResult:
        """运行协作 (异步模式).

        参数:
            team:        Agent 团队
            task:        原始任务描述
            execute_fn:  可选的 async 执行函数 (agent, task_node) -> Any

        返回:
            CollaborationResult
        """
        with tracer.start_as_current_span("TeamOrchestrator.run_collaboration_async") as span:
            span.set_attribute("mode", "async")
            span.set_attribute("team_size", len(team))

            start = time.monotonic()
            tree = TaskDecomposer().decompose_task(task)

            builder = DynamicTeamBuilder()
            team = builder.build_team(tree)

            # 并发执行
            agent_outputs: dict[str, Any] = {}
            tasks: list[asyncio.Task] = []

            for agent in team:
                for task_node in agent.tasks:
                    task_node.status = TaskStatus.RUNNING
                    coro = self._async_execute(
                        agent, task_node, execute_fn
                    )
                    tasks.append(asyncio.create_task(coro))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 收集结果
            idx = 0
            for agent in team:
                for task_node in agent.tasks:
                    if idx < len(results):
                        result = results[idx]
                        if isinstance(result, Exception):
                            task_node.status = TaskStatus.FAILED
                            task_node.error = str(result)
                        else:
                            task_node.result = result
                            task_node.status = TaskStatus.DONE
                            agent_outputs[agent.id] = result
                    idx += 1

            conflicts = []
            if self.config.detect_conflicts:
                conflicts = self._detect_conflicts(tree)

            summary = self._summarize(tree, team, conflicts)
            duration = time.monotonic() - start
            success = all(
                t.status == TaskStatus.DONE
                for t in self._collect_all_nodes(tree.root)
            )

            span.set_attribute("duration_seconds", duration)
            span.set_attribute("success", success)

            return CollaborationResult(
                task_tree=tree,
                agent_outputs=agent_outputs,
                conflicts=conflicts,
                summary=summary,
                duration_seconds=duration,
                success=success,
            )

    def _default_execute(self, agent: Agent, task_node: TaskNode) -> str:
        """默认执行 (mock)."""
        return f"[{agent.role.value}] executed: {task_node.description[:50]}..."

    async def _async_execute(
        self,
        agent: Agent,
        task_node: TaskNode,
        fn: Callable[[Agent, TaskNode], Any] | None,
    ) -> Any:
        """异步执行包装器."""
        if fn is not None:
            result = fn(agent, task_node)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return self._default_execute(agent, task_node)

    def _detect_conflicts(self, tree: TaskTree) -> list[Conflict]:
        """检测任务间的冲突 (基于描述相似度)."""
        conflicts: list[Conflict] = []
        leaves = self._collect_leaves(tree.root)

        for i, a in enumerate(leaves):
            for b in leaves[i + 1 :]:
                sim = self._similarity(a.description, b.description)
                if sim >= self.config.conflict_threshold:
                    conflicts.append(
                        Conflict(
                            task_a_id=a.id,
                            task_b_id=b.id,
                            description=f"High similarity ({sim:.2f}): "
                            f"'{a.description[:40]}' vs '{b.description[:40]}'",
                            severity="high" if sim > 0.9 else "medium",
                        )
                    )
        return conflicts

    def _similarity(self, text_a: str, text_b: str) -> float:
        """简单词集合相似度 (Jaccard)."""
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _summarize(
        self, tree: TaskTree, team: list[Agent], conflicts: list[Conflict]
    ) -> str:
        """生成汇总报告."""
        leaves = self._collect_leaves(tree.root)
        done = sum(1 for t in leaves if t.status == TaskStatus.DONE)
        failed = sum(1 for t in leaves if t.status == TaskStatus.FAILED)

        lines = [
            f"TaskTree: {tree.total_nodes} nodes, {tree.leaf_count} leaves",
            f"Team: {len(team)} agents",
            f"Results: {done} done, {failed} failed",
            f"Conflicts: {len(conflicts)} detected",
        ]
        if conflicts:
            lines.append("  Top conflicts:")
            for c in conflicts[:3]:
                lines.append(f"    - {c.description}")

        return "\n".join(lines)

    def _collect_leaves(self, node: TaskNode) -> list[TaskNode]:
        if node.is_leaf():
            return [node]
        result: list[TaskNode] = []
        for child in node.children:
            result.extend(self._collect_leaves(child))
        return result

    def _collect_all_nodes(self, node: TaskNode) -> list[TaskNode]:
        result = [node]
        for child in node.children:
            result.extend(self._collect_all_nodes(child))
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ──────────────────────────────────────────────────────────────────────────────


def decompose_task(task_text: str, max_depth: int = 3) -> TaskTree:
    """顶层入口：分解任务."""
    return TaskDecomposer().decompose_task(task_text, max_depth)


def build_team(task_tree: TaskTree) -> list[Agent]:
    """顶层入口：构建团队."""
    return DynamicTeamBuilder().build_team(task_tree)


def run_collaboration(
    team: list[Agent],
    task: str,
    mode: str = "sync",
    execute_fn: Callable[[Agent, TaskNode], Any] | None = None,
) -> CollaborationResult:
    """顶层入口：运行协作.

    参数:
        team:        Agent 团队 (由 build_team 生成)
        task:        原始任务
        mode:        "sync" | "async"
        execute_fn:  可选的执行函数
    """
    config = OrchestratorConfig(mode=mode)
    orchestrator = TeamOrchestrator(config)
    if mode == "async":
        raise TypeError("Use run_collaboration_async() for async mode")
    return orchestrator.run_collaboration(team, task, execute_fn)


async def run_collaboration_async(
    team: list[Agent],
    task: str,
    execute_fn: Callable[[Agent, TaskNode], Any] | None = None,
) -> CollaborationResult:
    """顶层入口：异步运行协作."""
    config = OrchestratorConfig(mode="async")
    orchestrator = TeamOrchestrator(config)
    return await orchestrator.run_collaboration_async(team, task, execute_fn)
