"""
MultiAgent 权限模型 — 路径级别访问控制

v0.3 §2.3 + §6 开放问题 #3
- 不同 Agent 看不同分类，实现权限隔离
- 支持前缀匹配（glob pattern）
- denied_paths 优先拒绝
- read_only_paths 限制写操作
"""

from __future__ import annotations

import asyncio
import json
import fnmatch
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiofiles


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class AgentPermission:
    """Agent 权限配置"""
    agent_id: str
    allowed_paths: list[str] = field(default_factory=list)
    read_only_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)
    max_write_per_day: int = 1000
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPermission":
        return cls(**data)


@dataclass
class PermissionContext:
    """当前操作的权限上下文"""
    agent_id: str
    operation: Literal["read", "write", "delete"]
    target_path: str
    granted: bool
    reason: str = ""


# ============================================================================
# 权限检查引擎
# ============================================================================


class PermissionEngine:
    """权限检查引擎

    支持：
    - 路径前缀/glob 匹配
    - denied_paths 优先拒绝
    - read_only_paths 限制写操作
    - 每日写限制追踪
    """

    PERMISSIONS_FILE: str = ".agent_permissions.json"
    SCHEMA_VERSION: int = 1

    def __init__(self, storage_path: Path):
        self.storage_path: Path = Path(storage_path)
        self._permissions_file: Path = self.storage_path / self.PERMISSIONS_FILE
        self._cache: dict[str, AgentPermission] = {}
        self._lock = asyncio.Lock()

    async def _load(self) -> dict:
        """从文件加载权限配置"""
        if not self._permissions_file.exists():
            return {"schema_version": self.SCHEMA_VERSION, "agents": {}}
        async with aiofiles.open(self._permissions_file, "r", encoding="utf-8") as f:
            return json.loads(await f.read())

    async def _save(self, data: dict) -> None:
        """持久化权限配置"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        tmp = self._permissions_file.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(self._permissions_file)

    async def register(
        self,
        agent_id: str,
        allowed_paths: list[str] | None = None,
        read_only_paths: list[str] | None = None,
        denied_paths: list[str] | None = None,
        max_write_per_day: int = 1000,
    ) -> AgentPermission:
        """注册新 Agent，返回权限配置

        Args:
            agent_id: Agent 唯一标识
            allowed_paths: 允许访问的路径列表（glob 模式）
            read_only_paths: 只读路径列表
            denied_paths: 拒绝访问的路径列表
            max_write_per_day: 每日写限制

        Returns:
            AgentPermission 实例
        """
        perm = AgentPermission(
            agent_id=agent_id,
            allowed_paths=allowed_paths or [],
            read_only_paths=read_only_paths or [],
            denied_paths=denied_paths or [],
            max_write_per_day=max_write_per_day,
        )

        async with self._lock:
            data = await self._load()
            data["agents"][agent_id] = perm.to_dict()
            await self._save(data)
            self._cache[agent_id] = perm

        return perm

    async def get(self, agent_id: str) -> AgentPermission | None:
        """获取 Agent 权限配置"""
        if agent_id in self._cache:
            return self._cache[agent_id]

        async with self._lock:
            data = await self._load()
            agent_data = data.get("agents", {}).get(agent_id)
            if agent_data:
                perm = AgentPermission.from_dict(agent_data)
                self._cache[agent_id] = perm
                return perm
            return None

    async def grant(self, agent_id: str, paths: list[str]) -> None:
        """授予路径访问权限

        Args:
            agent_id: Agent 唯一标识
            paths: 要授予的路径列表
        """
        perm = await self.get(agent_id)
        if not perm:
            perm = await self.register(agent_id, allowed_paths=paths)
            return

        async with self._lock:
            data = await self._load()
            agent_data = data["agents"].get(agent_id, {})
            current_allowed = set(agent_data.get("allowed_paths", []))
            current_allowed.update(paths)
            agent_data["allowed_paths"] = list(current_allowed)
            data["agents"][agent_id] = agent_data
            await self._save(data)

            perm.allowed_paths = list(current_allowed)
            self._cache[agent_id] = perm

    async def revoke(self, agent_id: str, paths: list[str]) -> None:
        """撤销路径访问权限

        Args:
            agent_id: Agent 唯一标识
            paths: 要撤销的路径列表
        """
        perm = await self.get(agent_id)
        if not perm:
            return

        async with self._lock:
            data = await self._load()
            agent_data = data["agents"].get(agent_id, {})
            current_allowed = set(agent_data.get("allowed_paths", []))
            current_allowed.difference_update(paths)
            agent_data["allowed_paths"] = list(current_allowed)
            data["agents"][agent_id] = agent_data
            await self._save(data)

            perm.allowed_paths = list(current_allowed)
            self._cache[agent_id] = perm

    async def check(
        self,
        agent_id: str,
        operation: str,
        target_path: str,
    ) -> PermissionContext:
        """检查操作是否允许，返回上下文

        优先级：
        1. denied_paths 优先拒绝
        2. allowed_paths 前缀匹配
        3. read_only_paths 匹配时拒绝 write/delete

        Args:
            agent_id: Agent 唯一标识
            operation: 操作类型 "read" | "write" | "delete"
            target_path: 目标路径

        Returns:
            PermissionContext
        """
        perm = await self.get(agent_id)

        if not perm:
            return PermissionContext(
                agent_id=agent_id,
                operation=operation,
                target_path=target_path,
                granted=False,
                reason="Agent not registered",
            )

        # 1. 检查 denied_paths（最高优先级）
        if self._match_any(target_path, perm.denied_paths):
            return PermissionContext(
                agent_id=agent_id,
                operation=operation,
                target_path=target_path,
                granted=False,
                reason="Path explicitly denied",
            )

        # 2. 检查是否在 allowed_paths 中
        if not self._match_any(target_path, perm.allowed_paths):
            return PermissionContext(
                agent_id=agent_id,
                operation=operation,
                target_path=target_path,
                granted=False,
                reason="Path not in allowed paths",
            )

        # 3. 检查 read_only_paths
        if operation in ("write", "delete"):
            if self._match_any(target_path, perm.read_only_paths):
                return PermissionContext(
                    agent_id=agent_id,
                    operation=operation,
                    target_path=target_path,
                    granted=False,
                    reason="Path is read-only",
                )

        # 4. 写操作检查每日限制
        if operation in ("write", "delete"):
            if await self._check_write_limit(agent_id, perm):
                return PermissionContext(
                    agent_id=agent_id,
                    operation=operation,
                    target_path=target_path,
                    granted=False,
                    reason="Daily write limit exceeded",
                )

        return PermissionContext(
            agent_id=agent_id,
            operation=operation,
            target_path=target_path,
            granted=True,
            reason="Access granted",
        )

    def _match_any(self, path: str, patterns: list[str]) -> bool:
        """检查路径是否匹配任意模式

        支持 glob 模式：
        - "A.项目/**" 匹配 A.项目 下的任意路径（包括深层）
        - "A.项目/*" 匹配 A.项目的直接子路径（单层）
        - "A.项目/石榴籽" 精确匹配
        """
        if not patterns:
            return False

        for pattern in patterns:
            # 标准化路径分隔符
            normalized_path = path.replace("\\", "/")
            normalized_pattern = pattern.replace("\\", "/")

            # 精确匹配
            if normalized_path == normalized_pattern:
                return True

            # 前缀匹配（处理 /** 后缀 - 匹配所有深层路径）
            if normalized_pattern.endswith("/**"):
                prefix = normalized_pattern[:-3]
                if normalized_path.startswith(prefix + "/") or normalized_path == prefix:
                    return True
                continue

            # 前缀匹配（处理 /* 后缀 - 单层通配，只匹配直接子路径）
            if normalized_pattern.endswith("/*"):
                prefix = normalized_pattern[:-2]
                # 必须以 prefix + "/" 开头
                if normalized_path.startswith(prefix + "/"):
                    # 检查剩余部分是否包含 /（如果包含，说明是更深层路径，不匹配）
                    rest = normalized_path[len(prefix) + 1:]
                    if "/" not in rest:
                        return True
                continue

            # 前缀匹配（无通配符的情况，如 "A.项目"）
            if normalized_path.startswith(normalized_pattern + "/"):
                return True

        return False

    async def list_allowed(self, agent_id: str) -> list[str]:
        """列出 Agent 可访问的所有路径

        Returns:
            allowed_paths 列表
        """
        perm = await self.get(agent_id)
        if not perm:
            return []
        return perm.allowed_paths.copy()

    def is_path_visible(self, agent_id: str, path: str) -> bool:
        """path 是否对 agent 可见

        用于快速检查（同步版本，使用缓存）
        """
        if agent_id in self._cache:
            perm = self._cache[agent_id]
            if self._match_any(path, perm.denied_paths):
                return False
            if self._match_any(path, perm.allowed_paths):
                return True
            return False
        return False

    async def _check_write_limit(self, agent_id: str, perm: AgentPermission) -> bool:
        """检查是否超过每日写限制

        简化实现：实际生产环境应追踪实际写入次数
        """
        # TODO: 实现真实的写限制追踪
        # 目前总是返回 False（不限制）
        return False

    async def list_agents(self) -> list[str]:
        """列出所有已注册的 Agent ID"""
        data = await self._load()
        return list(data.get("agents", {}).keys())

    async def unregister(self, agent_id: str) -> bool:
        """注销 Agent"""
        async with self._lock:
            data = await self._load()
            if agent_id in data["agents"]:
                del data["agents"][agent_id]
                await self._save(data)
                self._cache.pop(agent_id, None)
                return True
            return False


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "AgentPermission",
    "PermissionContext",
    "PermissionEngine",
]
