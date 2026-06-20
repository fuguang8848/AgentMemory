"""
MultiAgent 模块 — 多 Agent 共享与权限

v2.0 架构：§11 MultiAgent 机制
包含：MultiAgentLock / SharedLog / AgentRegistry + 权限模型
"""

# 从 permissions.py 导入权限类
from .permissions import (
    AgentPermission,
    PermissionContext,
    PermissionEngine,
)

# 从 memory_bus.py 导入 MemoryBus
from .memory_bus import (
    MemoryBus,
    MemoryMessage,
    SharedMemoryEntry,
    Subscription,
)

__all__ = [
    # MemoryBus
    "MemoryBus",
    "MemoryMessage",
    "SharedMemoryEntry",
    "Subscription",
    # 权限类
    "AgentPermission",
    "PermissionContext",
    "PermissionEngine",
]
