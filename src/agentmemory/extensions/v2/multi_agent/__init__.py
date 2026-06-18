"""
MultiAgent 模块 — 多 Agent 共享与权限

v2.0 架构：§11 MultiAgent 机制
包含：MultiAgentLock / SharedLog / AgentRegistry + 权限模型
"""

# 从 multi_agent_core.py 导入原有类 (本地路径, 不依赖 sys.modules 别名)
from agentmemory.extensions.v2.multi_agent_core import (
    MultiAgentLock,
    SharedLog,
    SharedLogEntry,
    AgentRegistry,
    TurnNotification,
    MultiAgent,
    MultiAgentLockTimeout,
    AgentNotRegisteredError,
    SharedLogError,
    generate_agent_id,
)

# 从 permissions.py 导入权限类 (相对 import, 不需 sys.modules 别名)
from .permissions import (
    AgentPermission,
    PermissionContext,
    PermissionEngine,
)

__all__ = [
    # 原有类
    "MultiAgentLock",
    "SharedLog",
    "SharedLogEntry",
    "AgentRegistry",
    "TurnNotification",
    "MultiAgent",
    "MultiAgentLockTimeout",
    "AgentNotRegisteredError",
    "SharedLogError",
    "generate_agent_id",
    # 权限类
    "AgentPermission",
    "PermissionContext",
    "PermissionEngine",
]
