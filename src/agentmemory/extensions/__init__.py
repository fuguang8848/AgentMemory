"""AgentMemory Extensions — 远端 YintaTriss/AgentMemory 移植层

目的: 把上游 v2.0.1 的新模块（DataLake / Library / TagIndex /
EmbeddingStateMachine / TieredLog / DecayEngine / SearchEngine /
MultiAgentLock / MemoryHermes / API v2）以**可插拔**方式并入本地升级版，
**不破坏**本地 core / pipeline / security / observability / compat 的
14:53 SOP 成果（特别是 SOP #14 安全必做所需的 CircuitBreaker /
RateLimiter / PIIRedactor / Observability）。

设计原则（来自决策报告）:
- 本地升级版 (`core/`、`pipeline/`、`security/`、`observability/`) 优先
- 远端模块放入 `extensions/v2/`，与本地平级，**不污染** `__init__.py` 顶层
- `extensions/adapters/` 写桥接（远端 → 本地 ABC）
- `extensions/api.py` 提供**软导入** — 缺失依赖时 warn，不崩
- `extensions/YINTA_INTEGRATION.md` 文档化所有变更
- `extensions/CHANGELOG.md` 跟踪升级

安装/升级/卸载一条命令:
    python -m agentmemory.extensions.install   (在 main repo root 跑)
"""
from __future__ import annotations

from .api import (
    # 远端核心
    load_yinta_modules,
    is_yinta_available,
    yinta_version,
    YINTA_VERSION,
    # 子模块命名空间
    yinta_data,
    yinta_search,
    yinta_multi_agent,
    yinta_memory,
    # 适配到本地的桥
    GeometricDecayPolicy,
    YintaMemoryProvider,
    YintaSearchStrategy,
)

__all__ = [
    "load_yinta_modules",
    "is_yinta_available",
    "yinta_version",
    "YINTA_VERSION",
    "yinta_data",
    "yinta_search",
    "yinta_multi_agent",
    "yinta_memory",
    "GeometricDecayPolicy",
    "YintaMemoryProvider",
    "YintaSearchStrategy",
]
