"""extensions/v2_aliases.py — 注册 sys.modules 别名 (远端代码 0 修改)

按"依赖深度"分批注册 alias, 每批在对应 v2 模块 import 之前注册.
避免 v2 包 __init__.py 跑时 alias 尚未生效的循环 import.

批设计 (与 api.py 的 phase 顺序对应):
    batch_pre  : 远端代码用作 "from xxx import" 但本身不依赖 v2 内部其他模块的
    batch_data : data/ 内部
    batch_ma   : multi_agent, 依赖 multi_agent_core (必须在 batch_pre 之后)
"""
from __future__ import annotations

import importlib
import sys
from typing import Any

# 一次性预装: 这些 target 在 extensions.v2 内是"叶子", 可独立 import
# (Phase 1 之前先装, Phase 1 import 时 alias 已就绪)
_ALIASES_LEAF = {
    "agentmemory.errors": "agentmemory.extensions.v2.errors",
    "agentmemory.config": "agentmemory.extensions.v2.config",
    "agentmemory.models": "agentmemory.extensions.v2.models",
    "agentmemory.llm_client": "agentmemory.extensions.v2.llm_client",
    "agentmemory.decay_engine": "agentmemory.extensions.v2.decay_engine",
    "agentmemory.L3_vector_store": "agentmemory.extensions.v2.L3_vector_store",
    "agentmemory.L4_file_persist": "agentmemory.extensions.v2.L4_file_persist",
}

# Phase 1 装完后, 这些 alias 也可生效 (multi_agent_core 自身)
_ALIASES_AFTER_PHASE1 = {
    "agentmemory.multi_agent_core": "agentmemory.extensions.v2.multi_agent_core",
}

# multi_agent 内部用 (Phase 2 之前装)
_ALIASES_AFTER_PHASE2 = {
    "agentmemory.multi_agent": "agentmemory.extensions.v2.multi_agent",
    "agentmemory.multi_agent.permissions": "agentmemory.extensions.v2.multi_agent.permissions",
}

# search 内部用 (Phase 3 之前装)
_ALIASES_AFTER_PHASE3 = {
    "agentmemory.search.hybrid_retriever": "agentmemory.extensions.v2.search.hybrid_retriever",
    "agentmemory.search.rrf_fusion": "agentmemory.extensions.v2.search.rrf_fusion",
}

# Phase 4 之前装 (api/workers/providers 互引用)
_ALIASES_AFTER_PHASE3b = {
    "agentmemory.providers": "agentmemory.extensions.v2.providers",
    "agentmemory.providers.protocols": "agentmemory.extensions.v2.providers.protocols",
    "agentmemory.providers.embedder": "agentmemory.extensions.v2.providers.embedder",
    "agentmemory.providers.vectorstore": "agentmemory.extensions.v2.providers.vectorstore",
    "agentmemory.providers.llm": "agentmemory.extensions.v2.providers.llm",
    "agentmemory.providers.registry": "agentmemory.extensions.v2.providers.registry",
    "agentmemory.workers": "agentmemory.extensions.v2.workers",
}


def _try_install(mapping: dict[str, str]) -> dict[str, str]:
    """尝试装一批 alias. 返回成功装的. 失败的不再重试."""
    installed: dict[str, str] = {}
    for alias, target in mapping.items():
        if alias in sys.modules:
            installed[alias] = target
            continue
        try:
            real_mod = importlib.import_module(target)
            sys.modules[alias] = real_mod
            installed[alias] = target
        except Exception:
            # target 自身依赖的 alias 还没装 — 跳过, 等下一批
            pass
    return installed


def install_pre() -> dict[str, str]:
    """Phase 1 import 之前调用"""
    return _try_install(_ALIASES_LEAF)


def install_after_phase1() -> dict[str, str]:
    """Phase 1 import 完之后调用"""
    return _try_install(_ALIASES_AFTER_PHASE1)


def install_after_phase2() -> dict[str, str]:
    """Phase 2 (multi_agent) import 之前调用"""
    return _try_install(_ALIASES_AFTER_PHASE2)


def install_after_phase3() -> dict[str, str]:
    """Phase 3 (search) import 之前调用"""
    return _try_install(_ALIASES_AFTER_PHASE3)


def install_after_phase3b() -> dict[str, str]:
    """Phase 3b (api/workers/providers 互引用) 之前调用"""
    return _try_install(_ALIASES_AFTER_PHASE3b)


def install_all() -> dict[str, str]:
    """一次性装所有 (按依赖顺序). 用于调试或用户手动触发."""
    return {
        **install_pre(),
        **install_after_phase1(),
        **install_after_phase2(),
        **install_after_phase3(),
        **install_after_phase3b(),
    }
