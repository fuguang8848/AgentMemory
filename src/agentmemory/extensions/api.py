"""extensions/api.py — 远端模块软导入 + 本地桥接 (单一入口)

设计:
- `load_yinta_modules()` 按依赖顺序加载（先低层 DataLake 等，后高层 MemoryHermes）
- 任意模块失败不阻断，返回 status dict
- 适配层（adapters/）在本地 core ABC 之上包远端实现

依赖图（远端 v2.0.1 经验证）:
    ✅ 独立 (Phase 1, 必成功)
        data/datalake, data/library, data/tag_index, data/embedding_state,
        data/tiered_log, decay_engine, multi_agent/permissions, multi_agent_core
    ⚠️ 中等 (Phase 2, 可能缺)
        search/search_engine — 缺 providers.protocols / rrf_fusion
    ❌ 高级 (Phase 3, 必然缺)
        memory_manager — 缺 L3_vector_store / L4_file_persist / providers.*
        api/v2/app — 依赖 memory_manager
        workers — 暂未移植
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 路径常量
# ----------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_V2_DIR = _THIS_DIR / "v2"

# 让远端模块能 `from agentmemory.extensions.v2.xxx import ...`
# 已通过包结构自动支持

YINTA_VERSION = "2.0.1"
SOURCE_REPO = "https://github.com/YintaTriss/AgentMemory"
SOURCE_TAG = f"v{YINTA_VERSION}"
LOCAL_PATH = "agentmemory.extensions.v2"
LOADED: dict[str, Any] = {}
FAILED: dict[str, str] = {}


# ----------------------------------------------------------------------
# 软导入函数
# ----------------------------------------------------------------------
def _safe_import(name: str) -> Any | None:
    """软导入 — 失败 log warn 并返回 None，绝不抛"""
    try:
        mod = importlib.import_module(name)
        LOADED[name] = mod
        return mod
    except Exception as e:
        FAILED[name] = f"{type(e).__name__}: {e}"
        logger.debug(f"[extensions] {name} 加载失败: {FAILED[name]}")
        return None


def is_yinta_available() -> bool:
    """至少一个独立模块可用即算成功"""
    return bool(LOADED)


def yinta_version() -> str:
    return YINTA_VERSION


# ----------------------------------------------------------------------
# 命名空间对象 — 惰性加载
# ----------------------------------------------------------------------
class _LazyNS:
    """延迟加载命名空间 — 访问属性时才尝试 import"""
    def __init__(self, dotted_path: str):
        self._path = dotted_path
        self._cache: dict[str, Any] = {}
        self._loaded = False
        self._error: str | None = None

    def _ensure(self):
        if self._loaded or self._error:
            return
        try:
            self._mod = importlib.import_module(self._path)
            self._loaded = True
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            logger.debug(f"[extensions] 命名空间 {self._path} 加载失败: {self._error}")

    def __getattr__(self, name: str):
        self._ensure()
        if name in self._cache:
            return self._cache[name]
        if not self._loaded:
            raise AttributeError(f"{self._path} 未加载: {self._error}")
        attr = getattr(self._mod, name, None)
        if attr is None:
            raise AttributeError(f"{self._path} 无属性 {name}")
        self._cache[name] = attr
        return attr


yinta_data = _LazyNS(f"{LOCAL_PATH}.data")
yinta_search = _LazyNS(f"{LOCAL_PATH}.search")
yinta_multi_agent = _LazyNS(f"{LOCAL_PATH}.multi_agent")
yinta_memory = _LazyNS(f"{LOCAL_PATH}")


# ----------------------------------------------------------------------
# 主加载函数 — 按依赖图分阶段
# ----------------------------------------------------------------------
def load_yinta_modules(eager: bool = True) -> dict[str, str]:
    """
    加载远端 v2.0.1 模块（分阶段，容错）。

    Returns:
        dict[module_path, "OK" | "FAIL: reason"]
    """
    FAILED.clear()
    LOADED.clear()

    # ---- 步骤 0a: 修复 ulid 与 Python 3.12 不兼容 ----
    try:
        from .ulid_compat import patch_ulid
        patch_ulid()
    except Exception as e:
        logger.warning(f"[extensions] ulid patch 失败: {e}")

    # ---- 步骤 0b: 先装 leaf alias, Phase 1 import 时 alias 已就绪 ----
    from .v2_aliases import (
        install_pre,
        install_after_phase1,
        install_after_phase2,
        install_after_phase3,
        install_after_phase3b,
    )
    pre = install_pre()
    logger.info(f"[extensions] pre-aliases: {len(pre)}")

    # ---- Phase 1: 独立 (无依赖，必成功) ----
    phase1 = [
        f"{LOCAL_PATH}.data",
        f"{LOCAL_PATH}.data.datalake",
        f"{LOCAL_PATH}.data.library",
        f"{LOCAL_PATH}.data.tag_index",
        f"{LOCAL_PATH}.data.embedding_state",
        f"{LOCAL_PATH}.data.tiered_log",
        f"{LOCAL_PATH}.decay_engine",
        f"{LOCAL_PATH}.multi_agent_core",
        f"{LOCAL_PATH}.errors",
        f"{LOCAL_PATH}.config",
        f"{LOCAL_PATH}.models",
        f"{LOCAL_PATH}.llm_client",
        f"{LOCAL_PATH}.L3_vector_store",
        f"{LOCAL_PATH}.L4_file_persist",
    ]
    for name in phase1:
        if eager:
            _safe_import(name)
    a1 = install_after_phase1()
    logger.info(f"[extensions] after-phase1 aliases: {len(a1)}")

    # ---- Phase 2: 依赖 multi_agent_core 的 ----
    phase2 = [
        f"{LOCAL_PATH}.multi_agent",
        f"{LOCAL_PATH}.multi_agent.permissions",
    ]
    a2 = install_after_phase2()
    logger.info(f"[extensions] after-phase2 aliases: {len(a2)}")
    for name in phase2:
        if eager:
            _safe_import(name)

    # ---- Phase 3: 中等 (search 依赖 multi_agent 的 PermissionEngine) ----
    phase3 = [
        f"{LOCAL_PATH}.search",
        f"{LOCAL_PATH}.search.search_engine",
    ]
    a3 = install_after_phase3()
    logger.info(f"[extensions] after-phase3 aliases: {len(a3)}")
    a3b = install_after_phase3b()
    logger.info(f"[extensions] after-phase3b aliases: {len(a3b)}")
    for name in phase3:
        if eager:
            _safe_import(name)

    # ---- Phase 4: 高级 (缺 L3/L4/providers, 移植失败预期) ----
    phase4 = [
        f"{LOCAL_PATH}.memory_manager",
        f"{LOCAL_PATH}.api",
        f"{LOCAL_PATH}.api.v2",
        f"{LOCAL_PATH}.api.v2.app",
    ]
    for name in phase4:
        if eager:
            _safe_import(name)

    # ---- 适配层（与远端解耦） ----
    for name in [
        "agentmemory.extensions.adapters",
        "agentmemory.extensions.adapters.geometric_decay",
        "agentmemory.extensions.adapters.yinta_memory_provider",
        "agentmemory.extensions.adapters.search_engine_adapter",
    ]:
        _safe_import(name)

    # ---- 修复 ulid 与 Python 3.12 不兼容 (在所有 import 之后) ----
    # datalake 里 from ulid import ULID 是复制引用, 必须在 datalake module namespace 里 patch
    try:
        from .ulid_compat import patch_ulid_in_module
        for mod_name in [
            f"{LOCAL_PATH}.data.datalake",
            f"{LOCAL_PATH}.data.library",
            f"{LOCAL_PATH}.memory_manager",
        ]:
            if mod_name in LOADED:
                patch_ulid_in_module(LOADED[mod_name])
    except Exception as e:
        logger.warning(f"[extensions] ulid patch 失败: {e}")

    return {
        **{k: "OK" for k in LOADED},
        **{k: f"FAIL: {v}" for k, v in FAILED.items()},
    }


# ----------------------------------------------------------------------
# 适配层 — 惰性 import
# ----------------------------------------------------------------------
def GeometricDecayPolicy(*args, **kwargs):
    """远端 DecayEngine 适配到本地 DecayPolicy ABC (Phase 1 即可用)"""
    from .adapters.geometric_decay import GeometricDecayPolicy as _G
    return _G(*args, **kwargs)


def YintaMemoryProvider(*args, **kwargs):
    """远端 DataLake + Decay + Library 组合适配到本地 MemoryProvider ABC (Phase 1)"""
    from .adapters.yinta_memory_provider import YintaMemoryProvider as _Y
    return _Y(*args, **kwargs)


def YintaSearchStrategy(*args, **kwargs):
    """远端 SearchEngine 适配到本地 RetrievalStrategy (Phase 2 — 可能不可用)"""
    from .adapters.search_engine_adapter import YintaSearchStrategy as _S
    return _S(*args, **kwargs)
