"""ulid_compat.py — 修复 python-ulid 1.1.0 与 Python 3.12 不兼容

ulid 1.1.0 调用 `MemoryView.__init__() missing buffer` 是已知 bug.
不影响我们的语义 (DataLake 只用 `mem_{ULID()}` 字符串拼接), 直接用 uuid 替代.

用法:
    from agentmemory.extensions.ulid_compat import patch_ulid
    patch_ulid()  # 在 import extensions.v2.data.datalake 之前调用
"""
from __future__ import annotations

import sys
import uuid
from types import ModuleType


def _ulid_str() -> str:
    """生成 26 字符唯一 ID (ULID 格式长度, 用 uuid4 实现)"""
    return uuid.uuid4().hex[:26].upper()


def patch_ulid() -> bool:
    """在 sys.modules 里替换 `ulid.ULID` 为一个 work 的版本. 成功返回 True."""
    if "ulid" not in sys.modules:
        return False
    ulid_mod = sys.modules["ulid"]
    # 如果已经 patch 过, 跳过
    if getattr(ulid_mod.ULID, "_v_patched", False):
        return True
    # 替换 ULID 类: 改为生成 26 字符 hex
    class _PatchedULID:
        _v_patched = True
        def __new__(cls):
            return _ulid_str()
        def __init__(self, *args, **kwargs):
            pass
        def __str__(self):
            return _ulid_str()
    ulid_mod.ULID = _PatchedULID
    return True


def _make_patched_ulid_class():
    """返回 patch 后的 ULID 类. 多次调用返回相同类."""
    class _PatchedULID:
        _v_patched = True
        def __new__(cls):
            return _ulid_str()
        def __init__(self, *args, **kwargs):
            pass
        def __str__(self):
            return _ulid_str()
    return _PatchedULID


def patch_ulid_in_module(target_module) -> bool:
    """在指定 module 的 namespace 里 patch ULID 名称.

    使用场景: datalake.py 里 `from ulid import ULID` 是复制引用, 修改
    sys.modules['ulid'].ULID 不会影响 datalake.ULID 名字. 必须直接
    修改 `target_module.ULID = patched_ULID_class`.
    """
    if target_module is None:
        return False
    if getattr(getattr(target_module, "ULID", None), "_v_patched", False):
        return True
    target_module.ULID = _make_patched_ulid_class()
    return True
