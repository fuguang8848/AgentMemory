"""Adapters — 远端 v2.0.1 适配到本地 core ABC

桥接原则（按 "Adapter Pattern"）:
- 远端不直接依赖本地（保持远端可独立使用）
- 本地通过适配层复用远端实现
- 失败时降级到本地默认实现
"""
from __future__ import annotations

# 占位 - 子模块按需导入
__all__: list[str] = []
