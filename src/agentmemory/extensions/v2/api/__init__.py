"""
AgentMemory HTTP REST API 模块

提供 FastAPI 实现的 REST API 接口，供外部服务调用记忆系统。
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
