"""
AgentMemory HTTP REST API

基于 FastAPI 的 REST API 实现，提供记忆存储/查询/管理接口。
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from memory_manager import MemoryHermes
from errors import (
    MemoryError,
    NotFoundError,
    ValidationError,
    StorageError,
)

# ============================================================================
# Request/Response Models
# ============================================================================


class MemoryStoreRequest(BaseModel):
    """存储记忆请求"""
    content: str = Field(..., min_length=1, description="记忆内容")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性 0-1")
    metadata: Optional[dict] = Field(default=None, description="元数据")


class MemoryStoreResponse(BaseModel):
    """存储记忆响应"""
    memory_id: str
    ulid: str


class MemoryQueryRequest(BaseModel):
    """查询记忆请求"""
    query: str = Field(..., min_length=1, description="查询文本")
    limit: int = Field(default=5, ge=1, le=100, description="返回数量")


class MemoryResult(BaseModel):
    """记忆结果"""
    id: str
    content: str
    score: float
    layer: Optional[str] = None
    importance: Optional[float] = None
    fact_type: Optional[str] = None
    tags: list[str] = []


class MemoryQueryResponse(BaseModel):
    """查询记忆响应"""
    results: list[MemoryResult]


class MemoryDeleteResponse(BaseModel):
    """删除记忆响应"""
    success: bool
    memory_id: str


class StatsResponse(BaseModel):
    """统计信息响应"""
    total: int
    by_layer: dict
    decay_threshold: float
    archive_count: int


class SessionEndRequest(BaseModel):
    """会话结束请求"""
    summary: Optional[str] = None


class SessionEndResponse(BaseModel):
    """会话结束响应"""
    stored: int
    archived: int
    stats: dict


class DecayResponse(BaseModel):
    """遗忘检查响应"""
    forgotten: int
    archived: int
    remaining: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str


# ============================================================================
# FastAPI App
# ============================================================================


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="AgentMemory API",
        description="四层闭环记忆系统 HTTP REST API",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # MemoryHermes instance (lazy initialization)
    _mh: Optional[MemoryHermes] = None

    def get_mh() -> MemoryHermes:
        """获取 MemoryHermes 实例"""
        nonlocal _mh
        if _mh is None:
            _mh = MemoryHermes()
        return _mh

    # =========================================================================
    # Routes
    # =========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check():
        """健康检查"""
        return HealthResponse(status="ok", version="1.0.0")

    @app.post(
        "/v1/memories",
        response_model=MemoryStoreResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["memories"],
    )
    async def store_memory(request: MemoryStoreRequest):
        """存储新记忆"""
        try:
            mh = get_mh()
            memory_id = await mh.store(
                request.content,
                request.metadata or {},
                request.importance,
            )
            # Extract ULID from memory_id (format: mem_<ulid>)
            ulid = memory_id.split("_", 1)[-1] if "_" in memory_id else memory_id
            return MemoryStoreResponse(memory_id=memory_id, ulid=ulid)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except StorageError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Storage failed: {e}")

    @app.get(
        "/v1/memories",
        response_model=MemoryQueryResponse,
        tags=["memories"],
    )
    async def query_memories(
        query: str = Query(..., min_length=1, description="查询文本"),
        limit: int = Query(default=5, ge=1, le=100, description="返回数量"),
    ):
        """查询相关记忆"""
        try:
            mh = get_mh()
            results = await mh.query(query, limit)
            return MemoryQueryResponse(
                results=[
                    MemoryResult(
                        id=r.get("id", ""),
                        content=r.get("content", ""),
                        score=r.get("score", 0.0),
                        layer=r.get("layer"),
                        importance=r.get("importance"),
                        fact_type=r.get("fact_type"),
                        tags=r.get("tags", []),
                    )
                    for r in results
                ]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    @app.delete(
        "/v1/memories/{memory_id}",
        response_model=MemoryDeleteResponse,
        tags=["memories"],
    )
    async def delete_memory(memory_id: str):
        """删除记忆"""
        try:
            mh = get_mh()
            success = await mh.forget(memory_id, permanent=True)
            if not success:
                raise MemoryNotFoundError(f"Memory {memory_id} not found")
            return MemoryDeleteResponse(success=True, memory_id=memory_id)
        except NotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

    @app.get("/v1/stats", response_model=StatsResponse, tags=["system"])
    async def get_stats():
        """获取统计信息"""
        try:
            mh = get_mh()
            stats = mh.get_stats()

            # Extract relevant stats
            vector_stats = stats.get("vector", {})
            return StatsResponse(
                total=vector_stats.get("total", 0),
                by_layer={
                    "l1_compress": stats.get("layers", {}).get("l1_compress", False),
                    "l2_graph": stats.get("layers", {}).get("l2_graph", False),
                    "l3_vector": stats.get("layers", {}).get("l3_vector", False),
                    "l4_files": stats.get("layers", {}).get("l4_files", False),
                },
                decay_threshold=0.1,  # Default threshold
                archive_count=stats.get("archive", {}).get("count", 0),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stats failed: {e}")

    @app.post("/v1/session/end", response_model=SessionEndResponse, tags=["session"])
    async def session_end(request: SessionEndRequest):
        """会话结束处理"""
        try:
            mh = get_mh()
            await mh.on_session_end(request.summary)
            stats = mh.get_stats()
            return SessionEndResponse(
                stored=stats.get("vector", {}).get("total", 0),
                archived=stats.get("archive", {}).get("count", 0),
                stats=stats,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Session end failed: {e}")

    @app.post("/v1/decay", response_model=DecayResponse, tags=["system"])
    async def run_decay():
        """运行遗忘检查"""
        try:
            mh = get_mh()
            # Run decay check
            result = await mh.run_decay_check()
            stats = mh.get_stats()
            return DecayResponse(
                forgotten=result.get("forgotten", 0),
                archived=result.get("archived", 0),
                remaining=stats.get("vector", {}).get("total", 0),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Decay check failed: {e}")

    return app


# Module-level app instance
app = create_app()


# ============================================================================
# CLI entry point
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8765):
    """运行 API 服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
