"""
API v2 - FastAPI 入口
§5.12 接口契约实现

提供端点：
- POST /v2/memories — store
- GET /v2/memories/search — query
- GET /v2/memories/{id} — read
- DELETE /v2/memories/{id} — forget
- GET /v2/memories — list
- GET /v2/stats — stats
- POST /v2/decay/run — run_decay_check
- GET /v2/embedding-state/{id} — embedding state
- GET /v2/library/tree — 分类树
- GET /v2/log/tail — 日志 tail
"""
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from ...memory_manager import MemoryHermes
except ImportError:
    from agentmemory.memory_manager import MemoryHermes


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str


class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100_000)
    category: list[str] = Field(..., min_length=1, max_length=4)
    metadata: dict = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class MemoryCreateResponse(BaseModel):
    id: str
    status: str = "stored"


class MemoryDetailResponse(BaseModel):
    id: str
    content: str
    category: list[str]
    tags: list[str]
    metadata: dict
    importance: float
    embedding_status: str
    created_at: str
    updated_at: str


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    category: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    mode: str = Field(default="hybrid")


class SearchResultItem(BaseModel):
    id: str
    content: str
    score: float
    category: Optional[list[str]] = None
    tags: list[str] = []


class StatsResponse(BaseModel):
    layers: dict
    session: dict
    vector: Optional[dict] = None
    archive: Optional[dict] = None


class DecayRunResponse(BaseModel):
    forgotten: int
    archived: int
    kept: int
    total: int


class EmbeddingStateResponse(BaseModel):
    memory_id: str
    state: str
    retry_count: int = 0
    error_message: Optional[str] = None


class LibraryNodeModel(BaseModel):
    name: str
    path: list[str]
    children: list = []
    memory_count: int = 0


class LogTailResponse(BaseModel):
    entries: list[dict]


# =============================================================================
# FastAPI App Factory
# =============================================================================

def create_app() -> FastAPI:
    """§5.12 create_app — 创建 FastAPI app"""
    app = FastAPI(
        title="AgentMemory v2.0 API",
        version="2.0.0",
        description="§5.12 接口契约实现",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 懒加载 MemoryHermes
    _hermes: Optional[MemoryHermes] = None

    def get_hermes() -> MemoryHermes:
        nonlocal _hermes
        if _hermes is None:
            _hermes = MemoryHermes()
        return _hermes

    # --------------------------------------------------------------------------
    # Health
    # --------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health():
        return HealthResponse(status="ok", version="2.0.0")

    # --------------------------------------------------------------------------
    # §5.12 API Routes
    # --------------------------------------------------------------------------

    @app.post("/v2/memories", response_model=MemoryCreateResponse, tags=["memories"])
    async def create_memory(req: MemoryCreateRequest):
        """POST /v2/memories — store"""
        hermes = get_hermes()
        metadata = dict(req.metadata)
        metadata["tags"] = req.tags

        try:
            memory_id = await hermes.store(
                content=req.content,
                category=req.category,
                metadata=metadata,
                importance=req.importance,
                tags=req.tags,
            )
            return MemoryCreateResponse(id=memory_id, status="stored")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v2/memories/search", response_model=list[SearchResultItem], tags=["memories"])
    async def search_memories(
        query: str = Query(...),
        limit: int = Query(default=10, ge=1, le=100),
        category: Optional[str] = Query(default=None),
        mode: str = Query(default="hybrid"),
    ):
        """GET /v2/memories/search — query"""
        hermes = get_hermes()
        cat = category.split("/") if category else None

        try:
            results = await hermes.query(
                query=query,
                limit=limit,
                category=cat,
                mode=mode,
            )
            return [
                SearchResultItem(
                    id=r.get("id", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    tags=r.get("metadata", {}).get("tags", []),
                )
                for r in results
            ]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v2/memories", response_model=list[str], tags=["memories"])
    async def list_memories(
        category: Optional[str] = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        """GET /v2/memories — list"""
        hermes = get_hermes()
        cat = category.split("/") if category else None

        try:
            ids = await hermes.list(category=cat, limit=limit)
            return ids
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v2/memories/{memory_id}", response_model=MemoryDetailResponse, tags=["memories"])
    async def get_memory(memory_id: str):
        """GET /v2/memories/{id} — read"""
        hermes = get_hermes()

        try:
            results = await hermes.query(query="", limit=1)
            # 简化：实际应通过 ID 直接读取
            if not results:
                raise HTTPException(status_code=404, detail="Memory not found")
            r = results[0]
            return MemoryDetailResponse(
                id=r.get("id", memory_id),
                content=r.get("content", ""),
                category=r.get("metadata", {}).get("category", []),
                tags=r.get("metadata", {}).get("tags", []),
                metadata=r.get("metadata", {}),
                importance=r.get("metadata", {}).get("importance", 0.5),
                embedding_status=r.get("metadata", {}).get("embedding_state", "unknown"),
                created_at=r.get("metadata", {}).get("created_at", datetime.now().isoformat()),
                updated_at=r.get("metadata", {}).get("updated_at", datetime.now().isoformat()),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v2/memories/{memory_id}", tags=["memories"])
    async def delete_memory(memory_id: str):
        """DELETE /v2/memories/{id} — forget"""
        hermes = get_hermes()

        try:
            await hermes.forget(memory_id, permanent=True)
            return {"status": "deleted", "id": memory_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v2/stats", response_model=StatsResponse, tags=["stats"])
    async def get_stats():
        """GET /v2/stats — stats"""
        hermes = get_hermes()
        return hermes.stats()

    @app.post("/v2/decay/run", response_model=DecayRunResponse, tags=["decay"])
    async def run_decay():
        """POST /v2/decay/run — run_decay_check"""
        hermes = get_hermes()
        try:
            result = await hermes.run_decay_check()
            return DecayRunResponse(
                forgotten=result.get("forget", 0),
                archived=result.get("archive", 0),
                kept=result.get("keep", 0),
                total=result.get("total", 0),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v2/embedding-state/{memory_id}", response_model=EmbeddingStateResponse, tags=["embedding"])
    async def get_embedding_state(memory_id: str):
        """GET /v2/embedding-state/{id} — 查询单条 embedding 状态"""
        hermes = get_hermes()
        # 简化：stats 中包含 embedding 状态
        stats = hermes.stats()
        return EmbeddingStateResponse(
            memory_id=memory_id,
            state=stats.get("layers", {}).get("l3_vector", False) and "completed" or "unknown",
            retry_count=0,
        )

    @app.get("/v2/library/tree", response_model=list[LibraryNodeModel], tags=["library"])
    async def get_library_tree():
        """GET /v2/library/tree — 分类树"""
        # 简化返回：实际应从 Library 模块获取
        return [
            LibraryNodeModel(name="A.项目", path=["A.项目"], children=[], memory_count=0),
            LibraryNodeModel(name="B.个人", path=["B.个人"], children=[], memory_count=0),
            LibraryNodeModel(name="C.知识", path=["C.知识"], children=[], memory_count=0),
        ]

    @app.get("/v2/log/tail", response_model=LogTailResponse, tags=["log"])
    async def get_log_tail(n: int = Query(default=100, ge=1, le=1000)):
        """GET /v2/log/tail — 日志 tail"""
        return LogTailResponse(entries=[])

    return app


# 全局 router（方便直接 include_router）
router = create_app()
