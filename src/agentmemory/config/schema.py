# agentmemory/config/schema.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "bailian", "minimax",
                       "ollama", "vllm", "lmstudio", "gemini", "mistral",
                       "groq", "deepseek", "local_gguf", "mock"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None           # 留空则读 ENV
    base_url: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    fallback_chain: list[str] = Field(default_factory=list)  # 失败时回退

class EmbedderConfig(BaseModel):
    provider: Literal["openai", "bge", "minilm", "m3e", "fastembed",
                       "cohere", "sentence_transformers", "mock"] = "minilm"
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dim: int = 384
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 32
    cache: bool = True
    cache_ttl_seconds: int = 86400 * 7

class VectorStoreConfig(BaseModel):
    provider: Literal["faiss", "sqlite_vec", "qdrant", "chroma",
                       "lancedb", "milvus", "pgvector", "pinecone", "memory"] = "faiss"
    path: str = "~/.agentmemory/vectors"
    collection: str = "agentmemory"
    metric: Literal["cosine", "ip", "l2"] = "cosine"
    index_params: dict[str, Any] = Field(default_factory=dict)
    host: str | None = None              # qdrant/milvus/pinecone
    port: int | None = None
    api_key: str | None = None

class GraphStoreConfig(BaseModel):
    provider: Literal["networkx", "kuzu", "neo4j", "memgraph", "memory"] = "networkx"
    path: str = "~/.agentmemory/graph"
    uri: str | None = None               # neo4j
    user: str | None = None
    password: str | None = None

class FileStoreConfig(BaseModel):
    provider: Literal["local_fs", "s3", "oss"] = "local_fs"
    path: str = "~/.agentmemory/memory"
    s3_bucket: str | None = None
    s3_region: str | None = None
    oss_bucket: str | None = None
    oss_endpoint: str | None = None
    encrypt: bool = False

class StorageConfig(BaseModel):
    provider: Literal["sqlite", "postgres", "duckdb", "memory"] = "sqlite"
    path: str = "~/.agentmemory/agentmemory.db"
    dsn: str | None = None               # postgres
    wal: bool = True                     # sqlite WAL 模式

class DecayConfig(BaseModel):
    policy: Literal["half_life", "importance_only", "adaptive", "none"] = "half_life"
    half_life_days: float = 14.0
    forget_threshold: float = 0.3
    archive_threshold: float = 0.5
    max_archive_size: int = 10_000
    schedule: str = "0 3 * * *"          # cron 表达式（默认每天凌晨 3 点）

class RetrievalConfig(BaseModel):
    default_strategy: list[str] = Field(
        default_factory=lambda: ["vector", "bm25", "importance"])
    weights: dict[str, float] = Field(
        default_factory=lambda: {"vector": 0.6, "bm25": 0.3, "importance": 0.1})
    rerank: bool = False
    reranker: str = "identity"
    top_k: int = 5
    min_score: float = 0.0
    graph_hops: int = 0

class SecurityConfig(BaseModel):
    pii_redact: bool = True
    pii_types: list[str] = Field(default_factory=lambda: [
        "phone", "email", "id_card", "credit_card", "ip", "mac"])
    encryption: bool = False
    encryption_key_env: str = "AGENTMEMORY_ENCRYPTION_KEY"
    rate_limit_per_minute: int = 0        # 0 = 不限
    circuit_breaker_threshold: int = 5

class ObservabilityConfig(BaseModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    otel_endpoint: str | None = None      # jaeger/tempo/otlp
    otel_service_name: str = "agentmemory"
    prometheus_port: int | None = None    # 暴露 /metrics
    redact_pii_in_logs: bool = True

class MiddlewareConfig(BaseModel):
    chain: list[str] = Field(default_factory=lambda: [
        "logging", "tracing", "metrics", "pii_redact", "ratelimit"])

class TenantConfig(BaseModel):
    default_tenant: str = "default"
    default_namespace: str = "default"
    enforce_isolation: bool = True
    cross_tenant_scope: list[str] = Field(default_factory=lambda: ["admin"])

class AgentMemoryConfig(BaseModel):
    version: str = "2.0"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    file_store: FileStoreConfig = Field(default_factory=FileStoreConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    middleware: MiddlewareConfig = Field(default_factory=MiddlewareConfig)
    tenant: TenantConfig = Field(default_factory=TenantConfig)
    custom: dict[str, Any] = Field(default_factory=dict)
