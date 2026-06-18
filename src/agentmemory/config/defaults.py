# agentmemory/config/defaults.py
"""Built-in default configuration (TOML format, matching ~/.agentmemory/config.toml)."""

DEFAULT_CONFIG_TOML = """
# agentmemory 2.0 默认配置
version = "2.0"

[llm]
provider = "openai"
model = "gpt-4o-mini"
# api_key 从 $OPENAI_API_KEY 读取
fallback_chain = ["anthropic", "ollama"]

[embedder]
provider = "minilm"
model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
dim = 384
cache = true

[vector_store]
provider = "faiss"
path = "~/.agentmemory/vectors"
collection = "agentmemory"
metric = "cosine"

[graph_store]
provider = "networkx"
path = "~/.agentmemory/graph"

[file_store]
provider = "local_fs"
path = "~/.agentmemory/memory"

[storage]
provider = "sqlite"
path = "~/.agentmemory/agentmemory.db"
wal = true

[decay]
policy = "half_life"
half_life_days = 14.0
forget_threshold = 0.3
archive_threshold = 0.5
schedule = "0 3 * * *"

[retrieval]
default_strategy = ["vector", "bm25", "importance"]
rerank = false
top_k = 5

[security]
pii_redact = true
encryption = false
rate_limit_per_minute = 0

[observability]
log_level = "INFO"
log_format = "json"
otel_service_name = "agentmemory"

[middleware]
chain = ["logging", "tracing", "metrics", "pii_redact", "ratelimit"]

[tenant]
default_tenant = "default"
default_namespace = "default"
enforce_isolation = true
"""

# Environment variable prefix for overrides
ENV_PREFIX = "AGENTMEMORY_"

# Default config file location
DEFAULT_CONFIG_PATH = "~/.agentmemory/config.toml"
