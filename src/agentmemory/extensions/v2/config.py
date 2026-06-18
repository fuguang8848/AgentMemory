"""
配置管理 - AgentMemory v2.0

v2.0 Config Schema:
- MemoryConfig: 顶层配置容器
- ProvidersConfig: LLM/Embedder/VectorStore provider 配置
- DataLakeConfig: DataLake 配置
- EmbeddingStateConfig: Embedding State Machine 配置
- MultiAgentConfig: MultiAgent 配置（规划中）
- DecayConfig: Decay 配置
- TieredLogConfig: TieredLog 配置

支持 YAML 加载和 v1.0 自动迁移。
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Literal


# =============================================================================
# v2.0 Config Dataclasses
# =============================================================================

@dataclass
class LLMProviderConfig:
    """LLM Provider 配置"""
    type: str = "bailian"
    model: str = "qwen3.6-plus"
    api_base: str = "https://token-plan.cn-beijing.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""


@dataclass
class EmbedderProviderConfig:
    """Embedder Provider 配置"""
    type: str = "dashscope"
    model: str = "text-embedding-v3"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    dimensions: int = 1024


@dataclass
class VectorStoreConfig:
    """VectorStore 配置"""
    type: str = "usearch"
    metric: str = "cosine"
    dimensions: int = 1024


@dataclass
class ProvidersConfig:
    """所有 Provider 的配置容器"""
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    embedder: EmbedderProviderConfig = field(default_factory=EmbedderProviderConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


@dataclass
class DataLakeConfig:
    """DataLake 配置"""
    enabled: bool = True
    data_root: str = "./memory_library"
    auto_commit: bool = True
    commit_interval_seconds: int = 300


@dataclass
class EmbeddingStateConfig:
    """Embedding State Machine 配置"""
    enabled: bool = True
    batch_size: int = 32
    retry_attempts: int = 3
    retry_delay_seconds: int = 5


@dataclass
class MultiAgentConfig:
    """MultiAgent 配置（规划中）"""
    enabled: bool = False
    agent_count: int = 3


@dataclass
class DecayConfig:
    """Decay 遗忘引擎配置（与 v2-architecture.md §5.9 DecayPolicy 一致）"""
    enabled: bool = True
    forget_threshold: float = 0.2  # 低于此值遗忘
    archive_threshold: float = 0.5  # 低于此值归档
    half_life_days: float = 30.0   # 半衰期 30 天（架构要求）
    max_archive_size: int = 1000


@dataclass
class TieredLogConfig:
    """TieredLog 配置"""
    enabled: bool = True
    hot_ttl_seconds: int = 3600  # 1小时
    warm_ttl_seconds: int = 86400  # 1天
    cold_ttl_seconds: int = 604800  # 7天
    max_hot_size: int = 1000
    max_warm_size: int = 5000
    max_cold_size: int = 50000


@dataclass
class LibraryConfig:
    """Library 分类白名单配置（§5.2）"""
    enabled: bool = True
    whitelist_path: str = "./agentmemory/library_seeds.json"
    max_depth: int = 4


@dataclass
class HybridSearchConfig:
    """混合搜索配置（§3.2 双轨检索）"""
    fusion_method: Literal["rrf", "rrf_weighted", "score_sum", "score_avg"] = "rrf_weighted"
    rrf_k: int = 60
    vector_weight: float = 0.5
    library_weight: float = 0.3
    tag_weight: float = 0.2
    min_score_threshold: float = 0.01  # RRF 分数低于此值不返回
    max_results_per_track: int = 50    # 每条轨最多返回多少结果


@dataclass
class MemoryConfig:
    """
    AgentMemory v2.0 顶层配置
    
    Attributes:
        version: 配置版本号
        data_root: 数据根目录
        providers: Provider 配置
        datalake: DataLake 配置
        embedding_state_machine: Embedding State Machine 配置
        multi_agent: MultiAgent 配置
        decay: Decay 配置
        tiered_log: TieredLog 配置
    """
    version: str = "2.0.0"
    data_root: str = "./memory_library"
    
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    datalake: DataLakeConfig = field(default_factory=DataLakeConfig)
    embedding_state_machine: EmbeddingStateConfig = field(default_factory=EmbeddingStateConfig)
    multi_agent: MultiAgentConfig = field(default_factory=MultiAgentConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    tiered_log: TieredLogConfig = field(default_factory=TieredLogConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    hybrid_search: HybridSearchConfig = field(default_factory=HybridSearchConfig)

    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryConfig":
        """从字典创建配置"""
        if data is None:
            return cls()
        
        # v2.0 直接结构
        if "version" in data and data["version"] == "2.0.0":
            return cls(
                version=data.get("version", "2.0.0"),
                data_root=data.get("data_root", "./memory_library"),
                providers=ProvidersConfig(**data.get("providers", {})) 
                          if isinstance(data.get("providers"), dict) else data.get("providers", ProvidersConfig()),
                datalake=DataLakeConfig(**data.get("datalake", {}))
                         if isinstance(data.get("datalake"), dict) else data.get("datalake", DataLakeConfig()),
                embedding_state_machine=EmbeddingStateConfig(**data.get("embedding_state_machine", {}))
                                        if isinstance(data.get("embedding_state_machine"), dict) else data.get("embedding_state_machine", EmbeddingStateConfig()),
                multi_agent=MultiAgentConfig(**data.get("multi_agent", {}))
                           if isinstance(data.get("multi_agent"), dict) else data.get("multi_agent", MultiAgentConfig()),
                decay=DecayConfig(**data.get("decay", {}))
                      if isinstance(data.get("decay"), dict) else data.get("decay", DecayConfig()),
                tiered_log=TieredLogConfig(**data.get("tiered_log", {}))
                          if isinstance(data.get("tiered_log"), dict) else data.get("tiered_log", TieredLogConfig()),
                library=LibraryConfig(**data.get("library", {}))
                          if isinstance(data.get("library"), dict) else data.get("library", LibraryConfig()),
                hybrid_search=HybridSearchConfig(**data.get("hybrid_search", {}))
                          if isinstance(data.get("hybrid_search"), dict) else data.get("hybrid_search", HybridSearchConfig()),
            )
        
        # v1.0 迁移
        return cls._migrate_from_v1(data)

    @classmethod
    def _migrate_from_v1(cls, v1_data: dict) -> "MemoryConfig":
        """
        从 v1.0 schema 迁移到 v2.0
        
        v1.0 结构示例:
        {
            "embedding": {"provider": "dashscope", "model": "text-embedding-v3", ...},
            "llm": {"provider": "bailian", "model": "qwen3.6-plus", ...},
            "decay": {"enabled": True, "half_life_days": 14.0, ...},
            "layers": {"l1_compress": True, "l2_graph": True, ...},
            "storage": {"data_dir": "data", "memory_dir": "memory"}
        }
        """
        v1_embedding = v1_data.get("embedding", {})
        v1_llm = v1_data.get("llm", {})
        v1_decay = v1_data.get("decay", {})
        v1_storage = v1_data.get("storage", {})
        v1_layers = v1_data.get("layers", {})
        
        # providers
        providers = ProvidersConfig(
            llm=LLMProviderConfig(
                type=v1_llm.get("provider", "bailian"),
                model=v1_llm.get("model", "qwen3.6-plus"),
                api_base=v1_llm.get("base_url", "https://token-plan.cn-beijing.aliyuncs.com/compatible-mode/v1"),
                api_key="",  # 从环境变量读取
            ),
            embedder=EmbedderProviderConfig(
                type=v1_embedding.get("provider", "dashscope"),
                model=v1_embedding.get("model", "text-embedding-v3"),
                api_base=v1_embedding.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                api_key="",  # 从环境变量读取
                dimensions=v1_embedding.get("dimensions", 1024),
            ),
            vector_store=VectorStoreConfig(
                type="usearch",
                metric="cosine",
                dimensions=v1_embedding.get("dimensions", 1024),
            ),
        )
        
        # datalake
        datalake = DataLakeConfig(
            enabled=v1_layers.get("l3_vector", True),
            data_root=v1_storage.get("data_dir", "data"),
        )
        
        # decay
        decay = DecayConfig(
            enabled=v1_decay.get("enabled", True),
            forget_threshold=v1_decay.get("threshold", 0.3),  # v1 threshold → v2 forget_threshold
            archive_threshold=v1_decay.get("archive_threshold", 0.5),
            half_life_days=v1_decay.get("half_life_days", 30.0),  # v2 默认 30 天
            max_archive_size=v1_decay.get("max_archive_size", 1000),
        )
        
        # tiered_log - 基于 layers.l4_files
        tiered_log = TieredLogConfig(
            enabled=v1_layers.get("l4_files", True),
        )
        
        return cls(
            version="2.0.0",
            data_root=v1_storage.get("data_dir", "data"),
            providers=providers,
            datalake=datalake,
            decay=decay,
            tiered_log=tiered_log,
        )

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "MemoryConfig":
        """从 YAML 文件加载配置"""
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data or {})
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config. Install with: pip install pyyaml")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        except Exception as e:
            raise ValueError(f"Failed to load config from {yaml_path}: {e}")

    @classmethod
    def load_from_json(cls, json_path: str) -> "MemoryConfig":
        """从 JSON 文件加载配置"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_api_key(self, env_var: str) -> str:
        """从环境变量获取 API Key"""
        return os.environ.get(env_var, "")

    def validate(self) -> None:
        """
        验证配置合法性
        
        Raises:
            ValueError: 配置不合法
        """
        errors = []
        
        # 版本检查
        if not self.version:
            errors.append("version cannot be empty")
        elif self.version != "2.0.0":
            errors.append(f"unsupported config version: {self.version} (supported: 2.0.0)")
        
        # decay 阈值检查
        if not 0.0 <= self.decay.forget_threshold <= 1.0:
            errors.append(f"decay.forget_threshold must be 0-1, got: {self.decay.forget_threshold}")
        if not 0.0 <= self.decay.archive_threshold <= 1.0:
            errors.append(f"decay.archive_threshold must be 0-1, got: {self.decay.archive_threshold}")
        if self.decay.forget_threshold > self.decay.archive_threshold:
            errors.append(f"decay.forget_threshold ({self.decay.forget_threshold}) must be <= archive_threshold ({self.decay.archive_threshold})")
        
        # 半衰期检查
        if self.decay.half_life_days <= 0:
            errors.append(f"decay.half_life_days must be positive, got: {self.decay.half_life_days}")
        
        # dimensions 检查
        if self.providers.embedder.dimensions <= 0:
            errors.append(f"providers.embedder.dimensions must be positive, got: {self.providers.embedder.dimensions}")
        
        if errors:
            raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))


# =============================================================================
# Legacy Config（v1.0 兼容）
# =============================================================================

# v1.0 默认配置（用于迁移时参考）
LEGACY_DEFAULT_CONFIG = {
    "version": "1.0.0",
    "embedding": {
        "provider": "dashscope",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "text-embedding-v3",
        "dimensions": 1024,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "llm": {
        "provider": "bailian",
        "api_key_env": "BAILIAN_API_KEY",
        "model": "qwen3.6-plus",
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    },
    "decay": {
        "enabled": True,
        "threshold": 0.3,
        "archive_threshold": 0.5,
        "half_life_days": 14.0,
        "max_archive_size": 1000
    },
    "hybrid_search": {
        "vector_weight": 0.6,
        "bm25_weight": 0.3,
        "importance_weight": 0.1
    },
    "layers": {
        "l1_compress": True,
        "l2_graph": True,
        "l3_vector": True,
        "l4_files": True
    },
    "storage": {
        "data_dir": "data",
        "memory_dir": "memory"
    }
}


class Config:
    """
    兼容层：支持 v1.0 风格的字典配置
    
    新代码应使用 MemoryConfig。
    """
    
    def __init__(self, config_path: str = None):
        self.config = LEGACY_DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                self._deep_merge(self.config, user_config)
    
    def _deep_merge(self, base: dict, update: dict):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, path: str, default=None):
        """Get config by dot-separated path, e.g. 'embedding.model'"""
        keys = path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_api_key(self, env_var: str) -> str:
        return os.environ.get(env_var, "")

    def get_storage_path(self, relative_path: str) -> str:
        pkg_dir = Path(__file__).parent.resolve()
        data_dir = self.config.get("storage", {}).get("data_dir", "src/data")
        base = (pkg_dir / data_dir).resolve()
        return str(base / relative_path)


# =============================================================================
# 全局单例
# =============================================================================

_memory_config: Optional[MemoryConfig] = None


def get_memory_config() -> MemoryConfig:
    """获取 MemoryConfig 全局单例"""
    global _memory_config
    if _memory_config is None:
        _memory_config = MemoryConfig()
    return _memory_config


_config: Optional[Config] = None


def get_config(config_path: str = None) -> Config:
    """获取 v1.0 兼容 Config 全局单例"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
