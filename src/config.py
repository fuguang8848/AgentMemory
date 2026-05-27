"""
配置管理
"""
import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
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
        "data_dir": "src/data",
        "memory_dir": "memory"
    }
}


class Config:
    def __init__(self, config_path: str = None):
        self.config = DEFAULT_CONFIG.copy()
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
        base = Path(self.config.get("storage", {}).get("data_dir", "src/data"))
        return str(base / relative_path)


_config = None


def get_config(config_path: str = None) -> Config:
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
