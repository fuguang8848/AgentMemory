# agentmemory/config/loader.py
"""Multi-source configuration loader with priority resolution.

Priority (high→low): CLI args > ENV > config= > ~/.agentmemory/config.toml > built-in defaults.
Supported formats: TOML > YAML > JSON > ENV.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_CONFIG_TOML, DEFAULT_CONFIG_PATH, ENV_PREFIX
from .schema import AgentMemoryConfig

# Try importing TOML libraries (tomllib for 3.11+, tomli for <3.11)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# Try importing YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def _expand_path(path: str | None) -> Path | None:
    """Expand ~ and environment variables in path."""
    if path is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _load_toml(content: str) -> dict[str, Any]:
    """Parse TOML string to dict."""
    if tomllib is None:
        raise ImportError("tomllib or tomli is required for TOML support")
    return tomllib.loads(content)


def _load_yaml(content: str) -> dict[str, Any]:
    """Parse YAML string to dict."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required for YAML support")
    return yaml.safe_load(content) or {}


def _load_json(content: str) -> dict[str, Any]:
    """Parse JSON string to dict."""
    return json.loads(content)


def _env_to_nested_dict(env_vars: dict[str, str]) -> dict[str, Any]:
    """Convert environment variables with AGENTMEMORY_* prefix to nested dict.
    
    Example: AGENTMEMORY_LLM__PROVIDER=anthropic -> {"llm": {"provider": "anthropic"}}
    """
    result: dict[str, Any] = {}
    prefix = ENV_PREFIX
    
    for key, value in env_vars.items():
        if not key.startswith(prefix):
            continue
        
        # Remove prefix and split by __ for nested keys
        remaining = key[len(prefix):]
        parts = remaining.split("__")
        
        # Convert to lowercase for case-insensitive matching
        current = result
        for i, part in enumerate(parts[:-1]):
            part = part.lower()
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value (try to parse as JSON, otherwise use string)
        final_key = parts[-1].lower()
        try:
            current[final_key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            current[final_key] = value
    
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _detect_format(path: Path) -> str:
    """Detect config file format from extension."""
    ext = path.suffix.lower()
    if ext == ".toml":
        return "toml"
    elif ext in (".yaml", ".yml"):
        return "yaml"
    elif ext == ".json":
        return "json"
    else:
        # Default to TOML for .config or no extension
        return "toml"


def _load_file(path: Path) -> dict[str, Any]:
    """Load configuration file based on detected format."""
    format_type = _detect_format(path)
    content = path.read_text(encoding="utf-8")
    
    if format_type == "toml":
        return _load_toml(content)
    elif format_type == "yaml":
        return _load_yaml(content)
    elif format_type == "json":
        return _load_json(content)
    else:
        raise ValueError(f"Unsupported config format: {format_type}")


def load_config(
    config_path: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
    **kwargs: Any,
) -> AgentMemoryConfig:
    """Load configuration with multi-source priority resolution.
    
    Priority (high→low): CLI kwargs > ENV > config_path > ~/.agentmemory/config.toml > defaults
    
    Args:
        config_path: Path to config file (TOML/YAML/JSON). If None, uses ~/.agentmemory/config.toml
        env_overrides: Dict of env variables (for testing). If None, reads from os.environ
        **kwargs: Explicit config overrides (highest priority)
    
    Returns:
        Validated AgentMemoryConfig instance
    """
    # 1. Start with built-in defaults
    config_data: dict[str, Any] = _load_toml(DEFAULT_CONFIG_TOML.strip())
    
    # 2. Load from ~/.agentmemory/config.toml if exists
    default_config_file = _expand_path(DEFAULT_CONFIG_PATH)
    if default_config_file and default_config_file.exists():
        try:
            file_config = _load_file(default_config_file)
            config_data = _deep_merge(config_data, file_config)
        except Exception as e:
            # Log warning but continue with defaults
            import logging
            logging.warning(f"Failed to load config from {default_config_file}: {e}")
    
    # 3. Load from explicit config_path if provided
    if config_path is not None:
        config_file = _expand_path(str(config_path)) if isinstance(config_path, str) else config_path
        if config_file and config_file.exists():
            try:
                file_config = _load_file(config_file)
                config_data = _deep_merge(config_data, file_config)
            except Exception as e:
                import logging
                logging.warning(f"Failed to load config from {config_file}: {e}")
    
    # 4. Apply environment variable overrides (AGENTMEMORY_* prefix)
    env_vars = env_overrides if env_overrides is not None else dict(os.environ)
    env_config = _env_to_nested_dict(env_vars)
    config_data = _deep_merge(config_data, env_config)
    
    # 5. Apply explicit kwargs (highest priority)
    if kwargs:
        config_data = _deep_merge(config_data, kwargs)
    
    # 6. Validate and return
    return AgentMemoryConfig.model_validate(config_data)


def get_config(**kwargs: Any) -> AgentMemoryConfig:
    """Convenience function to get config with optional overrides."""
    return load_config(**kwargs)


# Alias for backward compatibility
Config = AgentMemoryConfig

__all__ = ["load_config", "get_config", "Config", "AgentMemoryConfig"]
