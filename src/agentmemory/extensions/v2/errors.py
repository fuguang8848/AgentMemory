"""
MemoryError 体系
定义 AgentMemory 专用异常类
"""
from typing import Any


class MemoryError(Exception):
    """顶层异常类，所有 AgentMemory 异常都继承自此类"""
    
    code: str = "E000"
    
    def __init__(
        self,
        message: str,
        code: str = None,
        context: dict = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.code
        self.context = context or {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r}, context={self.context!r})"
    
    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigError(MemoryError):
    """配置错误（E001）
    
    YAML 解析失败、字段缺失、类型错误等配置相关问题
    """
    code = "E001"


class ProviderError(MemoryError):
    """Provider 调用失败（E002）
    
    API key 缺失、rate limit、网络错误等 Provider 相关问题
    """
    code = "E002"


class StorageError(MemoryError):
    """存储错误（E003）
    
    文件 IO 失败、JSON 损坏、磁盘满等存储相关问题
    """
    code = "E003"


class ValidationError(MemoryError):
    """数据验证错误（E004）
    
    Schema 不匹配、ULID 格式错误等数据验证问题
    """
    code = "E004"


class NotFoundError(MemoryError):
    """未找到错误（E005）
    
    memory_id 不存在、entity 不存在等问题
    """
    code = "E005"


class PermissionError(MemoryError):
    """权限错误（E006）
    
    access denied、role mismatch 等权限相关问题
    """
    code = "E006"


class RateLimitError(MemoryError):
    """速率限制错误（E007）
    
    API 429、本地 queue 满等速率限制问题
    """
    code = "E007"
