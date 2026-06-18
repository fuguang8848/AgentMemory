"""
Pydantic v2 数据契约
定义 AgentMemory 核心数据结构

v2.0 架构：
- MemoryEntry: v2.0 主模型（schema_version=2）
- Memory: v1.0 遗留模型（schema_version=1），保留兼容
- Fact/Entity/Relation: v1.0 遗留类型，保守保留
"""
from datetime import datetime, timezone
from typing import Literal, Any

try:
    from pydantic import BaseModel, ConfigDict, Field, Extra
except ImportError:
    # 兼容未安装 pydantic 的环境
    BaseModel = object
    ConfigDict = dict
    Field = lambda **kw: lambda f: f
    Extra = None

try:
    import ulid
    ULIDType = ulid.ULID
except ImportError:
    # 兼容未安装 ulid 的环境
    ULIDType = str


def _default_ulid() -> str:
    """生成默认 ULID 字符串"""
    try:
        return str(ulid.ULID())
    except Exception:
        import time
        return f"00{int(time.time() * 1000):017d}"


def _default_utcnow() -> datetime:
    """获取当前 UTC 时间"""
    return datetime.now(timezone.utc)


class MemoryEntry(BaseModel):
    """v2.0 主记忆模型（§7.2 MemoryEntry）
    
    Attributes:
        id: ULID 唯一标识
        content: 记忆内容（UTF-8 Markdown）
        category: 分类路径（最多 4 层）
        tags: 标签列表
        metadata: 自由元数据
        importance: 重要性分数（0-1）
        embedding_state: 向量化状态
        created_at: 创建时间
        updated_at: 更新时间
        last_access_at: 最后访问时间
        access_count: 访问次数
        schema_version: 固定为 2
    """
    model_config = ConfigDict(
        extra="forbid",
        ser_json_timedelta="iso8601",
        ser_json_bytes="utf8",
    )

    id: str = Field(default_factory=_default_ulid, description="ULID 唯一标识")
    content: str = Field(..., min_length=1, max_length=100_000, description="记忆内容")
    category: list[str] = Field(..., min_length=1, max_length=4, description="分类路径")
    tags: list[str] = Field(default_factory=list, max_length=50, description="标签列表")
    metadata: dict = Field(default_factory=dict, description="自由元数据")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性分数")
    embedding_state: Literal["pending", "generating", "completed", "failed", "permanent_failure"] = "pending"
    created_at: datetime = Field(default_factory=_default_utcnow, description="创建时间 UTC")
    updated_at: datetime = Field(default_factory=_default_utcnow, description="更新时间 UTC")
    last_access_at: datetime | None = Field(default=None, description="最后访问时间")
    access_count: int = Field(default=0, description="访问次数")
    schema_version: Literal[2] = Field(default=2, description="Schema 版本")

    def to_dict(self) -> dict:
        """转换为字典"""
        return self.model_dump()


class Memory(BaseModel):
    """记忆基础模型
    
    Attributes:
        id: 唯一标识符（ULID 格式）
        content: 记忆内容
        importance: 重要性分数（0-1）
        created_at: 创建时间（UTC ISO 8601）
        schema_version: Schema 版本号（固定为 1）
    """
    model_config = ConfigDict(
        extra="forbid",
        ser_json_timedelta="iso8601",
        ser_json_bytes="utf8",
    )
    
    id: str = Field(default_factory=_default_ulid, description="ULID 唯一标识")
    content: str = Field(..., description="记忆内容")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性分数")
    created_at: datetime = Field(default_factory=_default_utcnow, description="创建时间 UTC")
    schema_version: Literal[1] = Field(default=1, description="Schema 版本")
    
    def to_dict(self) -> dict:
        """转换为字典（兼容旧代码）"""
        return self.model_dump()


class Fact(Memory):
    """事实模型（继承 Memory）
    
    表示一个事实性陈述，包含主语、谓语、宾语
    
    Attributes:
        subject: 主语
        predicate: 谓语
        object: 宾语
    """
    
    subject: str = Field(..., description="主语")
    predicate: str = Field(..., description="谓语")
    object: str = Field(..., description="宾语")
    
    def to_triple(self) -> tuple[str, str, str]:
        """转换为三元组格式"""
        return (self.subject, self.predicate, self.object)


class Entity(BaseModel):
    """实体模型
    
    表示图中的一个节点
    
    Attributes:
        id: 唯一标识符（ULID 格式）
        name: 实体名称
        type: 实体类型（如 person, org, location）
        attributes: 实体属性字典
    """
    model_config = ConfigDict(
        extra="forbid",
    )
    
    id: str = Field(default_factory=_default_ulid, description="ULID 唯一标识")
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")
    attributes: dict = Field(default_factory=dict, description="实体属性")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return self.model_dump()


class Relation(BaseModel):
    """关系模型
    
    表示图中的一个边
    
    Attributes:
        id: 唯一标识符（ULID 格式）
        source: 源实体 ID
        target: 目标实体 ID
        type: 关系类型
        weight: 关系权重（0-1）
    """
    model_config = ConfigDict(
        extra="forbid",
    )
    
    id: str = Field(default_factory=_default_ulid, description="ULID 唯一标识")
    source: str = Field(..., description="源实体 ID")
    target: str = Field(..., description="目标实体 ID")
    type: str = Field(..., description="关系类型")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="关系权重")
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return self.model_dump()
