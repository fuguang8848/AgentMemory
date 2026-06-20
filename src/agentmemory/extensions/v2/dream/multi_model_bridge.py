"""
MultiModelBridge: 跨模型记忆同步桥接器
========================================

支持多模型共享记忆，同时保留各模型特定偏好的记忆同步系统。

核心概念：
- ModelIdentity: 模型身份标识（能力、偏好风格）
- MemorySlice: 记忆切片（共享标签 + 模型特定数据）
- MultiModelMemoryBridge: 跨模型记忆同步核心引擎

使用示例：
    bridge = MultiModelMemoryBridge()
    bridge.register_model("gpt4", provider="openai", capabilities=["reasoning", "coding"])
    bridge.register_model("claude", provider="anthropic", capabilities=["writing", "analysis"])
    
    # 存储跨模型共享记忆
    bridge.store("project_context", {"shared": True}, tags=["project"])
    
    # 按模型查询
    results = bridge.query("project_context", target_model="gpt4")
"""

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import threading


@dataclass
class ModelIdentity:
    """模型身份标识"""
    model_id: str                          # 模型唯一标识
    provider: str                          # 提供商 (openai, anthropic, etc.)
    capabilities: list[str] = field(default_factory=list)   # 能力列表
    preferred_style: str = "balanced"      # 偏好风格: concise, balanced, detailed
    metadata: dict[str, Any] = field(default_factory=dict)   # 额外元数据
    
    def has_capability(self, cap: str) -> bool:
        """检查是否具备某能力"""
        return cap.lower() in [c.lower() for c in self.capabilities]
    
    def supports_tags(self, tags: list[str]) -> bool:
        """检查是否支持指定标签的记忆"""
        return True  # 默认全部支持，可按需细化


@dataclass
class MemorySlice:
    """记忆切片：跨模型共享 + 模型特定数据"""
    memory_id: str                         # 记忆唯一ID
    shared_tags: list[str] = field(default_factory=list)    # 共享标签
    model_specific: dict[str, dict[str, Any]] = field(default_factory=dict)  # model_id -> 数据
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def get_for_model(self, model_id: str, default: Any = None) -> Any:
        """获取指定模型的数据"""
        return self.model_specific.get(model_id, default)
    
    def set_for_model(self, model_id: str, data: dict[str, Any]) -> None:
        """设置指定模型的数据"""
        self.model_specific[model_id] = data
        self.updated_at = datetime.now()
    
    def has_tag(self, tag: str) -> bool:
        """检查是否包含指定标签"""
        return tag in self.shared_tags


class MultiModelMemoryBridge:
    """
    跨模型记忆同步桥接器
    
    功能：
    - register_model: 注册模型身份
    - store: 存储记忆（支持跨模型共享）
    - query: 按模型或标签查询记忆
    - sync_preferences: 同步模型偏好设置
    - suggest_for_model: 为指定模型推荐记忆
    """
    
    def __init__(self):
        self._models: dict[str, ModelIdentity] = {}
        self._memories: dict[str, MemorySlice] = {}
        self._lock = threading.RLock()
    
    # === 模型管理 ===
    
    def register_model(
        self,
        model_id: str,
        provider: str,
        capabilities: list[str] | None = None,
        preferred_style: str = "balanced",
        metadata: dict[str, Any] | None = None
    ) -> ModelIdentity:
        """
        注册一个模型身份
        
        Args:
            model_id: 模型唯一标识
            provider: 提供商名称
            capabilities: 能力列表
            preferred_style: 偏好风格
            metadata: 额外元数据
        
        Returns:
            ModelIdentity 对象
        """
        with self._lock:
            identity = ModelIdentity(
                model_id=model_id,
                provider=provider,
                capabilities=capabilities or [],
                preferred_style=preferred_style,
                metadata=metadata or {}
            )
            self._models[model_id] = identity
            return identity
    
    def get_model(self, model_id: str) -> ModelIdentity | None:
        """获取模型身份"""
        return self._models.get(model_id)
    
    def list_models(self) -> list[ModelIdentity]:
        """列出所有注册的模型"""
        return list(self._models.values())
    
    # === 记忆存储 ===
    
    def store(
        self,
        memory_id: str,
        model_data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        shared_data: dict[str, Any] | None = None
    ) -> MemorySlice:
        """
        存储记忆切片
        
        Args:
            memory_id: 记忆唯一ID
            model_data: 包含所有模型数据的字典 {model_id: data}，或单个 dict（自动应用到所有模型）
            tags: 共享标签列表
            shared_data: 所有模型共享的通用数据
        
        Returns:
            MemorySlice 对象
        """
        with self._lock:
            if memory_id in self._memories:
                slice_ = self._memories[memory_id]
                if tags:
                    slice_.shared_tags = list(set(slice_.shared_tags + tags))
            else:
                slice_ = MemorySlice(
                    memory_id=memory_id,
                    shared_tags=tags or []
                )
                self._memories[memory_id] = slice_
            
            # 处理模型特定数据
            if model_data:
                if isinstance(model_data, dict):
                    # 检查是 model_id -> data 格式还是单个数据
                    first_val = next(iter(model_data.values()), None)
                    if isinstance(first_val, dict):
                        # model_id -> data 格式
                        for mid, data in model_data.items():
                            slice_.set_for_model(mid, data)
                    else:
                        # 单个数据，应用到所有已注册模型
                        for model_id in self._models:
                            slice_.set_for_model(model_id, model_data)
            
            if shared_data:
                # 共享数据同时写入所有模型
                for model_id in self._models:
                    existing = slice_.get_for_model(model_id, {})
                    existing.update(shared_data)
                    slice_.set_for_model(model_id, existing)
            
            return slice_
    
    def query(
        self,
        memory_id: str | None = None,
        tags: list[str] | None = None,
        target_model: str | None = None,
        min_access_count: int = 0
    ) -> list[MemorySlice]:
        """
        查询记忆切片
        
        Args:
            memory_id: 精确记忆ID（可选）
            tags: 需要匹配的标签（可选）
            target_model: 目标模型ID（可选）
            min_access_count: 最小访问次数
        
        Returns:
            匹配的 MemorySlice 列表
        """
        with self._lock:
            results = list(self._memories.values())
            
            if memory_id:
                results = [r for r in results if r.memory_id == memory_id]
            
            if tags:
                results = [r for r in results if any(r.has_tag(t) for t in tags)]
            
            if min_access_count > 0:
                results = [r for r in results if r.access_count >= min_access_count]
            
            # 增加目标模型的访问计数
            if target_model:
                for r in results:
                    r.access_count += 1
            
            return results
    
    def get(self, memory_id: str, model_id: str | None = None) -> MemorySlice | None:
        """获取单个记忆切片"""
        slice_ = self._memories.get(memory_id)
        if slice_ and model_id:
            slice_.access_count += 1
        return slice_
    
    # === 偏好同步 ===
    
    def sync_preferences(
        self,
        source_model: str,
        target_model: str,
        preference_keys: list[str] | None = None
    ) -> dict[str, Any]:
        """
        从源模型同步偏好到目标模型
        
        Args:
            source_model: 源模型ID
            target_model: 目标模型ID
            preference_keys: 要同步的偏好键列表（None=全部）
        
        Returns:
            同步的偏好数据
        """
        with self._lock:
            synced = {}
            
            for memory in self._memories.values():
                source_data = memory.get_for_model(source_model, {})
                target_data = memory.get_for_model(target_model, {})
                
                if preference_keys:
                    for key in preference_keys:
                        if key in source_data:
                            target_data[key] = source_data[key]
                            synced[key] = source_data[key]
                else:
                    target_data.update(source_data)
                    synced = source_data.copy()
                
                memory.set_for_model(target_model, target_data)
            
            return synced
    
    # === 记忆推荐 ===
    
    def suggest_for_model(
        self,
        model_id: str,
        tags: list[str] | None = None,
        limit: int = 5
    ) -> list[tuple[MemorySlice, float]]:
        """
        为指定模型推荐相关记忆
        
        Args:
            model_id: 目标模型ID
            tags: 偏好的标签列表
            limit: 返回数量上限
        
        Returns:
            (MemorySlice, relevance_score) 列表，按相关度排序
        """
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                return []
            
            candidates = list(self._memories.values())
            
            # 计算相关度分数
            scored = []
            for mem in candidates:
                score = 0.0
                
                # 标签匹配
                if tags:
                    matching_tags = sum(1 for t in tags if mem.has_tag(t))
                    score += matching_tags * 10
                
                # 访问频率
                score += min(mem.access_count, 100) * 0.5
                
                # 模型特定数据存在
                if mem.get_for_model(model_id):
                    score += 20
                
                # 能力匹配
                for tag in mem.shared_tags:
                    for cap in model.capabilities:
                        if tag.lower() in cap.lower() or cap.lower() in tag.lower():
                            score += 5
                
                if score > 0:
                    scored.append((mem, score))
            
            # 按分数排序并限制数量
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
    
    # === 统计 ===
    
    def get_stats(self) -> dict[str, Any]:
        """获取桥接器统计信息"""
        with self._lock:
            total_access = sum(m.access_count for m in self._memories.values())
            return {
                "registered_models": len(self._models),
                "total_memories": len(self._memories),
                "total_access_count": total_access,
                "models": [m.model_id for m in self._models.values()],
                "memory_tags": list(set(
                    tag for mem in self._memories.values() for tag in mem.shared_tags
                ))
            }
    
    def clear(self) -> None:
        """清空所有记忆和模型注册"""
        with self._lock:
            self._models.clear()
            self._memories.clear()


# === 便捷函数 ===

_bridge_instance: MultiModelMemoryBridge | None = None
_instance_lock = threading.Lock()


def get_bridge() -> MultiModelMemoryBridge:
    """获取全局单例桥接器"""
    global _bridge_instance
    if _bridge_instance is None:
        with _instance_lock:
            if _bridge_instance is None:
                _bridge_instance = MultiModelMemoryBridge()
    return _bridge_instance


def reset_bridge() -> None:
    """重置全局桥接器（主要用于测试）"""
    global _bridge_instance
    with _instance_lock:
        _bridge_instance = None
