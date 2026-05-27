"""
MemoryHermes - 顶尖记忆系统总管理器

融合 Hermes + Mem0 优点：
- Hermes: sync_turn, prefetch, on_session_end, Provider 抽象
- Mem0: LLM 事实提取, 混合检索, 遗忘算法

四层闭环：
L1: LCM 压缩层（对话 → 关键事实）
L2: Graph 图谱层（事实 → 实体关系）
L3: Vector 向量层（混合检索）
L4: Files 持久化层（记忆归档）
"""

import json
import asyncio
from datetime import datetime
from typing import Optional

from .config import get_config, Config
from .L1_lcm_compressor import LCMCompressor, FactType
from .L2_graph_store import GraphStore
from .L3_vector_store import VectorStore, HybridRetriever
from .L4_file_persist import FilePersistStore
from .decay_engine import DecayEngine, MemoryArchiver


class MemoryHermes:
    """
    顶尖记忆系统主类
    
    生命周期：
    1. turn_start  → prefetch() 预取相关记忆
    2. 生成回复
    3. turn_end    → sync_turn() 提取事实 + 写入记忆
    4. session_end → on_session_end() 关键决策总结
    5. 心跳触发    → 遗忘检查 + 归档低分记忆
    """
    
    def __init__(self, config_path: str = None):
        self.config = get_config(config_path)
        
        # 初始化各层
        self._init_layers()
        
        # 状态
        self._prefetch_cache = {}
        self._session_turns = []
        self._session_start = datetime.now()
    
    def _init_layers(self):
        """初始化所有层"""
        # L1: LCM 压缩层
        if self.config.get("layers.l1_compress", True):
            self.l1 = LCMCompressor(self.config)
        else:
            self.l1 = None
        
        # L2: Graph 图谱层
        if self.config.get("layers.l2_graph", True):
            self.graph = GraphStore(self.config)
        else:
            self.graph = None
        
        # L3: Vector 混合检索层
        if self.config.get("layers.l3_vector", True):
            self.vector = VectorStore(self.config)
            self.retriever = HybridRetriever(self.vector, self.config)
        else:
            self.vector = None
            self.retriever = None
        
        # L4: File 持久化层
        if self.config.get("layers.l4_files", True):
            self.files = FilePersistStore(self.config)
        else:
            self.files = None
        
        # 遗忘引擎
        if self.config.get("decay.enabled", True):
            self.decay = DecayEngine(self.config)
            self.archiver = MemoryArchiver(self.config)
        else:
            self.decay = None
            self.archiver = None
    
    # ==================== 核心 API ====================
    
    async def store(self, content: str, metadata: dict = None, importance: float = 0.5) -> str:
        """
        存储记忆（自动 LLM 事实提取 + 多层写入）
        
        Args:
            content: 记忆内容
            metadata: 元数据（source, tags, fact_type 等）
            importance: 重要性评分 0-1
        
        Returns:
            memory_id: 存储后的记忆 ID
        """
        if metadata is None:
            metadata = {}
        
        metadata["importance"] = importance
        metadata["stored_at"] = datetime.now().isoformat()
        
        # L1: 提取事实（如果启用）
        extracted_entities = []
        if self.l1:
            facts = await self.l1.extract_facts([content])
            for fact in facts:
                metadata["entities"] = fact.entities
                metadata["fact_type"] = fact.fact_type.value if hasattr(fact.fact_type, "value") else str(fact.fact_type)
                
                # L2: 写入图谱
                if self.graph and fact.entities:
                    for entity_name in fact.entities:
                        self._graph_add_entity(entity_name, metadata)
        
        # L3: 写入向量存储
        memory_id = None
        if self.vector:
            memory_id = self.vector.store(content, metadata, importance)
        
        # L4: 持久化到文件
        if self.files:
            self.files.store_fact(content, metadata)
        
        return memory_id
    
    async def query(self, query: str, limit: int = 5, filters: dict = None) -> list[dict]:
        """
        查询记忆（混合检索）
        
        Args:
            query: 查询字符串
            limit: 返回数量
            filters: 过滤条件
        
        Returns:
            list of {id, content, score, metadata}
        """
        results = []
        
        # L3: 混合检索
        if self.retriever:
            results = self.retriever.retrieve(query, limit=limit, filters=filters)
        
        # L4: 补充文件层搜索
        if self.files and len(results) < limit:
            file_results = self.files.search(query)
            for fr in file_results:
                if not any(r["id"] == fr.get("id") for r in results):
                    results.append(fr)
                    if len(results) >= limit:
                        break
        
        return results[:limit]
    
    async def prefetch(self, query: str) -> str:
        """
        预取相关记忆（后台异步，用于下次对话）
        
        参考 Hermes 的 prefetch 机制
        
        Returns:
            格式化的预取记忆字符串
        """
        if not self.retriever:
            return ""
        
        results = await self.query(query, limit=3)
        
        if not results:
            return ""
        
        # 格式化预取结果
        prefetch_text = "\n".join([
            f"- {r['content']}" for r in results
        ])
        
        # 缓存，供下次对话使用
        self._prefetch_cache[query] = prefetch_text
        return prefetch_text
    
    def get_prefetched(self, query: str = None) -> str:
        """获取预取的缓存记忆"""
        if query and query in self._prefetch_cache:
            return self._prefetch_cache[query]
        return "\n".join(self._prefetch_cache.values())
    
    async def forget(self, memory_id: str, permanent: bool = False) -> bool:
        """
        主动遗忘记忆
        
        Args:
            memory_id: 记忆 ID
            permanent: True=永久删除, False=归档
        
        Returns:
            是否成功
        """
        if self.vector:
            if permanent:
                self.vector.delete(memory_id)
            else:
                self.vector.update_importance(memory_id, 0.0)
        
        if self.archiver and not permanent:
            self.archiver.archive_to_deep_storage(memory_id)
        
        return True
    
    async def sync_turn(self, user_message: str, assistant_message: str) -> list[dict]:
        """
        同步对话轮到记忆（每轮对话后调用）
        
        参考 Hermes sync_turn 机制
        
        Returns:
            提取并存储的事实列表
        """
        self._session_turns.append({
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.now().isoformat()
        })
        
        if not self.l1:
            return []
        
        # LLM 提取事实
        conversation = [user_message, assistant_message]
        facts = await self.l1.extract_facts(conversation)
        
        stored = []
        for fact in facts:
            metadata = {
                "fact_type": fact.fact_type.value if hasattr(fact.fact_type, "value") else str(fact.fact_type),
                "entities": fact.entities,
                "importance": fact.importance,
                "source": "sync_turn"
            }
            
            memory_id = await self.store(fact.content, metadata, fact.importance)
            stored.append({
                "id": memory_id,
                "content": fact.content,
                "fact_type": metadata["fact_type"]
            })
        
        return stored
    
    async def on_session_end(self, summary: str = None) -> dict:
        """
        会话结束时调用
        
        参考 Hermes on_session_end 机制
        
        Args:
            summary: 可选的会话摘要
        
        Returns:
            会话总结统计
        """
        session_duration = (datetime.now() - self._session_start).total_seconds()
        
        stats = {
            "session_duration_seconds": session_duration,
            "total_turns": len(self._session_turns),
            "facts_stored": 0,
            "session_start": self._session_start.isoformat(),
            "session_end": datetime.now().isoformat()
        }
        
        # 统计 L3 存储
        if self.vector:
            v_stats = self.vector.get_stats()
            stats.update(v_stats)
        
        # 写入 L4 会话总结
        if self.files:
            self.files.append_session_summary(
                turn_count=stats["total_turns"],
                duration=session_duration,
                summary=summary
            )
        
        # 清理
        self._session_turns = []
        self._prefetch_cache = {}
        
        return stats
    
    async def run_decay_check(self) -> dict:
        """
        运行遗忘检查（心跳触发）
        
        Returns:
            遗忘统计 {forget: N, archive: N, keep: N}
        """
        if not self.decay or not self.vector:
            return {"status": "disabled"}
        
        # 获取所有记忆
        all_entries = self.vector.get_all_entries()
        
        # 计算遗忘
        result = self.decay.run_decay_check(all_entries)
        
        # 执行遗忘
        for memory_id in result.get("forget", []):
            await self.forget(memory_id, permanent=True)
        
        for memory_id in result.get("archive", []):
            await self.forget(memory_id, permanent=False)
        
        return {
            "forget": len(result.get("forget", [])),
            "archive": len(result.get("archive", [])),
            "keep": len(result.get("keep", [])),
            "total": len(all_entries)
        }
    
    def get_stats(self) -> dict:
        """获取记忆系统统计"""
        stats = {
            "layers": {
                "l1_compress": self.l1 is not None,
                "l2_graph": self.graph is not None,
                "l3_vector": self.vector is not None,
                "l4_files": self.files is not None
            },
            "session": {
                "turns": len(self._session_turns),
                "start": self._session_start.isoformat()
            }
        }
        
        if self.vector:
            stats["vector"] = self.vector.get_stats()
        
        if self.graph:
            stats["graph"] = self.graph.get_entity_count()
        
        if self.archiver:
            stats["archive"] = {
                "size": self.archiver.get_archive_size()
            }
        
        return stats
    
    def _graph_add_entity(self, name: str, metadata: dict):
        """辅助方法：添加实体到图谱"""
        if not self.graph:
            return
        
        # 简单实现，实际应该分析实体类型
        entity_type = "ENTITY"
        if "project" in metadata.get("fact_type", "").lower():
            entity_type = "PROJECT"
        elif "person" in metadata.get("fact_type", "").lower():
            entity_type = "PERSON"
        
        try:
            self.graph.add_entity(name, entity_type, properties=metadata)
        except Exception:
            pass  # 忽略重复实体
    
    # ==================== 兼容旧 API ====================
    
    async def execute(self, action: str, params: dict = None) -> dict:
        """
        兼容 AgentSymphony 技能接口
        
        Actions:
        - store: 存储记忆
        - query: 查询记忆  
        - query_by_type: 按类型查询
        - get_stats: 获取统计
        - forget: 遗忘
        - prefetch: 预取
        """
        if params is None:
            params = {}
        
        if action == "store":
            memory_id = await self.store(
                params.get("content", ""),
                params.get("metadata"),
                params.get("importance", 0.5)
            )
            return {"success": True, "id": memory_id}
        
        elif action == "query":
            results = await self.query(
                params.get("query", ""),
                params.get("limit", 5),
                params.get("filters")
            )
            return {"success": True, "results": results}
        
        elif action == "get_stats":
            return {"success": True, "stats": self.get_stats()}
        
        elif action == "forget":
            success = await self.forget(
                params.get("memory_id"),
                params.get("permanent", False)
            )
            return {"success": success}
        
        elif action == "prefetch":
            prefetched = await self.prefetch(params.get("query", ""))
            return {"success": True, "prefetched": prefetched}
        
        elif action == "session_end":
            stats = await self.on_session_end(params.get("summary"))
            return {"success": True, "stats": stats}
        
        return {"success": False, "error": f"Unknown action: {action}"}
