"""
MemoryHermes - 顶尖记忆系统总管理器

融合 Hermes + Mem0 优点:
- Hermes: sync_turn, prefetch, on_session_end, Provider 抽象
- Mem0: LLM 事实提取, 混合检索, 遗忘算法

四层闭环:
L1: LCM 压缩层(对话 → 关键事实)
L2: Graph 图谱层(事实 → 实体关系)
L3: Vector 向量层(混合检索)
L4: Files 持久化层(记忆归档)
"""

import json
import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from .config import get_config, Config
except ImportError:
    from config import get_config, Config
try:
    from .L3_vector_store import VectorStore, HybridRetriever
except ImportError:
    from L3_vector_store import VectorStore, HybridRetriever
try:
    from .L4_file_persist import FilePersistStore
except ImportError:
    from L4_file_persist import FilePersistStore
try:
    from .decay_engine import DecayEngine, MemoryArchiver, create_decay_engine
except ImportError:
    from decay_engine import DecayEngine, MemoryArchiver, create_decay_engine
try:
    from .providers.llm import get_llm_provider
except ImportError:
    from providers.llm import get_llm_provider
try:
    from .providers.embedder import get_embedder
except ImportError:
    from providers.embedder import get_embedder


# ==============================================================================
# PreferenceStore — mem0 风格的轻量偏好存储
# ==============================================================================

class PreferenceStore:
    """
    基于 JSON 文件的用户偏好存储器（mem0 风格）。

    存储路径：~/.openclaw/workspace/memory/preferences.json

    偏好结构::
        {
            "<user_id>": {
                "<key>": {
                    "value": <any>,
                    "category": "working|episodic|long_term|procedural",
                    "updated_at": "<iso timestamp>"
                }
            }
        }
    """

    PREF_FILE = Path.home() / ".openclaw" / "workspace" / "memory" / "preferences.json"

    def __init__(self, pref_file: Path = None):
        self._pref_file = pref_file or self.PREF_FILE
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self._pref_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._pref_file.exists():
            self._pref_file.write_text("{}")

    def _read(self) -> Dict[str, Any]:
        try:
            return json.loads(self._pref_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        self._pref_file.parent.mkdir(parents=True, exist_ok=True)
        self._pref_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def store(self, user_id: str, key: str, value: Any, category: str = "general") -> None:
        """存储或更新用户偏好"""
        prefs = self._read()
        if user_id not in prefs:
            prefs[user_id] = {}
        prefs[user_id][key] = {
            "value": value,
            "category": category,
            "updated_at": datetime.now().isoformat(),
        }
        self._write(prefs)

    def get(self, user_id: str, key: str = None) -> Any:
        """获取用户偏好，key=None 时返回该用户所有偏好"""
        prefs = self._read()
        user_prefs = prefs.get(user_id, {})
        if key is None:
            return user_prefs
        return user_prefs.get(key, {}).get("value")

    def get_by_category(self, user_id: str, category: str) -> Dict[str, Any]:
        """获取指定类别的所有偏好"""
        prefs = self._read()
        user_prefs = prefs.get(user_id, {})
        return {
            k: v for k, v in user_prefs.items()
            if v.get("category") == category
        }

    def delete(self, user_id: str, key: str = None) -> bool:
        """删除偏好，key=None 时删除整用户偏好"""
        prefs = self._read()
        if key is None:
            if user_id in prefs:
                del prefs[user_id]
                self._write(prefs)
                return True
            return False
        if user_id in prefs and key in prefs[user_id]:
            del prefs[user_id][key]
            self._write(prefs)
            return True
        return False


# ==============================================================================
# MemoryHermes — 顶尖记忆系统主类
# ==============================================================================

class MemoryHermes:
    """
    顶尖记忆系统主类

    生命周期:
    1. turn_start  → prefetch() 预取相关记忆
    2. 生成回复
    3. turn_end    → sync_turn() 提取事实 + 写入记忆
    4. session_end → on_session_end() 关键决策总结
    5. 心跳触发    → 遗忘检查 + 归档低分记忆
    """

    def __init__(self, config_path: str = None, llm_provider: str = None, embedder_provider: str = None):
        self.config = get_config(config_path)

        # Provider 配置(运行时可覆盖)
        self._llm_provider_name = llm_provider or self.config.get("llm_provider", "bailian")
        self._embedder_provider_name = embedder_provider or self.config.get("embedder_provider", "dashscope")

        # 初始化各层(v2.0: L1/L2 已移除)
        self._init_layers()

        # v2.0: L1 LCM 和 L2 Graph 已移除,显式置空
        self.l1 = None
        self.graph = None

        # 状态
        self._prefetch_cache = {}
        self._session_turns = []
        self._session_start = datetime.now()

        # 偏好存储（延迟初始化）
        self._pref_store: Optional[PreferenceStore] = None

    def _init_layers(self):
        """初始化所有层(v2.0 修订版:L1/L2 已移除)"""
        # L1/L2: v2.0 已移除这些层
        self.l1 = None
        self.graph = None

        # L3: Vector 混合检索层
        if self.config.get("layers.l3_vector", True):
            vector_path = self.config.get_storage_path("vectors.json")

            # 获取 embedder provider(无 API key 时自动回退到 mock)
            embedder = get_embedder(
                model=self.config.get("embedding.model"),
            )

            self.vector = VectorStore(
                storage_path=vector_path,
                embedding_model=self.config.get("embedding.model"),
                embedding_dims=self.config.get("embedding.dimensions"),
                embedder_provider=embedder,
            )
            self.retriever = HybridRetriever(self.vector)
        else:
            self.vector = None
            self.retriever = None

        # L4: File 持久化层
        if self.config.get("layers.l4_files", True):
            workspace = self.config.config.get("storage", {}).get("memory_dir", "memory")
            self.files = FilePersistStore(workspace)
        else:
            self.files = None

        # 遗忘引擎(v2.0: 使用 create_decay_engine + DecayPolicy)
        if self.config.get("decay.enabled", True):
            self.decay = create_decay_engine(
                half_life_days=self.config.get("decay.half_life_days", 30.0),
                forget_threshold=self.config.get("decay.threshold", 0.2),
                archive_threshold=self.config.get("decay.archive_threshold", 0.5),
            )
            self.archiver = MemoryArchiver(
                max_archived=self.config.get("decay.max_archive_size", 1000),
            )
        else:
            self.decay = None
            self.archiver = None

    # ==================== 核心 API ====================

    async def store(self, content: str, metadata: dict = None, importance: float = 0.5) -> str:
        """
        存储记忆(自动 LLM 事实提取 + 多层写入)

        Args:
            content: 记忆内容
            metadata: 元数据(source, tags, fact_type 等)
            importance: 重要性评分 0-1

        Returns:
            memory_id: 存储后的记忆 ID
        """
        if metadata is None:
            metadata = {}

        metadata["importance"] = importance
        metadata["stored_at"] = datetime.now().isoformat()

        # L1: 提取实体(轻量解析,无需 LLM API)
        # 对于已结构化的事实,直接从 content 和 metadata 中提取 entities
        entities = metadata.get("entities", [])
        fact_type = metadata.get("fact_type", "general")

        # 如果提供了 entities 列表,写入图谱
        if self.graph and entities:
            for entity_name in entities:
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
        查询记忆(混合检索)

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
            file_results = self.files.search(query) if hasattr(self.files, 'search') else []
            for fr in file_results:
                if not any(r["id"] == fr.get("id") for r in results):
                    results.append(fr)
                    if len(results) >= limit:
                        break

        return results[:limit]

    async def prefetch(self, query: str) -> str:
        """
        预取相关记忆(后台异步,用于下次对话)

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

        # 缓存,供下次对话使用
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
            memory_data = self.vector.get(memory_id) if self.vector else None
            self.archiver.archive_to_deep_storage(memory_id, memory_data)

        return True

    async def sync_turn(self, user_message: str, assistant_message: str) -> list[dict]:
        """
        同步对话轮到记忆(每轮对话后调用)

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
        extractor = self.l1._get_extractor()
        facts = await extractor.extract_facts(conversation)

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
            session_entry = f"[会话总结] turns={stats['total_turns']}, duration={session_duration:.1f}s, summary={summary}"
            self.files.store_fact(session_entry, {"category": "general"})

        # 清理
        self._session_turns = []
        self._prefetch_cache = {}

        return stats

    async def run_decay_check(self) -> dict:
        """
        运行遗忘检查(心跳触发)

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
        for item in result.get("forget", []):
            memory_id = item["entry"]["id"]
            await self.forget(memory_id, permanent=True)

        for item in result.get("archive", []):
            memory_id = item["entry"]["id"]
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
            archived_list = self.archiver.list_archived()
            stats["archive"] = {
                "count": len(archived_list),
                "entries": archived_list[:10]  # 最近10条
            }

        return stats

    # ============================================================================
    # §5.11 MemoryHermes 接口契约补齐
    # ============================================================================

    async def list(
        self,
        category: list[str] | None = None,
        since=None,
        until=None,
        limit: int = 100,
    ) -> list[str]:
        """§5.11 list — 列出记忆 ID 列表

        Args:
            category: 可选的分类路径过滤
            since: 可选的起始时间
            until: 可选的结束时间
            limit: 返回数量上限

        Returns:
            list[str]: 记忆 ID 列表
        """
        # v2.0: 通过 vector store 列出
        if self.vector:
            all_entries = self.vector.get_all_entries()
            ids = [e["id"] for e in all_entries]
            return ids[:limit]
        return []

    def stats(self) -> dict:
        """§5.11 stats — 同步统计（别名 get_stats）"""
        return self.get_stats()

    async def close(self) -> None:
        """§5.11 close — 关闭资源（清理 asyncio tasks 等）"""
        # 清理 session turns
        self._session_turns = []
        self._prefetch_cache = {}
        # v2.0: 如果有 embedding worker task，尝试取消
        if hasattr(self, '_embedding_worker_task') and self._embedding_worker_task:
            self._embedding_worker_task.cancel()
            try:
                await self._embedding_worker_task
            except asyncio.CancelledError:
                pass

    def _graph_add_entity(self, name: str, metadata: dict):
        """辅助方法：添加实体到图谱（v2.0 已移除 GraphStore，此方法为空占位）"""
        # v2.0: L2 Graph 已移除，图谱功能由 TagIndex + Library 替代
        pass

    # ============================================================================
    # mem0 风格偏好学习 API
    # ============================================================================

    @property
    def pref_store(self) -> PreferenceStore:
        """懒加载偏好存储器"""
        if self._pref_store is None:
            self._pref_store = PreferenceStore()
        return self._pref_store

    def learn_preference(self, user_id: str, key: str, value: Any, category: str = "general") -> None:
        """
        存储用户偏好（mem0 风格）。

        Args:
            user_id: 用户 ID
            key: 偏好键名
            value: 偏好值
            category: 分类（working/episodic/long_term/procedural）
        """
        self.pref_store.store(user_id, key, value, category)

    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户所有偏好（mem0 风格）。

        Args:
            user_id: 用户 ID

        Returns:
            {key: {value, category, updated_at}, ...}
        """
        return self.pref_store.get(user_id)

    def query_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """
        按记忆类型检索（mem0 风格）。

        Args:
            memory_type: 类型（working/episodic/long_term/procedural）

        Returns:
            该类型的所有偏好记录列表
        """
        # 读取所有用户的所有偏好，过滤指定类型
        prefs = self._pref_store._read() if self._pref_store else {}
        results = []
        for uid, user_prefs in prefs.items():
            for key, item in user_prefs.items():
                if item.get("category") == memory_type:
                    results.append({
                        "user_id": uid,
                        "key": key,
                        **item,
                    })
        return results

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
