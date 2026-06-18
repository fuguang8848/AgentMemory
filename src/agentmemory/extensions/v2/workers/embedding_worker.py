"""
AgentMemory v2.0 - Embedding Worker

后台异步处理向量生成任务：
- 从 .embedding_state.json 取 pending 任务
- 调用 Embedder 生成向量
- 写入 VectorStore 索引
- 更新状态机，失败重试 3 次后升级为 permanent_failure
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Protocol, Callable, Awaitable

from ..providers.protocols import (
    BaseEmbedderProvider,
    BaseVectorStoreProvider,
    VectorEntry,
)
from ..providers.embedder import get_embedder
from ..providers.vectorstore import get_vectorstore


logger = logging.getLogger(__name__)


class EmbeddingStatus(str, Enum):
    """Embedding 任务状态"""
    PENDING = "pending"           # 待处理
    GENERATING = "generating"      # 生成中
    COMPLETED = "completed"        # 已完成
    FAILED = "failed"             # 失败（可重试）
    PERMANENT_FAILURE = "permanent_failure"  # 永久失败


@dataclass
class EmbeddingTask:
    """Embedding 任务"""
    id: str
    content: str
    metadata: dict
    status: EmbeddingStatus = EmbeddingStatus.PENDING
    retry_count: int = 0
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingTask":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            status=EmbeddingStatus(data.get("status", "pending")),
            retry_count=data.get("retry_count", 0),
            error_message=data.get("error_message"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            completed_at=data.get("completed_at"),
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


@dataclass
class EmbeddingState:
    """Embedding 状态机"""
    tasks: dict[str, EmbeddingTask]  # task_id -> task
    
    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingState":
        tasks = {
            id: EmbeddingTask.from_dict(t)
            for id, t in data.get("tasks", {}).items()
        }
        return cls(tasks=tasks)
    
    def to_dict(self) -> dict:
        return {
            "tasks": {
                id: task.to_dict()
                for id, task in self.tasks.items()
            }
        }


class EmbeddingWorker:
    """
    Embedding Worker
    
    后台异步处理向量生成任务，实现：
    - 状态机管理（pending → generating → completed/failed）
    - 失败重试（最多 3 次）
    - 永久失败升级
    - 批量处理
    """
    
    MAX_RETRIES = 3
    BATCH_SIZE = 32
    RETRY_DELAY_BASE = 2.0  # 基础重试延迟（秒）
    
    def __init__(
        self,
        state_path: str | Path = ".embedding_state.json",
        embedder: BaseEmbedderProvider | None = None,
        vectorstore: BaseVectorStoreProvider | None = None,
        batch_size: int = BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
        on_task_complete: Callable[[EmbeddingTask], Awaitable[None]] | None = None,
        on_task_fail: Callable[[EmbeddingTask], Awaitable[None]] | None = None,
    ):
        """
        初始化 Embedding Worker
        
        Args:
            state_path: 状态文件路径
            embedder: Embedder Provider（None 则自动检测）
            vectorstore: VectorStore Provider（None 则自动检测）
            batch_size: 批处理大小
            max_retries: 最大重试次数
            on_task_complete: 任务完成回调
            on_task_fail: 任务失败回调
        """
        self._state_path = Path(state_path)
        self._embedder = embedder
        self._vectorstore = vectorstore
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._on_task_complete = on_task_complete
        self._on_task_fail = on_task_fail
        
        self._state: EmbeddingState | None = None
        self._lock = asyncio.Lock()
        self._running = False
        self._task_handle: asyncio.Task | None = None
    
    @property
    def embedder(self) -> BaseEmbedderProvider:
        """获取 Embedder（懒加载）"""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder
    
    @property
    def vectorstore(self) -> BaseVectorStoreProvider:
        """获取 VectorStore（懒加载）"""
        if self._vectorstore is None:
            # 获取与 embedder 相同的维度
            self._vectorstore = get_vectorstore(
                dimensions=self.embedder.dimensions
            )
        return self._vectorstore
    
    def _now_iso(self) -> str:
        """获取当前时间 ISO 格式"""
        return datetime.now().isoformat()
    
    async def _load_state(self) -> EmbeddingState:
        """加载状态"""
        if self._state is not None:
            return self._state
        
        if self._state_path.exists():
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._state = EmbeddingState.from_dict(data)
        else:
            self._state = EmbeddingState(tasks={})
        
        return self._state
    
    async def _save_state(self) -> None:
        """保存状态"""
        if self._state is None:
            return
        
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, ensure_ascii=False, indent=2)
    
    async def add_task(
        self,
        id: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """
        添加 Embedding 任务
        
        Args:
            id: 任务 ID
            content: 待向量化的文本内容
            metadata: 额外元数据
        """
        async with self._lock:
            state = await self._load_state()
            
            if id in state.tasks:
                task = state.tasks[id]
                if task.status == EmbeddingStatus.COMPLETED:
                    logger.debug(f"Task {id} already completed, skipping")
                    return
                # 重置状态以便重新处理
                task.status = EmbeddingStatus.PENDING
                task.retry_count = 0
                task.error_message = None
            
            task = EmbeddingTask(
                id=id,
                content=content,
                metadata=metadata or {},
                status=EmbeddingStatus.PENDING,
                created_at=self._now_iso(),
                updated_at=self._now_iso(),
            )
            state.tasks[id] = task
            await self._save_state()
            
            logger.info(f"Added embedding task: {id}")
    
    async def add_tasks_batch(
        self,
        items: list[tuple[str, str, dict]],
    ) -> None:
        """
        批量添加 Embedding 任务
        
        Args:
            items: [(id, content, metadata), ...]
        """
        async with self._lock:
            state = await self._load_state()
            now = self._now_iso()
            
            for id, content, metadata in items:
                if id in state.tasks:
                    task = state.tasks[id]
                    if task.status == EmbeddingStatus.COMPLETED:
                        continue
                    task.status = EmbeddingStatus.PENDING
                    task.retry_count = 0
                    task.error_message = None
                
                task = EmbeddingTask(
                    id=id,
                    content=content,
                    metadata=metadata or {},
                    status=EmbeddingStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                )
                state.tasks[id] = task
            
            await self._save_state()
            logger.info(f"Added {len(items)} embedding tasks")
    
    async def get_task_status(self, id: str) -> EmbeddingStatus | None:
        """获取任务状态"""
        state = await self._load_state()
        task = state.tasks.get(id)
        return task.status if task else None
    
    async def _process_task(self, task: EmbeddingTask) -> bool:
        """
        处理单个任务
        
        Returns:
            True if success, False otherwise
        """
        try:
            # 生成向量
            task.status = EmbeddingStatus.GENERATING
            task.updated_at = self._now_iso()
            await self._save_state()
            
            vector = self.embedder.embed(task.content)
            
            # 写入 VectorStore
            entry = VectorEntry(
                id=task.id,
                vector=vector,
                metadata=task.metadata,
            )
            await self.vectorstore.upsert_async([entry])
            await self.vectorstore.persist_async()
            
            # 更新状态
            task.status = EmbeddingStatus.COMPLETED
            task.completed_at = self._now_iso()
            task.updated_at = self._now_iso()
            task.error_message = None
            await self._save_state()
            
            logger.info(f"Completed embedding task: {task.id}")
            
            if self._on_task_complete:
                await self._on_task_complete(task)
            
            return True
            
        except Exception as e:
            task.retry_count += 1
            task.error_message = str(e)
            task.updated_at = self._now_iso()
            
            if task.retry_count >= self._max_retries:
                task.status = EmbeddingStatus.PERMANENT_FAILURE
                logger.error(
                    f"Permanent failure for task {task.id} "
                    f"after {task.retry_count} retries: {e}"
                )
                if self._on_task_fail:
                    await self._on_task_fail(task)
            else:
                task.status = EmbeddingStatus.PENDING
                logger.warning(
                    f"Retry {task.retry_count}/{self._max_retries} "
                    f"for task {task.id}: {e}"
                )
            
            await self._save_state()
            return False
    
    async def _run_batch(self) -> int:
        """运行一批任务"""
        state = await self._load_state()
        
        # 收集 pending 任务
        pending_tasks = [
            task for task in state.tasks.values()
            if task.status == EmbeddingStatus.PENDING
        ]
        
        if not pending_tasks:
            return 0
        
        # 按创建时间排序
        pending_tasks.sort(
            key=lambda t: t.created_at or "",
        )
        
        # 取一批
        batch = pending_tasks[:self._batch_size]
        
        logger.info(f"Processing batch of {len(batch)} tasks")
        
        success_count = 0
        for task in batch:
            if await self._process_task(task):
                success_count += 1
            
            # 重试延迟
            if task.status == EmbeddingStatus.PENDING:
                delay = self.RETRY_DELAY_BASE ** task.retry_count
                await asyncio.sleep(delay)
        
        return success_count
    
    async def run_once(self) -> int:
        """运行一次处理循环"""
        return await self._run_batch()
    
    async def run_forever(
        self,
        interval: float = 1.0,
        max_idle_time: float | None = None,
    ) -> None:
        """
        持续运行 worker
        
        Args:
            interval: 检查间隔（秒）
            max_idle_time: 最大空闲时间（秒），超过后退出
        """
        self._running = True
        last_success_time = datetime.now()
        
        logger.info("Starting embedding worker")
        
        while self._running:
            try:
                # 检查是否有任务
                state = await self._load_state()
                pending_count = sum(
                    1 for t in state.tasks.values()
                    if t.status == EmbeddingStatus.PENDING
                )
                
                if pending_count > 0:
                    await self._run_batch()
                    last_success_time = datetime.now()
                else:
                    # 没有任务，等待
                    await asyncio.sleep(interval)
                
                # 检查空闲超时
                if max_idle_time is not None:
                    idle_time = (datetime.now() - last_success_time).total_seconds()
                    if idle_time > max_idle_time:
                        logger.info(
                            f"Idle timeout after {idle_time:.1f}s, stopping worker"
                        )
                        break
                
            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(interval)
        
        self._running = False
        logger.info("Embedding worker stopped")
    
    def stop(self) -> None:
        """停止 worker"""
        self._running = False
        if self._task_handle:
            self._task_handle.cancel()
            self._task_handle = None
    
    async def start_background(
        self,
        interval: float = 1.0,
        max_idle_time: float | None = 300.0,
    ) -> None:
        """
        启动后台 worker 任务
        
        Args:
            interval: 检查间隔
            max_idle_time: 最大空闲时间
        """
        if self._task_handle is not None:
            logger.warning("Worker already running")
            return
        
        self._task_handle = asyncio.create_task(
            self.run_forever(interval, max_idle_time)
        )
        logger.info("Started background embedding worker")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        state = asyncio.get_event_loop().run_until_complete(
            self._load_state()
        )
        
        stats = {
            "total": len(state.tasks),
            "pending": 0,
            "generating": 0,
            "completed": 0,
            "failed": 0,
            "permanent_failure": 0,
        }
        
        for task in state.tasks.values():
            if task.status == EmbeddingStatus.PENDING:
                stats["pending"] += 1
            elif task.status == EmbeddingStatus.GENERATING:
                stats["generating"] += 1
            elif task.status == EmbeddingStatus.COMPLETED:
                stats["completed"] += 1
            elif task.status == EmbeddingStatus.FAILED:
                stats["failed"] += 1
            elif task.status == EmbeddingStatus.PERMANENT_FAILURE:
                stats["permanent_failure"] += 1
        
        return stats


def create_worker(
    memory_dir: str | Path = "memory",
    **kwargs,
) -> EmbeddingWorker:
    """
    工厂函数：创建 Embedding Worker
    
    Args:
        memory_dir: 记忆存储目录
        **kwargs: 传递给 EmbeddingWorker 的参数
        
    Returns:
        EmbeddingWorker 实例
    """
    memory_dir = Path(memory_dir)
    state_path = memory_dir / ".embedding_state.json"
    vectorstore_path = memory_dir / "memory.usearch"
    
    # 确保目录存在
    memory_dir.mkdir(parents=True, exist_ok=True)
    
    return EmbeddingWorker(
        state_path=str(state_path),
        vectorstore=get_vectorstore(
            path=str(vectorstore_path),
            dimensions=kwargs.get("dimensions", 384),
        ),
        **kwargs,
    )
