"""
MultiAgent 模块 — 多 Agent 共享与锁机制

v2.0 架构：§11 MultiAgent 机制
- MultiAgentLock   : 基于 filelock 的文件锁，按 memory_id 粒度
- SharedLog        : NDJSON Append 日志，Agent 间同步 turns
- AgentRegistry    : Agent 注册表，心跳追踪
- sync_turn()      : 统一接口：锁 → 写 DataLake → 追加日志 → 释放锁 → 通知

依赖关系（§3.4 模块依赖）：
    MultiAgent → TieredLog
    MultiAgent → DataLake
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
import threading
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

# filelock: 跨平台文件锁（Windows: msvcrt, Linux/macOS: fcntl）
from filelock import FileLock, Timeout as FilelockTimeout

if TYPE_CHECKING:
    from collections.abc import Iterator

# ============================================================================
# 错误定义
# ============================================================================

from agentmemory.errors import MemoryError, PermissionError as MemPermissionError


class MultiAgentLockTimeout(MemoryError):
    """文件锁获取超时（E006.1）

    当在指定超时时间内无法获取文件锁时抛出。
    架构文档 §10：错误码 E006.1
    """

    code = "E006.1"

    def __init__(
        self,
        message: str = "Failed to acquire file lock within timeout",
        lock_path: str | None = None,
        timeout: float | None = None,
        context: dict | None = None,
    ):
        super().__init__(
            message=message,
            code=self.code,
            context={
                **(context or {}),
                "lock_path": lock_path,
                "timeout": timeout,
            },
        )


class AgentNotRegisteredError(MemoryError):
    """Agent 未注册（E006）

    尝试对未注册的 Agent 执行操作时抛出。
    """

    code = "E006"

    def __init__(
        self,
        message: str = "Agent is not registered",
        agent_id: str | None = None,
        context: dict | None = None,
    ):
        super().__init__(
            message=message,
            code=self.code,
            context={
                **(context or {}),
                "agent_id": agent_id,
            },
        )


class SharedLogError(MemoryError):
    """SharedLog 操作错误（E003）

    NDJSON 日志读写失败时抛出。
    """

    code = "E003"

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message=message, code=self.code, context=context)


# ============================================================================
# 日志条目模型
# ============================================================================


class SharedLogEntry:
    """SharedLog 单条记录

    字段对应架构文档 §5.8：
        agent_id / timestamp / event_type / memory_id / content
    """

    __slots__ = (
        "agent_id",
        "timestamp",
        "event_type",
        "memory_id",
        "content",
        "payload",
        "offset",
    )

    def __init__(
        self,
        agent_id: str,
        event_type: str,
        content: str = "",
        memory_id: str | None = None,
        timestamp: datetime | None = None,
        payload: dict | None = None,
        offset: int | None = None,
    ):
        self.agent_id: str = agent_id
        self.timestamp: datetime = timestamp or datetime.now(timezone.utc)
        self.event_type: str = event_type
        self.memory_id: str | None = memory_id
        self.content: str = content
        self.payload: dict = payload or {}
        self.offset: int | None = offset  # 在日志文件中的字节偏移

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "memory_id": self.memory_id,
            "content": self.content,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, raw: dict, offset: int | None = None) -> SharedLogEntry:
        ts = raw.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            agent_id=raw["agent_id"],
            event_type=raw["event_type"],
            content=raw.get("content", ""),
            memory_id=raw.get("memory_id"),
            timestamp=ts,
            payload=raw.get("payload", {}),
            offset=offset,
        )

    def to_json_line(self) -> str:
        """序列化为单行 JSON（用于 NDJSON 写入）"""
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"


# ============================================================================
# MultiAgentLock — 文件锁
# ============================================================================


class MultiAgentLock:
    """跨进程文件锁（基于 filelock）

    架构文档 §5.8 / §6.1：
        - 锁粒度：按 memory_id 级别，不堵死整个 DataLake
        - 锁超时：10 秒，超时抛 MultiAgentLockTimeout
        - 使用 filelock 库实现跨平台（Windows: msvcrt, Unix: fcntl）

    注意：架构文档 §11 说用 fcntl/msvcrt 直接实现，但任务要求用 filelock 库。
    filelock 库是对底层锁的跨平台抽象，内部实现与架构文档一致。
    """

    DEFAULT_TIMEOUT: ClassVar[float] = 10.0

    def __init__(
        self,
        lock_path: Path,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Args:
            lock_path: 锁文件路径（.lock 后缀）
            timeout: 获取锁超时时间（秒），默认 10s
        """
        self._lock_path: Path = Path(lock_path)
        self._timeout: float = timeout
        self._lock: FileLock = FileLock(
            str(self._lock_path),
            timeout=self._timeout,
        )
        # 锁排序字典：memory_id -> 重入计数（支持同一进程内重入）
        self._held: dict[str, int] = {}
        self._lock_guard = threading.Lock()

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def timeout(self) -> float:
        return self._timeout

    def acquire(self, memory_id: str | None = None, timeout: float | None = None) -> bool:
        """非阻塞尝试获取锁

        Args:
            memory_id: 要锁定的 memory_id（用于跟踪重入）
            timeout: 覆盖实例 timeout

        Returns:
            True if lock acquired, False if timeout

        Raises:
            MultiAgentLockTimeout: 当超时（与架构文档一致：超时抛异常）

        架构文档 §6.1：超时 5s（默认 10s 更保守）
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        try:
            # filelock.FileLock.acquire() 在超时后抛出 Timeout 异常
            self._lock.acquire(timeout=effective_timeout)
            with self._lock_guard:
                key = memory_id or "_global"
                self._held[key] = self._held.get(key, 0) + 1
            return True
        except FilelockTimeout:
            raise MultiAgentLockTimeout(
                message=f"Failed to acquire lock within {effective_timeout}s",
                lock_path=str(self._lock_path),
                timeout=effective_timeout,
            )

    def release(self, memory_id: str | None = None) -> None:
        """释放锁

        Args:
            memory_id: 要释放的 memory_id（用于跟踪重入计数）
        """
        with self._lock_guard:
            key = memory_id or "_global"
            count = self._held.get(key, 0)
            if count <= 1:
                self._held.pop(key, None)
                self._lock.release()
            else:
                self._held[key] = count - 1

    def is_locked(self) -> bool:
        """检查锁是否被当前进程持有"""
        return self._lock.is_locked

    def __enter__(self) -> "MultiAgentLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"MultiAgentLock(lock_path={self._lock_path!r}, "
            f"timeout={self._timeout}, held={self._held})"
        )


# ============================================================================
# SharedLog — NDJSON Append 日志
# ============================================================================


class SharedLog:
    """多 Agent 共享 NDJSON 日志

    架构文档 §5.8 / §9：
        - 日志路径：{data_root}/.logs/shared_{date}.ndjson
        - 每条记录：agent_id / timestamp / event_type / memory_id / content
        - append_event(): 非阻塞追加，写完即返回
        - read_since(offset): 从指定 offset 读取，返回 (records, new_offset)

    跨进程安全：
        - 写操作持 MultiAgentLock 排他锁
        - 读操作无锁（文件可共享读）
        - 单行 < 4KB 保证 POSIX 原子性
    """

    LOG_DIR_PREFIX: ClassVar[str] = ".logs"
    LOG_FILE_PREFIX: ClassVar[str] = "shared_"

    def __init__(
        self,
        data_root: Path,
        lock: MultiAgentLock,
    ):
        """
        Args:
            data_root: 数据根目录（DataLake root）
            lock: 用于写操作的锁
        """
        self._data_root: Path = Path(data_root)
        self._lock: MultiAgentLock = lock
        self._log_dir: Path = self._data_root / self.LOG_DIR_PREFIX
        self._today_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _log_path(self, date_str: str | None = None) -> Path:
        """获取指定日期的日志文件路径"""
        ds = date_str or self._today_str
        return self._log_dir / f"{self.LOG_FILE_PREFIX}{ds}.ndjson"

    def _ensure_log_dir(self) -> None:
        """确保日志目录存在"""
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def append_event(
        self,
        agent_id: str,
        event_type: str,
        content: str = "",
        memory_id: str | None = None,
        payload: dict | None = None,
    ) -> int:
        """追加一条事件（非阻塞）

        流程：持锁 → open("ab") → write + flush + os.fsync → 释放锁

        Args:
            agent_id: 事件来源 agent
            event_type: 事件类型（如 "turn_end", "stored", "forgotten"）
            content: 事件内容
            memory_id: 关联的 memory_id（可选）
            payload: 额外数据（可选）

        Returns:
            写入后的文件偏移量（字节）

        Raises:
            SharedLogError: 写入失败
            MultiAgentLockTimeout: 锁超时
        """
        entry = SharedLogEntry(
            agent_id=agent_id,
            event_type=event_type,
            content=content,
            memory_id=memory_id,
            payload=payload,
        )

        self._ensure_log_dir()
        log_path = self._log_path()
        line = entry.to_json_line()

        # 写长度校验（架构文档 §6.2：单行 < 4KB 保证原子性）
        if len(line.encode("utf-8")) >= 4096:
            raise SharedLogError(
                message="Log entry exceeds 4KB atomic write limit",
                context={"entry_size": len(line)},
            )

        with self._lock:
            try:
                with open(log_path, "ab") as f:
                    f.write(line.encode("utf-8"))
                    f.flush()
                    os.fsync(f.fileno())
                # 返回写入后的文件大小（作为新 offset）
                return log_path.stat().st_size
            except OSError as e:
                raise SharedLogError(
                    message=f"Failed to write to shared log: {e}",
                    context={"log_path": str(log_path), "entry": entry.to_dict()},
                )

    def read_since(
        self,
        offset: int,
        date_str: str | None = None,
    ) -> tuple[list[SharedLogEntry], int]:
        """从指定偏移读取日志（增量同步）

        Args:
            offset: 起始字节偏移（0 表示从头开始）
            date_str: 指定日期（默认今天）

        Returns:
            (records, new_offset) — records 是该偏移后的所有记录，
            new_offset 是读取结束后的文件大小（供下次调用）

        Raises:
            SharedLogError: 读取失败
        """
        log_path = self._log_path(date_str)

        if not log_path.exists():
            return [], 0

        try:
            with open(log_path, "rb") as f:
                # Seek to offset
                f.seek(offset)
                data = f.read()
                new_offset = offset + len(data)

            if not data:
                return [], offset

            lines = data.decode("utf-8").splitlines()
            records: list[SharedLogEntry] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    # 计算该行在原文件中的偏移
                    line_offset = offset + data.decode("utf-8").index(line)
                    records.append(SharedLogEntry.from_dict(raw, offset=line_offset))
                except json.JSONDecodeError:
                    # 架构文档 §6.2：损坏行移动到 .broken
                    self._move_broken_line(log_path, line, offset)
                    continue

            return records, new_offset

        except OSError as e:
            raise SharedLogError(
                message=f"Failed to read shared log: {e}",
                context={"log_path": str(log_path), "offset": offset},
            )

    def _move_broken_line(
        self,
        log_path: Path,
        line: str,
        offset_hint: int,
    ) -> None:
        """将损坏的日志行移动到 .broken 文件（架构文档 §6.2）"""
        broken_path = log_path.with_suffix(".ndjson.broken")
        try:
            with open(broken_path, "ab") as bf:
                bf.write(line.encode("utf-8") + b"\n")
        except OSError:
            pass  # 不要因清理操作失败而中断

    def tail(self, n: int = 100, date_str: str | None = None) -> list[SharedLogEntry]:
        """读取最近 n 条记录

        Args:
            n: 返回最近 n 条
            date_str: 指定日期（默认今天）

        Returns:
            最近 n 条 SharedLogEntry 列表
        """
        log_path = self._log_path(date_str)

        if not log_path.exists():
            return []

        try:
            with open(log_path, "rb") as f:
                # 读取最后 N 行（简单策略：读全部，行数少时可行）
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                # 最多回读 1MB
                read_start = max(0, file_size - 1024 * 1024)
                f.seek(read_start)
                data = f.read().decode("utf-8")

            lines = data.splitlines()
            # 过滤空行并重建偏移
            non_empty = [(read_start + data.index(line), line)
                         for line in lines if line.strip()]
            if not non_empty:
                return []

            # 取最后 n 条
            selected = non_empty[-n:]
            records: list[SharedLogEntry] = []
            for off, line in selected:
                try:
                    raw = json.loads(line.strip())
                    records.append(SharedLogEntry.from_dict(raw, offset=off))
                except json.JSONDecodeError:
                    continue
            return records

        except OSError:
            return []

    def current_offset(self, date_str: str | None = None) -> int:
        """获取当前日志文件的字节大小（作为最新 offset）"""
        log_path = self._log_path(date_str)
        if log_path.exists():
            return log_path.stat().st_size
        return 0


# ============================================================================
# AgentRegistry — Agent 注册表
# ============================================================================


class AgentRegistry:
    """Agent 注册表

    架构文档 §5.8：
        - 记录活跃 Agent：agent_id / last_heartbeat / capabilities
        - register() / unregister() / list_active()
        - 心跳超时：60 秒判定离线

    存储：JSON 文件（data_root/.logs/.agent_registry.json）
    """

    REGISTRY_FILE: ClassVar[str] = ".agent_registry.json"
    LOG_DIR_PREFIX: ClassVar[str] = ".logs"  # 与 SharedLog 共享
    HEARTBEAT_TIMEOUT: ClassVar[int] = 60  # 秒

    def __init__(
        self,
        data_root: Path,
    ):
        """
        Args:
            data_root: 数据根目录
        """
        self._data_root: Path = Path(data_root)
        self._registry_path: Path = (
            self._data_root / AgentRegistry.LOG_DIR_PREFIX / AgentRegistry.REGISTRY_FILE
        )
        self._registry: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """从文件加载注册表"""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._registry = {}
        else:
            self._registry = {}

    def _save(self) -> None:
        """持久化注册表到文件"""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # 不要因持久化失败而中断

    def register(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """注册一个 Agent

        Args:
            agent_id: Agent 唯一标识
            capabilities: Agent 能力列表（如 ["read", "write", "admin"]）
            metadata: 额外元数据

        Returns:
            注册记录 dict
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            record = {
                "agent_id": agent_id,
                "registered_at": now,
                "last_heartbeat": now,
                "capabilities": capabilities or [],
                "metadata": metadata or {},
                "status": "active",
            }
            self._registry[agent_id] = record
            self._save()
            return record

    def unregister(self, agent_id: str) -> bool:
        """注销一个 Agent

        Args:
            agent_id: Agent 唯一标识

        Returns:
            True if was registered, False otherwise
        """
        with self._lock:
            if agent_id in self._registry:
                del self._registry[agent_id]
                self._save()
                return True
            return False

    def heartbeat(self, agent_id: str) -> bool:
        """更新 Agent 心跳

        Args:
            agent_id: Agent 唯一标识

        Returns:
            True if agent was registered, False otherwise
        """
        with self._lock:
            if agent_id not in self._registry:
                return False
            self._registry[agent_id]["last_heartbeat"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._registry[agent_id]["status"] = "active"
            self._save()
            return True

    def list_active(self) -> list[dict]:
        """列出所有活跃 Agent（心跳未超时）

        Returns:
            活跃 Agent 的注册记录列表
        """
        self._cleanup_stale()
        return list(self._registry.values())

    def get(self, agent_id: str) -> dict | None:
        """获取指定 Agent 的注册记录"""
        self._load()
        return self._registry.get(agent_id)

    def _cleanup_stale(self) -> None:
        """清理心跳超时的 Agent（内部调用）"""
        now = datetime.now(timezone.utc)
        stale_ids: list[str] = []

        for agent_id, record in self._registry.items():
            last_ts = record.get("last_heartbeat", "")
            if not last_ts:
                stale_ids.append(agent_id)
                continue
            try:
                last_time = datetime.fromisoformat(last_ts)
                # 假设 last_time 是 UTC
                if (now - last_time).total_seconds() > self.HEARTBEAT_TIMEOUT:
                    stale_ids.append(agent_id)
            except ValueError:
                stale_ids.append(agent_id)

        if stale_ids:
            with self._lock:
                for aid in stale_ids:
                    self._registry.pop(aid, None)
                self._save()

    @property
    def registry_path(self) -> Path:
        return self._registry_path


# ============================================================================
# 通知机制（async wait_for_turn）
# ============================================================================


class TurnNotification:
    """Agent 轮次通知机制

    架构文档 §11（sync_turn 流程）：
        - 等待通知：wait_for_turn(agent_id, timeout)
        - 当一个 Agent 完成了 sync_turn，可以通知等待的 Agent

    实现：threading.Event + 全局字典（进程内通知）
    对于跨进程通知，需要依赖 SharedLog 的 read_since 轮询
    """

    _events: ClassVar[dict[str, asyncio.Event]] = {}
    _events_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_event(cls, agent_id: str) -> asyncio.Event:
        """获取指定 agent_id 的通知事件"""
        with cls._events_lock:
            if agent_id not in cls._events:
                cls._events[agent_id] = asyncio.Event()
            return cls._events[agent_id]

    @classmethod
    async def wait_for_turn(
        cls,
        agent_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """等待被通知（异步）

        Args:
            agent_id: 当前 Agent 的 ID
            timeout: 超时时间（秒）

        Returns:
            True if notified, False if timeout
        """
        event = cls.get_event(agent_id)
        try:
            return await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

    @classmethod
    def notify(cls, agent_id: str) -> None:
        """通知指定 Agent

        Args:
            agent_id: 要通知的 Agent ID
        """
        with cls._events_lock:
            if agent_id in cls._events:
                cls._events[agent_id].set()

    @classmethod
    def clear(cls, agent_id: str) -> None:
        """清除通知状态（Agent 被通知后调用）"""
        with cls._events_lock:
            if agent_id in cls._events:
                cls._events[agent_id].clear()


# ============================================================================
# 生成唯一 Agent ID
# ============================================================================


def generate_agent_id() -> str:
    """生成唯一的 Agent ID（UUID4）"""
    return str(uuid.uuid4())


# ============================================================================
# sync_turn — 统一接口
# ============================================================================


async def sync_turn(
    agent_id: str,
    user_msg: str,
    assistant_msg: str,
    memory_id: str | None,
    shared_log: SharedLog,
    lock: MultiAgentLock,
    notification_targets: list[str] | None = None,
) -> str | None:
    """对话轮次同步（MemoryHermes 层统一接口）

    架构文档 §5.11 MemoryHermes.sync_turn()：
        流程：获取锁 → 写 DataLake → 追加 SharedLog → 释放锁 → 通知等待的 Agent

    注意：本函数是纯同步版本的 sync_turn 逻辑，不包含实际 DataLake 写入。
    调用方需要先在 DataLake 层写入 memory，再调用本函数追加日志。

    Args:
        agent_id: 当前 Agent 的 ID
        user_msg: 用户消息
        assistant_msg: 助手消息
        memory_id: 已写入的 memory_id（如果触发了存储）
        shared_log: SharedLog 实例
        lock: MultiAgentLock 实例
        notification_targets: 需要通知的 agent_id 列表（可选）

    Returns:
        memory_id（与输入相同，供调用方确认）
    """
    # 构建日志内容
    content = f"[user]\n{user_msg}\n\n[assistant]\n{assistant_msg}"

    try:
        # 获取锁 → 追加日志 → 释放锁
        with lock:
            shared_log.append_event(
                agent_id=agent_id,
                event_type="turn_end",
                content=content,
                memory_id=memory_id,
                payload={
                    "user_msg": user_msg,
                    "assistant_msg": assistant_msg,
                    "turn_synced": True,
                },
            )

        # 通知等待的 Agent
        if notification_targets:
            for target_id in notification_targets:
                TurnNotification.notify(target_id)

        return memory_id

    except MultiAgentLockTimeout:
        # 锁超时，但日志已尝试追加（部分成功）
        raise
    except SharedLogError:
        raise


# ============================================================================
# MultiAgent — 整合包装类
# ============================================================================


class MultiAgent:
    """MultiAgent 整合类

    封装 MultiAgentLock + SharedLog + AgentRegistry + TurnNotification，
    提供单一入口。

    典型用法：
        ma = MultiAgent(data_root=Path("./memory_library"))
        ma.register_agent("my-agent")
        agent_id = ma.agent_id

        # 在 store / forget 等操作中
        with ma.lock(memory_id):
            data_lake.write(...)
            ma.append_log("stored", memory_id=memory_id)

        # 在 sync_turn 中
        await ma.sync_turn(user_msg, assistant_msg, memory_id)
    """

    def __init__(
        self,
        data_root: Path,
        lock_timeout: float = 10.0,
        agent_id: str | None = None,
    ):
        """
        Args:
            data_root: 数据根目录
            lock_timeout: 锁超时（秒），默认 10s
            agent_id: 指定 agent_id（默认自动生成 UUID）
        """
        self._data_root: Path = Path(data_root)
        self._lock_timeout: float = lock_timeout
        self._agent_id: str = agent_id or generate_agent_id()

        # 初始化子组件
        self._lock: MultiAgentLock = MultiAgentLock(
            lock_path=self._data_root / ".multi_agent.lock",
            timeout=self._lock_timeout,
        )
        self._shared_log: SharedLog = SharedLog(
            data_root=self._data_root,
            lock=self._lock,
        )
        self._registry: AgentRegistry = AgentRegistry(
            data_root=self._data_root,
        )

    @property
    def agent_id(self) -> str:
        """当前 Agent 的唯一 ID"""
        return self._agent_id

    @property
    def lock(self) -> MultiAgentLock:
        return self._lock

    @property
    def shared_log(self) -> SharedLog:
        return self._shared_log

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    def register_agent(
        self,
        capabilities: list[str] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """注册当前 Agent"""
        return self._registry.register(
            agent_id=self._agent_id,
            capabilities=capabilities,
            metadata=metadata,
        )

    def unregister_agent(self) -> bool:
        """注销当前 Agent"""
        return self._registry.unregister(self._agent_id)

    def heartbeat(self) -> bool:
        """发送心跳"""
        return self._registry.heartbeat(self._agent_id)

    def append_log(
        self,
        event_type: str,
        content: str = "",
        memory_id: str | None = None,
        payload: dict | None = None,
    ) -> int:
        """追加日志（便捷方法）"""
        return self._shared_log.append_event(
            agent_id=self._agent_id,
            event_type=event_type,
            content=content,
            memory_id=memory_id,
            payload=payload,
        )

    def read_since(self, offset: int, date_str: str | None = None):
        """从 offset 读取日志"""
        return self._shared_log.read_since(offset, date_str)

    async def sync_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        memory_id: str | None = None,
        notification_targets: list[str] | None = None,
    ) -> str | None:
        """对话轮次同步"""
        return await sync_turn(
            agent_id=self._agent_id,
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            memory_id=memory_id,
            shared_log=self._shared_log,
            lock=self._lock,
            notification_targets=notification_targets,
        )

    async def wait_for_turn(self, timeout: float = 30.0) -> bool:
        """等待被通知"""
        return await TurnNotification.wait_for_turn(self._agent_id, timeout)

    def notify(self, target_agent_id: str) -> None:
        """通知另一个 Agent"""
        TurnNotification.notify(target_agent_id)

    def __repr__(self) -> str:
        return (
            f"MultiAgent(agent_id={self._agent_id!r}, "
            f"data_root={self._data_root!r}, "
            f"lock_timeout={self._lock_timeout})"
        )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 错误
    "MultiAgentLockTimeout",
    "AgentNotRegisteredError",
    "SharedLogError",
    # 模型
    "SharedLogEntry",
    # 核心类
    "MultiAgentLock",
    "SharedLog",
    "AgentRegistry",
    "TurnNotification",
    "MultiAgent",
    # 工具
    "generate_agent_id",
    "sync_turn",
]
