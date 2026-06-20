"""
睡眠调度器 — SleepScheduler
============================

参考：openclaw-auto-dream 定时任务（默认每天凌晨4点）

支持：
1. 固定时间调度（cron 表达式）
2. 间隔调度（每N小时触发一次）
3. 事件触发（记忆量超过阈值时触发）
4. 手动触发
"""

import threading
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Callable
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SleepScheduler:
    """
    梦境睡眠调度器

    使用示例：
        dream = DreamNet()
        scheduler = SleepScheduler(dream, default_hour=4, default_minute=0)  # 每天4:00 AM
        scheduler.start()

        # 或者间隔模式（每6小时）
        scheduler = SleepScheduler(dream, interval_hours=6)
        scheduler.start()
    """

    def __init__(
        self,
        dream_net,                    # DreamNet 实例
        interval_hours: Optional[int] = None,
        default_hour: int = 4,
        default_minute: int = 0,
        enabled: bool = True,
    ):
        self.dream = dream_net
        self.interval_hours = interval_hours
        self.default_hour = default_hour
        self.default_minute = default_minute
        self.enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_run: Optional[str] = None
        self._last_result: Optional[str] = None
        self._state_file = Path("~/.openclaw/workspace/memory/.dream_scheduler_state.json").expanduser()

    def start(self):
        """启动调度器（后台线程）"""
        if self._thread and self._thread.is_alive():
            logger.warning("Scheduler already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="DreamScheduler")
        self._thread.start()
        logger.info(f"Dream scheduler started (interval={self.interval_hours}h or daily@{self.default_hour:02d}:{self.default_minute:02d})")

    def stop(self):
        """停止调度器"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Dream scheduler stopped")

    def _run_loop(self):
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            next_run = self._next_scheduled_time(now)

            wait_seconds = (next_run - now).total_seconds()
            if wait_seconds > 0:
                logger.debug(f"Next dream cycle at {next_run.isoformat()}, waiting {wait_seconds:.0f}s")
                if self._stop_event.wait(timeout=min(wait_seconds, 3600)):
                    break  # 被 stop() 唤醒

            if self._stop_event.is_set():
                break

            # 执行梦境周期
            try:
                logger.info("Triggering scheduled dream cycle")
                result = self.dream.run_dream_cycle()
                self._last_run = datetime.now(timezone.utc).isoformat()
                self._last_result = "success" if result.success else f"failed: {result.error}"
                self._save_state()
            except Exception as e:
                self._last_result = f"error: {e}"
                logger.exception("Dream cycle failed")

    def _next_scheduled_time(self, now: datetime) -> datetime:
        """计算下次调度时间"""
        if self.interval_hours:
            # 间隔模式：下次 = 现在 + interval
            return now
        else:
            # 固定时间模式
            next_time = now.replace(hour=self.default_hour, minute=self.default_minute, second=0, microsecond=0)
            if next_time <= now:
                next_time = next_time.replace(day=next_time.day + 1)
            return next_time

    def _save_state(self):
        """持久化调度器状态"""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump({
                "last_run": self._last_run,
                "last_result": self._last_result,
                "enabled": self.enabled,
            }, f, indent=2)

    def trigger_now(self) -> dict:
        """立即触发一次梦境周期"""
        logger.info("Manually triggering dream cycle")
        result = self.dream.run_dream_cycle()
        self._last_run = datetime.now(timezone.utc).isoformat()
        self._last_result = "success" if result.success else f"failed: {result.error}"
        self._save_state()
        return {
            "success": result.success,
            "last_run": self._last_run,
            "entries_processed": result.entries_processed,
            "entries_archived": result.entries_archived,
            "insights": result.insights,
        }
