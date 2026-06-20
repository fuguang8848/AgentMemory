"""Skill 版本化 + 热更新模块

导出:
    SkillVersion       版本化 skill 定义
    SkillRegistry      维护版本历史的注册表
    SkillHotReload     热更新监控器（watchdog）
    SkillDiff         版本差异对比工具

功能:
    - 每个 skill 维护多版本历史
    - 支持回滚到任意历史版本
    - 热更新：监控 skill 文件变化，自动重新加载
    - 版本 diff 对比两个版本的 prompt 差异
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "SkillVersion",
    "SkillRegistry",
    "SkillHotReload",
    "SkillDiff",
    "skill_diff",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SkillVersion:
    """单个版本的 skill 定义"""

    version_id: str           # SHA256(version + prompt) 前8位
    version_num: int          # 版本序号（从1递增）
    prompt_template: str      # skill prompt 模板
    metadata: dict[str, Any]  # 描述、标签、参数等
    created_at: str           # ISO 时间戳
    created_by: str = "system"

    @classmethod
    def create(cls, version_num: int, prompt_template: str, metadata: dict, created_by: str = "system") -> SkillVersion:
        """工厂方法：创建新版本，自动计算 version_id"""
        raw = f"{version_num}:{prompt_template}"
        version_id = hashlib.sha256(raw.encode()).hexdigest()[:8]
        return cls(
            version_id=version_id,
            version_num=version_num,
            prompt_template=prompt_template,
            metadata=metadata,
            created_at=datetime.utcnow().isoformat() + "Z",
            created_by=created_by,
        )


@dataclass
class SkillHistory:
    """单个 skill 的完整版本历史"""

    skill_name: str
    versions: list[SkillVersion] = field(default_factory=list)
    current_version_id: str | None = None

    def add_version(self, version: SkillVersion) -> None:
        self.versions.append(version)
        self.current_version_id = version.version_id

    def get_version(self, version_id: str) -> SkillVersion | None:
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None

    def get_current(self) -> SkillVersion | None:
        if self.current_version_id:
            return self.get_version(self.current_version_id)
        return self.versions[-1] if self.versions else None

    def rollback(self, version_id: str) -> bool:
        """回滚到指定版本"""
        target = self.get_version(version_id)
        if target is None:
            return False
        self.current_version_id = version_id
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────


class SkillRegistry:
    """维护所有 skill 的版本历史注册表"""

    def __init__(self, storage_path: str | None = None):
        self._skills: dict[str, SkillHistory] = {}
        self._storage_path = storage_path

    def register_skill(
        self,
        skill_name: str,
        prompt_template: str,
        metadata: dict | None = None,
        created_by: str = "system",
    ) -> SkillVersion:
        """注册或更新一个 skill（创建新版本）"""
        history = self._skills.setdefault(skill_name, SkillHistory(skill_name=skill_name))
        version_num = len(history.versions) + 1
        version = SkillVersion.create(
            version_num=version_num,
            prompt_template=prompt_template,
            metadata=metadata or {},
            created_by=created_by,
        )
        history.add_version(version)
        return version

    def get_current(self, skill_name: str) -> SkillVersion | None:
        """获取 skill 当前活跃版本"""
        return self._skills.get(skill_name, SkillHistory(skill_name="")).get_current()

    def list_versions(self, skill_name: str) -> list[SkillVersion]:
        """列出 skill 所有历史版本"""
        h = self._skills.get(skill_name)
        return list(h.versions) if h else []

    def rollback(self, skill_name: str, version_id: str) -> bool:
        """回滚 skill 到指定版本"""
        h = self._skills.get(skill_name)
        if h is None:
            return False
        return h.rollback(version_id)

    def save(self, path: str | None = None) -> None:
        """持久化注册表到 JSON 文件"""
        target = path or self._storage_path
        if not target:
            return
        data = {
            name: {
                "skill_name": h.skill_name,
                "current_version_id": h.current_version_id,
                "versions": [
                    {**asdict(v), "version_num": v.version_num}
                    for v in h.versions
                ],
            }
            for name, h in self._skills.items()
        }
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | None = None) -> None:
        """从 JSON 文件加载注册表"""
        target = path or self._storage_path
        if not target or not Path(target).exists():
            return
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        self._skills.clear()
        for name, item in data.items():
            h = SkillHistory(skill_name=item["skill_name"], current_version_id=item.get("current_version_id"))
            for v_data in item.get("versions", []):
                h.versions.append(SkillVersion(**v_data))
            self._skills[name] = h


# ──────────────────────────────────────────────────────────────────────────────
# SkillDiff
# ──────────────────────────────────────────────────────────────────────────────


def skill_diff(v1: SkillVersion, v2: SkillVersion) -> dict[str, Any]:
    """对比两个版本的 prompt 差异，返回结构化 diff"""
    p1, p2 = v1.prompt_template, v2.prompt_template
    lines1, lines2 = p1.splitlines(), p2.splitlines()

    added   = [l for l in lines2 if l not in lines1]
    removed = [l for l in lines1 if l not in lines2]
    unchanged = [l for l in lines1 if l in lines2]

    return {
        "v1": v1.version_id,
        "v2": v2.version_id,
        "v1_num": v1.version_num,
        "v2_num": v2.version_num,
        "added_lines":   added,
        "removed_lines": removed,
        "unchanged_count": len(unchanged),
        "change_count": len(added) + len(removed),
    }


class SkillDiff:
    """SkillDiff 包装器，提供友好的对比报告"""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def compare(self, skill_name: str, v1_id: str, v2_id: str) -> str:
        """生成人类可读的 diff 报告"""
        h = self.registry._skills.get(skill_name)
        if not h:
            return f"Skill '{skill_name}' not found"

        v1 = h.get_version(v1_id)
        v2 = h.get_version(v2_id)
        if not v1 or not v2:
            return "Version not found"

        diff = skill_diff(v1, v2)
        lines = [
            f"SkillDiff: {skill_name}",
            f"  v{diff['v1_num']} ({diff['v1']}) → v{diff['v2_num']} ({diff['v2']})",
            f"  unchanged: {diff['unchanged_count']} lines",
            f"  changed:   {diff['change_count']} lines",
            "",
            "  Removed:",
        ]
        for l in diff["removed_lines"]:
            lines.append(f"    - {l}")
        lines.append("  Added:")
        for l in diff["added_lines"]:
            lines.append(f"    + {l}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# HotReload — watchdog 监控
# ──────────────────────────────────────────────────────────────────────────────

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    _WATCHDOG_AVAILABLE = True
except Exception:
    _WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEvent = object


class _SkillFileHandler(FileSystemEventHandler if _WATCHDOG_AVAILABLE else object):
    """监控 skill 目录文件变化，触发回调"""

    def __init__(self, callback: callable):
        self.callback = callback

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and event.src_path.endswith((".md", ".yaml", ".yml", ".json")):
            self.callback(event.src_path)


class SkillHotReload:
    """热更新监控器：监控 skill 文件变化，自动重新加载"""

    def __init__(self, registry: SkillRegistry, watch_dirs: list[str] | None = None):
        self.registry = registry
        self._watch_dirs = watch_dirs or []
        self._observer: Observer | None = None
        self._reload_callbacks: list[callable] = []

    def add_reload_callback(self, cb: callable) -> None:
        """注册 reload 回调（接受 skill_name, version_id 参数）"""
        self._reload_callbacks.append(cb)

    def start(self) -> None:
        """启动文件监控"""
        if not _WATCHDOG_AVAILABLE:
            import logging
            logging.warning("watchdog not available — SkillHotReload disabled")
            return

        self._observer = Observer()
        handler = _SkillFileHandler(self._on_file_changed)
        for d in self._watch_dirs:
            p = Path(d)
            if p.exists():
                self._observer.schedule(handler, str(p), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        """停止文件监控"""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

    def _on_file_changed(self, filepath: str) -> None:
        """文件变化时触发：解析 skill 名，通知回调"""
        import logging
        logger = logging.getLogger(__name__)

        skill_name = Path(filepath).stem
        current = self.registry.get_current(skill_name)

        logger.info(f"Skill file changed: {filepath}, reloading {skill_name}")
        for cb in self._reload_callbacks:
            try:
                cb(skill_name, current.version_id if current else None)
            except Exception as exc:
                logger.error(f"Reload callback error: {exc}")
