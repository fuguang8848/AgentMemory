"""
L4 File Persist Layer - 文件系统持久化层
对应 Hermes 的 BuiltinMemory（MEMORY.md + 每日日记）
"""

import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class MemoryCategory(str, Enum):
    GENERAL = "general"
    PROJECT = "project"
    DECISION = "decision"
    PREFERENCE = "preference"
    LEARNING = "learning"


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class DiaryEntry:
    """日记条目"""
    time: str
    category: str
    content: str
    importance: float = 1.0

    def to_markdown(self) -> str:
        return f"- [{self.time}] [{self.category}] {self.content}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FactEntry:
    """事实记忆条目"""
    fact: str
    category: str
    importance: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IndexEntry:
    """索引条目"""
    date: str
    path: str
    entries_count: int

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# DailyMemory 类
# =============================================================================

class DailyMemory:
    """
    每日日记 - memory/YYYY-MM-DD.md
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.memory_dir = self.workspace_path / "memory"
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """确保目录存在"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _get_date_path(self, date: Optional[str] = None) -> Path:
        """获取指定日期的文件路径"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        # 验证日期格式
        if not re.match(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError(f"Invalid date format: {date}. Expected YYYY-MM-DD")
        return self.memory_dir / f"{date}.md"

    def _parse_time(self) -> str:
        """获取当前时间字符串"""
        return datetime.now().strftime("%H:%M")

    def _read_raw(self, date: str) -> str:
        """读取指定日期的原始内容（支持多种编码）"""
        path = self._get_date_path(date)
        if path.exists():
            # 尝试多种编码
            for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
                try:
                    return path.read_text(encoding=enc)
                except (UnicodeDecodeError, LookupError):
                    continue
            # 最后尝试二进制读取并替换错误字节
            try:
                raw = path.read_bytes()
                return raw.decode("utf-8", errors="replace")
            except Exception:
                pass
        return ""


    def _ensure_header(self, content: str, date: str) -> str:
        """确保文件有正确的标题头"""
        header = f"# {date} 日记\n\n"
        if content.strip().startswith("#"):
            return content
        return header + content

    def append(self, entry: str, category: str = "general") -> bool:
        """
        追加日记条目

        Args:
            entry: 日记内容
            category: 分类 (general/project/decision/preference/learning)

        Returns:
            bool: 成功返回 True
        """
        try:
            date = datetime.now().strftime("%Y-%m-%d")
            path = self._get_date_path(date)

            # 读取现有内容
            content = ""
            if path.exists():
                content = path.read_text(encoding="utf-8")

            # 确保有标题头
            content = self._ensure_header(content, date)

            # 构建分类区域
            category_header = f"## {category}\n"
            if f"## {category}" not in content:
                # 找到最后一个 ## 标题的位置，在其后面插入新分类
                last_header_pos = content.rfind("## ")
                if last_header_pos != -1:
                    # 找到该标题后的第一个换行
                    next_newline = content.find("\n", last_header_pos)
                    if next_newline != -1:
                        content = content[:next_newline + 1] + "\n" + category_header + content[next_newline + 1:]
                    else:
                        content += "\n" + category_header
                else:
                    content += "\n" + category_header

            # 添加条目
            time_str = self._parse_time()
            new_entry = f"- [{time_str}] {entry}\n"

            # 找到对应分类区域并追加
            lines = content.split("\n")
            in_category = False
            insert_pos = len(content)

            for i, line in enumerate(lines):
                if line.strip() == f"## {category}":
                    in_category = True
                elif in_category and line.strip().startswith("## "):
                    # 进入下一个分类，前一行结束
                    insert_pos = content.find("\n" + line)
                    break
                elif in_category:
                    insert_pos = content.find("\n" + line) + len("\n" + line) if i < len(lines) - 1 else len(content)

            if in_category:
                # 在分类区域内追加
                before = content[:insert_pos]
                after = content[insert_pos:]
                content = before + new_entry + after
            else:
                content += new_entry

            # 写入文件
            path.write_text(content, encoding="utf-8")

            # 更新索引
            self._update_index(date)

            return True

        except Exception as e:
            print(f"Error appending diary entry: {e}")
            return False

    def read(self, date: Optional[str] = None) -> str:
        """
        读取指定日期的日记

        Args:
            date: 日期字符串 (默认今天)

        Returns:
            str: 日记内容
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            path = self._get_date_path(date)

            if path.exists():
                return self._read_raw(date)
            return f"# {date} 日记\n\n_暂无记录_"

        except Exception as e:
            print(f"Error reading diary: {e}")
            return ""

    def list_entries(self, date: Optional[str] = None) -> list[dict]:
        """
        列出指定日期的所有条目

        Args:
            date: 日期字符串 (默认今天)

        Returns:
            list[dict]: 条目列表
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")

            content = self.read(date)
            entries = []

            # 解析 markdown 列表
            current_category = "general"
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("## "):
                    current_category = line.replace("## ", "").strip()
                elif line.startswith("- ["):
                    # 匹配格式: - [HH:MM] [category] content 或 - [HH:MM] content
                    match = re.match(r"- \[(\d{2}:\d{2})\](?: \[([^\]]+)\])? (.+)", line)
                    if match:
                        time_str, cat, content_text = match.groups()
                        entries.append({
                            "time": time_str,
                            "category": cat if cat else current_category,
                            "content": content_text,
                            "date": date
                        })

            return entries

        except Exception as e:
            print(f"Error listing entries: {e}")
            return []

    def search(self, keyword: str) -> list[dict]:
        """
        搜索所有日记中的关键词

        Args:
            keyword: 搜索关键词

        Returns:
            list[dict]: 匹配的条目列表
        """
        results = []
        try:
            # 遍历所有日记文件
            if not self.memory_dir.exists():
                return results

            for md_file in sorted(self.memory_dir.glob("*.md")):
                date = md_file.stem  # 文件名作为日期
                # 尝试多种编码
                content = ""
                for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
                    try:
                        content = md_file.read_text(encoding=enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                if not content:
                    continue

                # 简单关键词匹配（不区分大小写）
                if keyword.lower() in content.lower():
                    # 解析匹配的行
                    lines = content.split("\n")
                    current_category = "general"

                    for line in lines:
                        if line.strip().startswith("## "):
                            current_category = line.strip().replace("## ", "")
                        elif line.strip().startswith("- [") and keyword.lower() in line.lower():
                            match = re.match(r"- \[(\d{2}:\d{2})\](?: \[([^\]]+)\])? (.+)", line.strip())
                            if match:
                                time_str, cat, content_text = match.groups()
                                results.append({
                                    "time": time_str,
                                    "category": cat if cat else current_category,
                                    "content": content_text,
                                    "date": date
                                })

            return results

        except Exception as e:
            print(f"Error searching diary: {e}")
            return []

    def _update_index(self, date: str) -> None:
        """更新索引文件"""
        try:
            index_path = self.memory_dir / "index.json"
            index_data = {"entries": []}

            if index_path.exists():
                raw = index_path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                index_data = {"entries": parsed.get("entries", [])}

            # 获取当前条目数
            entries = self.list_entries(date)

            # 构建以 date 为键的字典
            entry_map = {}
            for e in index_data["entries"]:
                if isinstance(e, dict) and "date" in e:
                    entry_map[e["date"]] = e

            entry_map[date] = {
                "date": date,
                "path": f"memory/{date}.md",
                "entries_count": len(entries)
            }

            # 重新排序并保存
            index_data["entries"] = sorted(
                entry_map.values(),
                key=lambda x: x["date"],
                reverse=True
            )

            index_path.write_text(json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8")

        except Exception as e:
            print(f"Error updating index: {e}")


# =============================================================================
# MemoryMD 类
# =============================================================================

class MemoryMD:
    """
    长期记忆 - MEMORY.md
    """

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.file_path = self.workspace_path / "MEMORY.md"
        self._ensure_file()

    def _ensure_file(self) -> None:
        """确保文件存在"""
        if not self.file_path.exists():
            # 创建基础结构
            initial_content = """# 长期记忆

_最后更新: {date}_

---

## 事实 (Facts)

---

## 偏好 (Preferences)

---

## 学习 (Learnings)

---

## 决策 (Decisions)

""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M"))
            self.file_path.write_text(initial_content, encoding="utf-8")

    def _get_section_content(self, section: str) -> str:
        """获取指定区域的内容"""
        try:
            content = self.file_path.read_text(encoding="utf-8")
            pattern = rf"## {section}.*?\n(.*?)(?=\n## |\Z)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""
        except Exception:
            return ""

    def append_fact(self, fact: str, category: str, importance: float = 1.0) -> bool:
        """
        添加新事实

        Args:
            fact: 事实内容
            category: 分类 (general/project/decision/preference/learning)
            importance: 重要性 (0.0-1.0)

        Returns:
            bool: 成功返回 True
        """
        try:
            content = self.file_path.read_text(encoding="utf-8")

            # 更新最后更新时间
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            content = re.sub(
                r"_最后更新: .*?_",
                f"_最后更新: {now}_",
                content
            )

            # 构建新条目
            date_str = datetime.now().strftime("%Y-%m-%d")
            new_entry = f"- [{date_str}] [{category.upper()}] {fact} _(importance: {importance})_\n"

            # 找到 Facts 区域
            facts_pattern = r"(## 事实 \(Facts\)\n)(---*)"
            match = re.search(facts_pattern, content)

            if match:
                insert_pos = match.end()
                content = content[:insert_pos] + new_entry + "\n" + content[insert_pos:]
            else:
                # 如果没有 Facts 区域，在适当位置插入
                content += "\n" + new_entry

            self.file_path.write_text(content, encoding="utf-8")
            return True

        except Exception as e:
            print(f"Error appending fact: {e}")
            return False

    def update_preference(self, key: str, value: str) -> bool:
        """
        更新偏好设置

        Args:
            key: 偏好键
            value: 偏好值

        Returns:
            bool: 成功返回 True
        """
        try:
            content = self.file_path.read_text(encoding="utf-8")

            # 更新最后更新时间
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            content = re.sub(
                r"_最后更新: .*?_",
                f"_最后更新: {now}_",
                content
            )

            # 检查是否已存在该偏好
            pref_pattern = rf"- \*\*{re.escape(key)}\*\*: (.+?)(?:\n|$)"
            match = re.search(pref_pattern, content)

            if match:
                # 更新现有偏好
                old_line = match.group(0)
                new_line = f"- **{key}**: {value}"
                content = content.replace(old_line, new_line)
            else:
                # 添加新偏好到 Preferences 区域
                pref_section_pattern = r"(## 偏好 \(Preferences\)\n)(---*)"
                pref_match = re.search(pref_section_pattern, content)

                if pref_match:
                    insert_pos = pref_match.end()
                    new_entry = f"- **{key}**: {value}\n"
                    content = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]

            self.file_path.write_text(content, encoding="utf-8")
            return True

        except Exception as e:
            print(f"Error updating preference: {e}")
            return False

    def read(self) -> str:
        """
        读取完整的 MEMORY.md 内容

        Returns:
            str: 完整内容
        """
        try:
            return self.file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading MEMORY.md: {e}")
            return ""

    def search(self, query: str) -> list[dict]:
        """
        搜索记忆内容

        Args:
            query: 搜索查询

        Returns:
            list[dict]: 匹配的结果列表
        """
        results = []
        try:
            content = self.file_path.read_text(encoding="utf-8")

            if query.lower() not in content.lower():
                return results

            lines = content.split("\n")
            current_section = ""

            for line in lines:
                if line.strip().startswith("## "):
                    current_section = line.strip().replace("## ", "").replace(" (", "/").replace(")", "")
                elif line.strip().startswith("- [") and query.lower() in line.lower():
                    # 匹配事实条目
                    match = re.match(r"- \[([^\]]+)\] \[([^\]]+)\] (.+?) \(importance: ([\d.]+)\)", line.strip())
                    if match:
                        date_str, category, fact_text, importance = match.groups()
                        results.append({
                            "type": "fact",
                            "section": current_section,
                            "date": date_str,
                            "category": category,
                            "content": fact_text,
                            "importance": float(importance)
                        })
                elif line.strip().startswith("- **") and query.lower() in line.lower():
                    # 匹配偏好条目
                    match = re.match(r"- \*\*(.+?)\*\*: (.+)", line.strip())
                    if match:
                        key, value = match.groups()
                        results.append({
                            "type": "preference",
                            "section": current_section,
                            "key": key,
                            "value": value
                        })

            return results

        except Exception as e:
            print(f"Error searching MEMORY.md: {e}")
            return []

    def get_all_facts(self) -> list[dict]:
        """获取所有事实记忆"""
        return self.search("")

    def get_preferences(self) -> dict:
        """获取所有偏好设置"""
        prefs = {}
        try:
            content = self.file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            in_prefs = False

            for line in lines:
                if "## 偏好" in line:
                    in_prefs = True
                elif in_prefs and line.strip().startswith("## "):
                    break
                elif in_prefs and line.strip().startswith("- **"):
                    match = re.match(r"- \*\*(.+?)\*\*: (.+)", line.strip())
                    if match:
                        key, value = match.groups()
                        prefs[key] = value

        except Exception:
            pass
        return prefs


# =============================================================================
# FilePersistStore 类
# =============================================================================

class FilePersistStore:
    """
    文件持久化存储总接口
    整合 DailyMemory 和 MemoryMD，提供统一的 L4 层接口
    """

    def __init__(self, workspace_path: Optional[str] = None):
        if workspace_path is None:
            # 默认使用当前工作目录
            workspace_path = str(Path(__file__).parent.parent.parent.parent)
        self.workspace_path = workspace_path
        self.daily_memory = DailyMemory(workspace_path)
        self.memory_md = MemoryMD(workspace_path)

    def store_fact(self, content: str, metadata: dict) -> bool:
        """
        存储事实到 L4 层

        Args:
            content: 事实内容
            metadata: 元数据，包含 category, importance 等

        Returns:
            bool: 成功返回 True
        """
        try:
            category = metadata.get("category", "general")
            importance = metadata.get("importance", 1.0)

            # 同时写入日记和长期记忆
            # 1. 写入日记（用于时间线追溯）
            self.daily_memory.append(
                entry=f"{content}",
                category=category
            )

            # 2. 对于重要的事实，写入 MEMORY.md
            if importance >= 0.7:
                self.memory_md.append_fact(
                    fact=content,
                    category=category,
                    importance=importance
                )

            return True

        except Exception as e:
            print(f"Error storing fact: {e}")
            return False

    def get_recent(self, days: int = 7) -> list[str]:
        """
        获取最近 N 天的日记

        Args:
            days: 天数

        Returns:
            list[str]: 日记内容列表（按日期倒序）
        """
        results = []
        try:
            today = datetime.now()

            for i in range(days):
                date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                content = self.daily_memory.read(date)
                if content and "暂无记录" not in content:
                    results.append(content)

            return results

        except Exception as e:
            print(f"Error getting recent diaries: {e}")
            return []

    def get_all_facts(self) -> list[dict]:
        """
        获取所有长期记忆

        Returns:
            list[dict]: 事实列表
        """
        return self.memory_md.get_all_facts()

    def sync_from_l3(self, vector_store_stats: dict) -> int:
        """
        从 L3（向量库）同步重要记忆到 L4

        当 L3 中有高频或高权重记忆时，将其持久化到文件系统

        Args:
            vector_store_stats: 向量库统计信息，包含：
                - top_memories: list[dict] - 高权重记忆
                - high_frequency: list[dict] - 高频记忆

        Returns:
            int: 同步的记忆数量
        """
        synced_count = 0

        try:
            top_memories = vector_store_stats.get("top_memories", [])
            high_frequency = vector_store_stats.get("high_frequency", [])

            # 处理高权重记忆
            for memory in top_memories:
                importance = memory.get("score", 0.5)
                if importance >= 0.8:
                    self.store_fact(
                        content=memory.get("content", ""),
                        metadata={
                            "category": memory.get("category", "general"),
                            "importance": importance
                        }
                    )
                    synced_count += 1

            # 处理高频记忆
            for memory in high_frequency:
                self.store_fact(
                    content=memory.get("content", ""),
                    metadata={
                        "category": memory.get("category", "general"),
                        "importance": 0.6  # 高频记忆给予中等权重
                    }
                )
                synced_count += 1

        except Exception as e:
            print(f"Error syncing from L3: {e}")

        return synced_count

    def export(self, format: str = "json") -> str:
        """
        导出记忆数据

        Args:
            format: 导出格式 (json/markdown)

        Returns:
            str: 导出的数据
        """
        try:
            if format == "json":
                export_data = {
                    "exported_at": datetime.now().isoformat(),
                    "long_term_memory": {
                        "facts": self.memory_md.get_all_facts(),
                        "preferences": self.memory_md.get_preferences()
                    },
                    "recent_diaries": [],
                    "index": self._get_index()
                }

                # 收集最近日记
                recent = self.get_recent(30)
                for diary in recent:
                    export_data["recent_diaries"].append(diary)

                return json.dumps(export_data, ensure_ascii=False, indent=2)

            elif format == "markdown":
                lines = [
                    "# 记忆导出",
                    f"_导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
                    "---",
                    "## 长期记忆 (MEMORY.md)",
                    self.memory_md.read(),
                    "---",
                    "## 最近日记"
                ]

                recent = self.get_recent(7)
                for diary in recent:
                    lines.append(f"\n{diary}\n---")

                return "\n".join(lines)

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            print(f"Error exporting memory: {e}")
            return ""

    def _get_index(self) -> dict:
        """获取索引信息"""
        try:
            index_path = Path(self.workspace_path) / "memory" / "index.json"
            if index_path.exists():
                return json.loads(index_path.read_text(encoding="utf-8"))
            return {"entries": []}
        except Exception:
            return {"entries": []}

    def get_stats(self) -> dict:
        """获取存储统计信息"""
        try:
            index = self._get_index()
            total_entries = sum(e.get("entries_count", 0) for e in index.get("entries", []))

            return {
                "total_diary_files": len(index.get("entries", [])),
                "total_diary_entries": total_entries,
                "memory_md_exists": self.memory_md.file_path.exists(),
                "workspace": self.workspace_path
            }
        except Exception:
            return {}


# =============================================================================
# 便捷函数
# =============================================================================

def get_default_store() -> FilePersistStore:
    """获取默认的存储实例"""
    return FilePersistStore()


# =============================================================================
# 入口点（可直接测试）
# =============================================================================

if __name__ == "__main__":
    # 测试代码
    store = get_default_store()

    print("=== FilePersistStore Test ===\n")

    # 测试存储事实
    print("1. Testing store_fact...")
    store.store_fact(
        content="测试记忆条目",
        metadata={"category": "general", "importance": 0.9}
    )
    print("   Done.\n")

    # 测试获取最近日记
    print("2. Testing get_recent...")
    recent = store.get_recent(3)
    print(f"   Found {len(recent)} recent diaries\n")

    # 测试获取统计
    print("3. Testing get_stats...")
    stats = store.get_stats()
    print(f"   Stats: {stats}\n")

    # 测试导出
    print("4. Testing export...")
    exported = store.export("json")
    print(f"   Exported {len(exported)} chars\n")

    print("=== All tests completed ===")
