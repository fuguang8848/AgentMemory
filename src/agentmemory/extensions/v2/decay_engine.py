"""
遗忘引擎 - Mem0 风格的遗忘算法
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


# 默认配置
DEFAULT_HALF_LIFE_DAYS: float = 14.0
DEFAULT_FORGET_THRESHOLD: float = 0.3
DEFAULT_ARCHIVE_THRESHOLD: float = 0.5
DEFAULT_MAX_ARCHIVED: int = 1000

ARCHIVE_DIR = Path(__file__).parent / "data" / "archive"


@dataclass
class DecayPolicy:
    """遗忘策略（可配置，与架构文档 v2-architecture.md §5.9 一致）"""
    weight_access: float = 0.3          # log(1+access) 的指数
    weight_importance: float = 0.4      # importance 的指数
    weight_recency: float = 0.3         # recency 的指数
    half_life_days: float = 30.0        # 半衰期（30天）
    forget_threshold: float = 0.2        # 低于此值遗忘
    archive_threshold: float = 0.5       # 低于此值归档


@dataclass
class DecayScore:
    """记忆衰减评分"""
    memory_id: str
    score: float
    components: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DecayEngine:
    """
    遗忘引擎 v2
    
    公式（与架构文档 v2-architecture.md §5.9 一致）:
        score = (log(1+access))^0.3 × importance^0.4 × recency^0.3
        recency = 0.5 ** (age_days / half_life_days)
    """

    def __init__(
        self,
        policy: Optional[DecayPolicy] = None,
        max_archived: int = DEFAULT_MAX_ARCHIVED,
    ):
        self.policy = policy or DecayPolicy()
        self.forget_threshold = self.policy.forget_threshold
        self.archive_threshold = self.policy.archive_threshold
        self.max_archived = max_archived

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """解析时间戳为 datetime"""
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            if ts > 1e12:
                ts = ts / 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        if hasattr(ts, 'tzinfo'):
            return ts
        return None

    def _age_days(self, created_at: Any, last_accessed: Any) -> float:
        """计算记忆年龄（天数，以最近访问时间为主）"""
        ref_time = last_accessed or created_at
        dt = self._parse_timestamp(ref_time)
        if dt is None:
            return 0.0
        now = datetime.now(timezone.utc)
        age = (now - dt).total_seconds() / 86400.0
        return max(0.0, age)

    def calculate_score(self, entry: dict) -> DecayScore:
        """
        计算综合衰减分数（架构公式）
        
        score = (log(1+access))^0.3 × importance^0.4 × recency^0.3
        recency = 0.5 ** (age_days / half_life_days)
        
        Args:
            entry: 记忆条目，需包含 access_count, importance, last_accessed, created_at
            
        Returns:
            DecayScore 对象
        """
        memory_id = entry.get("id") or entry.get("memory_id") or "unknown"
        
        # 提取字段
        access_count = entry.get("access_count", 0)
        importance = entry.get("importance", 0.5)
        last_accessed = entry.get("last_accessed")
        created_at = entry.get("created_at")
        
        # 归一化 importance
        importance = max(0.0, min(1.0, float(importance)))
        
        # 计算三因子
        # 1. 访问频率因子: (log(1+access))^0.3
        # 注意：该值可能超过 1.0（高频访问时），最终分数会截断到 [0,1]
        access_factor = (math.log1p(access_count)) ** self.policy.weight_access
        
        # 2. 重要性因子: importance^0.4
        importance_factor = importance ** self.policy.weight_importance
        
        # 3. 时效性因子: recency^0.3，recency = 0.5^(age_days/half_life_days)
        age_days = self._age_days(created_at, last_accessed)
        recency = 0.5 ** (age_days / self.policy.half_life_days)
        recency_factor = recency ** self.policy.weight_recency
        
        # 乘积为最终分数
        final_score = access_factor * importance_factor * recency_factor
        # 架构公式是连乘，各因子范围不同，乘积可能超 1.0，截断到 [0,1]
        final_score = max(0.0, min(1.0, final_score))
        
        # 构建原因
        reasons = [
            f"访问频率: log(1+{access_count})^0.3 = {access_factor:.4f}",
            f"重要性: {importance}^0.4 = {importance_factor:.4f}",
            f"时效性: 0.5^({age_days:.1f}/{self.policy.half_life_days})^0.3 = {recency_factor:.4f}",
            f"最终分数: {access_factor:.4f} × {importance_factor:.4f} × {recency_factor:.4f} = {final_score:.4f}",
        ]
        
        if final_score < 0.2:
            summary = "极低分 — 建议遗忘"
        elif final_score < self.forget_threshold:
            summary = "低于遗忘阈值 — 建议遗忘"
        elif final_score < self.archive_threshold:
            summary = "中等分数 — 建议归档"
        else:
            summary = "高分 — 保留"
        reasons.insert(0, summary)
        
        return DecayScore(
            memory_id=memory_id,
            score=final_score,
            components={
                "access_factor": access_factor,
                "importance_factor": importance_factor,
                "recency_factor": recency_factor,
            },
            reasons=reasons,
        )

    def calculate_score_from_fields(
        self,
        importance: float,
        access_count: int,
        last_accessed: Any,
        created_at: Any,
    ) -> float:
        """
        直接从字段计算分数（符合架构协议 DecayEngine.calculate_score）
        
        score = (log(1+access))^0.3 × importance^0.4 × recency^0.3
        recency = 0.5 ** (age_days / half_life_days)
        """
        entry = {
            "importance": importance,
            "access_count": access_count,
            "last_accessed": last_accessed,
            "created_at": created_at,
        }
        return self.calculate_score(entry).score

    def should_forget(self, score: DecayScore) -> bool:
        """
        判断是否应该遗忘
        
        Args:
            score: DecayScore 对象
            
        Returns:
            True 表示应该遗忘
        """
        return score.score < self.forget_threshold

    def should_archive(self, score: DecayScore) -> bool:
        """
        判断是否应该归档
        
        中等分数（0.3-0.5）→ 归档不删除
        
        Args:
            score: DecayScore 对象
            
        Returns:
            True 表示应该归档
        """
        return self.forget_threshold <= score.score < self.archive_threshold

    def run_decay_check(self, entries: list[dict]) -> dict:
        """
        运行衰减检查
        
        Args:
            entries: 记忆条目列表
            
        Returns:
            {"forget": [...], "archive": [...], "keep": [...]}，每个元素包含 entry 和 score
        """
        result: dict = {
            "forget": [],
            "archive": [],
            "keep": [],
        }
        
        for entry in entries:
            try:
                decay_score = self.calculate_score(entry)
                
                entry_with_score = {
                    "entry": entry,
                    "score": decay_score.to_dict(),
                }
                
                if self.should_forget(decay_score):
                    result["forget"].append(entry_with_score)
                elif self.should_archive(decay_score):
                    result["archive"].append(entry_with_score)
                else:
                    result["keep"].append(entry_with_score)
                    
            except Exception as e:
                # 错误处理：记录失败条目
                entry_id = entry.get("id") or entry.get("memory_id", "unknown")
                result["keep"].append({
                    "entry": entry,
                    "score": None,
                    "error": str(e),
                })
                
        return result


class MemoryArchiver:
    """
    记忆归档器
    
    将低价值记忆归档到深层存储，支持恢复
    """
    
    def __init__(self, archive_dir: Optional[Path] = None, max_archived: int = DEFAULT_MAX_ARCHIVED):
        self.archive_dir = archive_dir or ARCHIVE_DIR
        self.archive_dir = Path(self.archive_dir)
        self.max_archived = max_archived
        self._ensure_archive_dir()
        
    def _ensure_archive_dir(self) -> None:
        """确保归档目录存在"""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_archive_path(self, memory_id: str) -> Path:
        """获取记忆归档文件路径"""
        # 使用前两个字符作为子目录，避免单目录文件过多
        prefix = memory_id[:2] if len(memory_id) >= 2 else memory_id
        subdir = self.archive_dir / prefix
        subdir.mkdir(exist_ok=True)
        return subdir / f"{memory_id}.json"
        
    def archive_to_deep_storage(self, memory_id: str, memory_data: dict) -> bool:
        """
        归档记忆到深层存储
        
        Args:
            memory_id: 记忆ID
            memory_data: 记忆数据
            
        Returns:
            True 表示归档成功
        """
        try:
            # 检查是否已存在
            archive_path = self._get_archive_path(memory_id)
            if archive_path.exists():
                return False
                
            # 添加归档元数据
            archive_record = {
                "memory_id": memory_id,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "original_data": memory_data,
            }
            
            # 写入归档文件
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(archive_record, f, ensure_ascii=False, indent=2)
                
            # 检查并清理超过最大数量的归档
            self._cleanup_old_archives()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"归档失败: {e}") from e
            
    def restore_from_archive(self, memory_id: str) -> Optional[dict]:
        """
        从归档恢复记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            记忆数据，如果不存在则返回 None
        """
        try:
            archive_path = self._get_archive_path(memory_id)
            
            if not archive_path.exists():
                return None
                
            with open(archive_path, "r", encoding="utf-8") as f:
                archive_record = json.load(f)
                
            # 删除归档文件
            archive_path.unlink()
            
            # 返回原始数据
            return archive_record.get("original_data")
            
        except Exception as e:
            raise RuntimeError(f"恢复失败: {e}") from e
            
    def list_archived(self) -> list[dict]:
        """
        列出所有已归档的记忆
        
        Returns:
            归档记忆列表（不含完整数据，只含元信息）
        """
        try:
            archived = []
            
            for item in self.archive_dir.rglob("*.json"):
                try:
                    with open(item, "r", encoding="utf-8") as f:
                        record = json.load(f)
                        
                    archived.append({
                        "memory_id": record.get("memory_id"),
                        "archived_at": record.get("archived_at"),
                        "file_path": str(item.relative_to(self.archive_dir)),
                    })
                    
                except (json.JSONDecodeError, IOError):
                    continue
                    
            # 按归档时间排序
            archived.sort(key=lambda x: x.get("archived_at", ""), reverse=True)
            
            return archived
            
        except Exception as e:
            raise RuntimeError(f"列出归档失败: {e}") from e
            
    def _cleanup_old_archives(self) -> int:
        """
        清理超过最大数量的旧归档
        
        Returns:
            清理的文件数量
        """
        archived = self.list_archived()
        
        if len(archived) <= self.max_archived:
            return 0
            
        # 删除最旧的归档
        to_delete = archived[self.max_archived:]
        deleted = 0
        
        for item in to_delete:
            try:
                path = self.archive_dir / item["file_path"]
                if path.exists():
                    path.unlink()
                    deleted += 1
            except OSError:
                continue
                
        return deleted
        
    def delete_archive(self, memory_id: str) -> bool:
        """
        删除归档（不恢复）
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            True 表示删除成功
        """
        try:
            archive_path = self._get_archive_path(memory_id)
            
            if not archive_path.exists():
                return False
                
            archive_path.unlink()
            return True
            
        except Exception:
            return False


# 便捷函数
def create_decay_engine(
    half_life_days: float = 30.0,
    forget_threshold: float = 0.2,
    archive_threshold: float = 0.5,
) -> DecayEngine:
    """创建遗忘引擎实例（兼容旧 API，内部使用 DecayPolicy）"""
    policy = DecayPolicy(
        half_life_days=half_life_days,
        forget_threshold=forget_threshold,
        archive_threshold=archive_threshold,
    )
    return DecayEngine(policy=policy)


def create_archiver(
    archive_dir: Optional[str] = None,
    max_archived: int = DEFAULT_MAX_ARCHIVED,
) -> MemoryArchiver:
    """创建归档器实例"""
    path = Path(archive_dir) if archive_dir else None
    return MemoryArchiver(archive_dir=path, max_archived=max_archived)
