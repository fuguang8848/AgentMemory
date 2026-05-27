"""
遗忘引擎 - Mem0 风格的遗忘算法
"""

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# 默认配置
DEFAULT_HALF_LIFE_DAYS: float = 14.0
DEFAULT_FORGET_THRESHOLD: float = 0.3
DEFAULT_ARCHIVE_THRESHOLD: float = 0.5
DEFAULT_MAX_ARCHIVED: int = 1000

ARCHIVE_DIR = Path(__file__).parent / "data" / "archive"


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
    遗忘引擎
    
    基于访问频率、重要性和时效性计算记忆衰减分数
    """

    def __init__(
        self,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        forget_threshold: float = DEFAULT_FORGET_THRESHOLD,
        archive_threshold: float = DEFAULT_ARCHIVE_THRESHOLD,
        max_archived: int = DEFAULT_MAX_ARCHIVED,
    ):
        self.half_life_days = half_life_days
        self.forget_threshold = forget_threshold
        self.archive_threshold = archive_threshold
        self.max_archived = max_archived

    def decay_factor(self, recency_days: float, half_life_days: Optional[float] = None) -> float:
        """
        计算衰减因子
        
        公式: 2^(-recency_days / half_life_days)
        14天前的记忆衰减到 0.5
        
        Args:
            recency_days: 距离上次访问的天数
            half_life_days: 半衰期天数
            
        Returns:
            衰减因子 (0-1)
        """
        if half_life_days is None:
            half_life_days = self.half_life_days
            
        if recency_days < 0:
            recency_days = 0
            
        return 2.0 ** (-recency_days / half_life_days)

    def calculate_access_freq_score(self, entry: dict) -> tuple[float, str]:
        """
        计算访问频率评分
        
        Args:
            entry: 记忆条目
            
        Returns:
            (score, reason)
        """
        access_count = entry.get("access_count", 0)
        
        # 归一化: 使用对数函数平滑处理高频访问
        # 假设访问10次以上接近饱和
        if access_count == 0:
            score = 0.0
            reason = f"无访问记录 (count=0)"
        elif access_count == 1:
            score = 0.2
            reason = f"仅访问1次 (count=1)"
        elif access_count <= 3:
            score = 0.4
            reason = f"少量访问 (count={access_count})"
        elif access_count <= 10:
            score = 0.6 + 0.2 * (access_count - 3) / 7
            reason = f"适度访问 (count={access_count})"
        else:
            score = min(0.8 + 0.2 * math.log10(access_count - 9), 1.0)
            reason = f"高频访问 (count={access_count})"
            
        return score, reason

    def calculate_recency_score(self, entry: dict) -> tuple[float, str]:
        """
        计算时效性评分
        
        Args:
            entry: 记忆条目
            
        Returns:
            (score, reason)
        """
        last_accessed = entry.get("last_accessed")
        
        if last_accessed is None:
            # 无访问记录，检查 created_at
            last_accessed = entry.get("created_at")
            
        if last_accessed is None:
            score = 0.0
            reason = "无时间戳信息"
            return score, reason
            
        # 解析时间戳
        if isinstance(last_accessed, (int, float)):
            # Unix 时间戳（秒或毫秒）
            if last_accessed > 1e12:
                last_accessed = last_accessed / 1000
            last_accessed_dt = datetime.fromtimestamp(last_accessed, tz=timezone.utc)
        elif isinstance(last_accessed, str):
            # ISO 格式字符串
            last_accessed_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
            # 如果解析为 naive datetime，假设为 UTC
            if last_accessed_dt.tzinfo is None:
                last_accessed_dt = last_accessed_dt.replace(tzinfo=timezone.utc)
        else:
            last_accessed_dt = last_accessed
            if last_accessed_dt.tzinfo is None:
                last_accessed_dt = last_accessed_dt.replace(tzinfo=timezone.utc)
            
        # 计算天数差
        now = datetime.now(timezone.utc)
        recency_days = (now - last_accessed_dt).total_seconds() / 86400.0
        
        if recency_days < 0:
            recency_days = 0
            reason = "未来时间（异常）"
        elif recency_days < 1:
            reason = f"刚刚访问 ({recency_days:.1f}小时前)"
        elif recency_days < 7:
            reason = f"近期访问 ({recency_days:.1f}天前)"
        elif recency_days < 14:
            reason = f"一周前访问 ({recency_days:.1f}天前)"
        elif recency_days < 30:
            reason = f"较早访问 ({recency_days:.1f}天前)"
        else:
            reason = f"久未访问 ({recency_days:.1f}天前)"
            
        score = self.decay_factor(recency_days)
        return score, reason

    def calculate_score(self, entry: dict) -> DecayScore:
        """
        计算综合衰减分数
        
        公式: final = access_freq * 0.3 + importance * 0.3 + recency * 0.4
        
        Args:
            entry: 记忆条目
            
        Returns:
            DecayScore 对象
        """
        memory_id = entry.get("id") or entry.get("memory_id") or "unknown"
        
        # 各维度得分
        access_freq_score, access_reason = self.calculate_access_freq_score(entry)
        importance = entry.get("importance", 0.5)  # 默认0.5
        recency_score, recency_reason = self.calculate_recency_score(entry)
        
        # 验证 importance 范围
        if not 0 <= importance <= 1:
            importance = 0.5
            
        # 加权计算
        final_score = (
            access_freq_score * 0.3 +
            importance * 0.3 +
            recency_score * 0.4
        )
        
        # 限制范围
        final_score = max(0.0, min(1.0, final_score))
        
        # 构建原因列表
        reasons = [
            f"访问频率: {access_reason}",
            f"重要性: {importance:.2f}",
            f"时效性: {recency_reason}",
            f"权重计算: {access_freq_score:.2f}*0.3 + {importance:.2f}*0.3 + {recency_score:.2f}*0.4 = {final_score:.3f}",
        ]
        
        # 详细评分说明
        if final_score < 0.2:
            summary = "极低分 - 建议遗忘"
        elif final_score < self.forget_threshold:
            summary = "低于遗忘阈值 - 建议遗忘"
        elif final_score < self.archive_threshold:
            summary = "中等分数 - 建议归档"
        else:
            summary = "高分 - 保留"
        reasons.insert(0, summary)
        
        return DecayScore(
            memory_id=memory_id,
            score=final_score,
            components={
                "access_freq": access_freq_score,
                "importance": importance,
                "recency": recency_score,
            },
            reasons=reasons,
        )

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
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    forget_threshold: float = DEFAULT_FORGET_THRESHOLD,
    archive_threshold: float = DEFAULT_ARCHIVE_THRESHOLD,
) -> DecayEngine:
    """创建遗忘引擎实例"""
    return DecayEngine(
        half_life_days=half_life_days,
        forget_threshold=forget_threshold,
        archive_threshold=archive_threshold,
    )


def create_archiver(
    archive_dir: Optional[str] = None,
    max_archived: int = DEFAULT_MAX_ARCHIVED,
) -> MemoryArchiver:
    """创建归档器实例"""
    path = Path(archive_dir) if archive_dir else None
    return MemoryArchiver(archive_dir=path, max_archived=max_archived)
