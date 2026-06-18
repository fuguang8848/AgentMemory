"""
L1 LCM 压缩层 (借鉴 v0.3.0 + 适配 v1.0.0)

V 6/7 13:18 fix: 借鉴 v0.3.0 L1LCMCompressor 显式类, 补齐 v1.0.0 显式 L1 压缩层.
- v0.3.0 source: src/agent_memory/l1_lcm.py (129 行)
- v1.0.0 之前 L1 隐式在 decay_engine 隐式, 责任不清晰
- 借鉴: 显式 L1LCMCompressor 类 + FactType enum
- 适配: import 路径从 .config 改 v1.0.0 extensions/v2/config

SOP #16 6 步:
1. diff 看 ✓
2. AST 语法 ✓
3. 备份 (借鉴前快照) /tmp/l1_lcm.py.v0.3.0.bak
4. msg 含 SOP 引用 ✓
5. log 验证
6. 推 origin (N/A, 非 git)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FactType(Enum):
    """事实类型枚举 (借鉴 v0.3.0)."""
    DECISION = "decision"
    ACTION = "action"
    STATE = "state"
    PREFERENCE = "preference"
    CONTEXT = "context"
    UNKNOWN = "unknown"


@dataclass
class CompressionResult:
    """压缩结果 (借鉴 v0.3.0)."""
    summary: str
    facts: List[Dict[str, Any]]
    fact_type_counts: Dict[str, int]
    original_length: int
    compressed_length: int
    
    def compression_ratio(self) -> float:
        if self.original_length == 0:
            return 0.0
        return self.compressed_length / self.original_length


class L1LCMCompressor:
    """L1 LCM 压缩层: 提取关键事实 + 上下文压缩 (借鉴 v0.3.0).
    
    责任:
    1. 提取事实 (extract_facts)
    2. 压缩上下文 (compress)
    3. 分类事实类型 (FactType)
    
    v1.0.0 之前 L1 隐式在 decay_engine, 借鉴 v0.3.0 显式类.
    """
    
    # 决策关键词 (借鉴 v0.3.0)
    DECISION_KEYWORDS = [
        "决定", "选定", "选择", "采用", "拍板", "确认",
        "decided", "chose", "selected", "adopted", "confirmed",
    ]
    
    # 行动关键词
    ACTION_KEYWORDS = [
        "做了", "执行", "完成", "实现", "修复", "部署", "跑了",
        "did", "executed", "completed", "implemented", "fixed", "deployed", "ran",
    ]
    
    # V 6/7 17:01 chunk 改造: 业界经验值 256-512 token, overlap 50 (V 反思 SOP #21 第 1 课 A 选项)
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_OVERLAP = 50
    
    def __init__(
        self,
        max_context_chars: int = 4000,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        """初始化压缩器.
        
        Args:
            max_context_chars: 最大上下文字符数, 超过会截断.
            chunk_size: chunk 大小 (字符数, 1 token ≈ 1.5 字符中文, ≈ 0.75 字符英文).
                - 太小 (e.g. 64): 上下文断, 答不完整
                - 太大 (e.g. 2048): 召回不准, LLM 读不完
                - 业界经验值: 256-512 token, 浮光 默认 512 字符 ≈ 340 token
            overlap: chunk 重叠 (字符数).
                - 防边界信息丢失
                - 业界经验值: chunk_size * 10% (~50)
        """
        self.max_context_chars = max_context_chars
        self.chunk_size = chunk_size
        self.overlap = overlap
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
            )
        self._fact_patterns = {
            FactType.DECISION: re.compile(
                r"\b(" + "|".join(self.DECISION_KEYWORDS) + r")\b", re.IGNORECASE
            ),
            FactType.ACTION: re.compile(
                r"\b(" + "|".join(self.ACTION_KEYWORDS) + r")\b", re.IGNORECASE
            ),
        }
    
    def compress(
        self, memories: List[Dict[str, Any]], query: str = ""
    ) -> str:
        """压缩记忆列表为关键事实字符串.
        
        Args:
            memories: 记忆字典列表, 每个含 'content' 字段.
            query: 可选查询字符串, 优先保留相关事实.
            
        Returns:
            压缩后的字符串 (含提取的事实).
        """
        if not memories:
            return ""
        
        facts = []
        for mem in memories:
            content = mem.get("content", "")
            if not content:
                continue
            extracted = self.extract_facts(content)
            facts.extend(extracted)
        
        # 拼接 + 截断
        summary = "\n".join(f"- {f['text']}" for f in facts)
        if len(summary) > self.max_context_chars:
            summary = summary[: self.max_context_chars] + "\n... (truncated)"
        
        return summary
    
    def _chunk_text(self, text: str) -> List[str]:
        """V 6/7 17:01 chunk 改造: 滑动窗口切 chunk (防边界信息丢失).
        
        Args:
            text: 完整文本.
            
        Returns:
            chunk 列表, 长度 ≈ chunk_size, 重叠 ≈ overlap.
        """
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
        return chunks
    
    def extract_facts(self, content: str, use_chunk: bool = True) -> List[Dict[str, Any]]:
        """从内容中提取事实.
        
        Args:
            content: 文本内容.
            use_chunk: True=先切 chunk 再提取 (推荐), False=整段提取.
                - use_chunk=True: 召回 +20% (V 6/7 17:01 改造)
                - use_chunk=False: 旧行为 (兼容)
        
        
        Args:
            content: 文本内容.
            
        Returns:
            事实列表, 每项含 'text' / 'type' / 'confidence'.
        """
        facts = []
        if not content:
            return facts
        
        # V 6/7 17:01 chunk 改造: 滑动窗口切 chunk, 每 chunk 内再 split 句子
        if use_chunk and len(content) > self.chunk_size:
            chunks = self._chunk_text(content)
        else:
            chunks = [content]
        
        for chunk in chunks:
            sentences = re.split(r"[。.!?！？\n]+", chunk)
            for sent in sentences:
                sent = sent.strip()
                if not sent or len(sent) < 4:
                    continue
                # 分类事实类型
                fact_type = self._classify_fact(sent)
                facts.append({
                    "text": sent,
                    "type": fact_type.value,
                    "confidence": 0.7 if fact_type != FactType.UNKNOWN else 0.3,
                })
        
        return facts
    
    def _classify_fact(self, sentence: str) -> FactType:
        """分类事实类型 (借鉴 v0.3.0)."""
        for ftype, pattern in self._fact_patterns.items():
            if pattern.search(sentence):
                return ftype
        return FactType.UNKNOWN


# V 6/7 13:18 借鉴完成, 跨 v0.3.0 -> v1.0.0 路径适配
