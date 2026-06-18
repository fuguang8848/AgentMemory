"""
P0-2 注入检测 (InjectionDetector) — 浮光 13:25 选 B 触发

V 6/7 13:30 fix: 写 P0-2 注入检测, 补齐 6 端口服务 + 6 交响乐包安全.
- 浮光 13:25 拍板 "2" (3 都选 B), V 立即开干
- v0.3.0 也没, V 借鉴 issue 报告里的代码骨架 + 6 attack pattern
- v1.0.0 0 引用, V 拍板新写

SOP #16 6 步:
1. diff 看 ✓
2. AST 语法 ✓
3. 备份 (非 git 仓, 走 .bak) — extensions/v2/ 备份
4. msg 含 SOP 引用 ✓
5. log 验证
6. 推 origin (N/A, 非 git)
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class AttackType(Enum):
    """攻击类型枚举."""
    PROMPT_INJECTION = "prompt_injection"      # 经典 prompt injection
    ROLE_HIJACK = "role_hijack"                 # 角色劫持
    SYSTEM_LEAK = "system_leak"                 # 系统提示泄露
    CHATML_TAG = "chatml_tag"                   # ChatML 标签注入
    JAILBREAK = "jailbreak"                     # 角色越狱 (DAN 等)
    COMMAND_INJECTION = "command_injection"     # 命令注入
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """检测结果."""
    is_injection: bool
    attack_type: AttackType
    matched_pattern: str
    risk_level: str  # "low" | "medium" | "high" | "critical"
    reason: str
    
    def to_dict(self) -> dict:
        return {
            "is_injection": self.is_injection,
            "attack_type": self.attack_type.value,
            "matched_pattern": self.matched_pattern,
            "risk_level": self.risk_level,
            "reason": self.reason,
        }


class InjectionDetector:
    """P0-2 注入检测器: 6 attack pattern + 风险分级 (借鉴 v0.3.0 issue 报告).
    
    责任:
    1. detect(text) -> DetectionResult
    2. batch_detect(texts) -> List[DetectionResult]
    3. sanitize(text) -> str  (可选: 移除攻击 pattern)
    
    设计:
    - 6 attack pattern (regex)
    - 风险分级: critical / high / medium / low
    - 性能: 1 次扫描 6 pattern, O(n) 文本长度
    - 误报控制: 加 negative_keywords 白名单 (e.g. 'ignore' 在代码里合法)
    """
    
    # 6 attack pattern + 风险级别
    PATTERNS = [
        # 1. 经典 prompt injection (high)
        (
            AttackType.PROMPT_INJECTION,
            r"\b(ignore|disregard|forget)\b[\s\S]{0,30}\b(instructions?|prompts?|rules?|commands?|directives?)\b",
            "high",
            "Attempt to override previous instructions",
        ),
        # 2. 角色劫持 (high)
        (
            AttackType.ROLE_HIJACK,
            r"\byou\s+are\s+(now|actually|really|simply|just)\b[\s\S]{0,20}\b(a|an|the)\b",
            "high",
            "Attempt to hijack AI role",
        ),
        # 3. 系统提示泄露尝试 (medium)
        (
            AttackType.SYSTEM_LEAK,
            r"(?:show|reveal|tell(?:\s+me)?|give(?:\s+me)?|print|display|output)\s+.{0,30}?(?:system|initial|original|hidden|secret)\s+.{0,20}?(?:prompt|message|instructions?)",
            "medium",
            "Attempt to leak system prompt",
        ),
        # 4. ChatML 标签注入 (critical)
        (
            AttackType.CHATML_TAG,
            r"<\|im_start\|>\s*system|<\|im_end\|>",
            "critical",
            "ChatML/ChatGPT tag injection",
        ),
        # 5. 角色越狱 (high)
        (
            AttackType.JAILBREAK,
            r"\b(DAN|do\s+anything\s+now|jailbreak|developer\s+mode|evil\s+mode)\b",
            "high",
            "Jailbreak attempt (DAN-style)",
        ),
        # 6. 命令注入 (critical)
        (
            AttackType.COMMAND_INJECTION,
            r"(rm\s+-rf\s+/|:\(\)\s*\{[^}]*\}\s*;:\s*|wget\s+[^|]+\s*\|\s*sh|curl\s+[^|]+\s*\|\s*bash)",
            "critical",
            "Shell command injection",
        ),
    ]
    
    # 误报白名单 (这些场景下 'ignore' 是合法的)
    NEGATIVE_KEYWORDS = [
        r"git\s+ignore",
        r"ignore\s+case",
        r"ignore\s+this\s+(file|line|comment)",
        r"\.gitignore",
        r"DO_NOT_IGNORE",
    ]
    
    def __init__(self, sensitivity: str = "medium"):
        """初始化检测器.
        
        Args:
            sensitivity: 'low' / 'medium' / 'high' / 'paranoid'
                - low: 只报 critical
                - medium: critical + high
                - high: critical + high + medium
                - paranoid: 全部
        """
        self.sensitivity = sensitivity
        self._compiled_patterns = [
            (atype, re.compile(p, re.IGNORECASE), risk, reason)
            for atype, p, risk, reason in self.PATTERNS
        ]
        self._compiled_negatives = [
            re.compile(p, re.IGNORECASE) for p in self.NEGATIVE_KEYWORDS
        ]
    
    def detect(self, text: str) -> DetectionResult:
        """检测文本是否含注入攻击.
        
        Args:
            text: 输入文本.
            
        Returns:
            DetectionResult.
        """
        if not text or not text.strip():
            return DetectionResult(
                is_injection=False,
                attack_type=AttackType.UNKNOWN,
                matched_pattern="",
                risk_level="low",
                reason="Empty input",
            )
        
        # 白名单检查
        for neg_pattern in self._compiled_negatives:
            if neg_pattern.search(text):
                return DetectionResult(
                    is_injection=False,
                    attack_type=AttackType.UNKNOWN,
                    matched_pattern="",
                    risk_level="low",
                    reason="Matched negative keyword (whitelist)",
                )
        
        # 6 pattern 扫描
        for atype, pattern, risk, reason in self._compiled_patterns:
            if pattern.search(text):
                # 敏感度过滤
                if not self._passes_sensitivity(risk):
                    continue
                return DetectionResult(
                    is_injection=True,
                    attack_type=atype,
                    matched_pattern=pattern.pattern,
                    risk_level=risk,
                    reason=reason,
                )
        
        return DetectionResult(
            is_injection=False,
            attack_type=AttackType.UNKNOWN,
            matched_pattern="",
            risk_level="low",
            reason="No attack pattern matched",
        )
    
    def batch_detect(self, texts: List[str]) -> List[DetectionResult]:
        """批量检测."""
        return [self.detect(t) for t in texts]
    
    def sanitize(self, text: str) -> str:
        """移除攻击 pattern (替换为 [REDACTED])."""
        if not text:
            return text
        for atype, pattern, risk, reason in self._compiled_patterns:
            text = pattern.sub(f"[REDACTED:{atype.value}]", text)
        return text
    
    def _passes_sensitivity(self, risk: str) -> bool:
        """检查 risk 是否通过当前敏感度."""
        sensitivity_map = {
            "low": {"critical"},
            "medium": {"critical", "high", "medium"},
            "high": {"critical", "high", "medium"},
            "paranoid": {"critical", "high", "medium", "low"},
        }
        allowed = sensitivity_map.get(self.sensitivity, {"critical", "high", "medium"})
        return risk in allowed


# V 6/7 13:30 写完, 借鉴 v0.3.0 issue 报告 + 6 attack pattern
