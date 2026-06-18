"""
图书馆分类 (借鉴 v0.3.0 + 适配 v1.0.0)

V 6/7 13:21 fix: 借鉴 v0.3.0 LibraryClassifier 完整类, 补齐 v1.0.0 缺分类的问题.
- v0.3.0 source: src/agent_memory/library.py (256 行, 7 方法)
- v1.0.0 之前 0 引用 LibraryClassifier, 分类责任缺
- 借鉴: 5 顶层 / 4 层深度, 关键词字典 + tokenize
- 适配: 简化 256 → 150 行, 适配 v1.0.0 路径

SOP #16 6 步:
1. diff 看 ✓
2. AST 语法 ✓
3. 备份 /tmp/library.py.v0.3.0.bak
4. msg 含 SOP 引用 ✓
5. log 验证
6. 推 origin (N/A, 非 git)
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Set, Tuple


# 图书馆顶层分类 (5 个)
DEFAULT_CATEGORIES = {
    "Project": ["项目", "project", "工程", "engineering", "石榴籽", "shiliuzi"],
    "Knowledge": ["知识", "knowledge", "笔记", "notes", "笔记", "文章", "article"],
    "Task": ["任务", "task", "todo", "待办", "完成", "done"],
    "People": ["人物", "people", "同事", "coworker", "浮光", "fuguang"],
    "Tools": ["工具", "tools", "脚本", "script", "skill", "技能"],
}


class LibraryError(Exception):
    """图书馆分类错误基类."""
    pass


class CategoryNotFoundError(LibraryError):
    """分类不存在错误."""
    pass


class CategoryDepthExceededError(LibraryError):
    """分类深度超限错误."""
    pass


class LibraryClassifier:
    """图书馆分类器: 5 顶层 / 4 层深度 (借鉴 v0.3.0).
    
    责任:
    1. 分类 (classify) - 把内容归到 5 顶层 / 4 层
    2. 关键词管理 (add_keyword / get_categories / get_all_paths)
    3. 路径验证 (_validate_path)
    """
    
    MAX_DEPTH = 4  # 最大分类深度
    
    def __init__(
        self,
        max_depth: int = MAX_DEPTH,
        dictionary: Optional[Dict[str, List[str]]] = None,
    ):
        """初始化分类器.
        
        Args:
            max_depth: 最大分类深度.
            dictionary: 自定义分类字典, 默认 5 顶层.
        """
        self.max_depth = max_depth
        self.dictionary: Dict[str, List[str]] = dictionary or {
            k: list(v) for k, v in DEFAULT_CATEGORIES.items()
        }
    
    def _tokenize(self, text: str) -> Set[str]:
        """分词 (借鉴 v0.3.0)."""
        # 简单分词: 中英文都支持, 拆出连续 2+ 字符
        tokens = set()
        # 英文 + 数字
        for m in re.finditer(r"[A-Za-z0-9_-]{2,}", text):
            tokens.add(m.group().lower())
        # 中文 2 字组合
        for m in re.finditer(r"[\u4e00-\u9fff]{2,}", text):
            tokens.add(m.group())
        return tokens
    
    def _validate_path(self, path: str) -> str:
        """验证分类路径.
        
        Args:
            path: 分类路径, e.g. "Project/Shiliuzi/Corpus".
            
        Returns:
            标准化后的路径.
            
        Raises:
            CategoryDepthExceededError: 深度超 max_depth.
            CategoryNotFoundError: 顶层分类不在字典里.
        """
        parts = [p.strip() for p in path.split("/") if p.strip()]
        if not parts:
            raise LibraryError(f"Empty path: {path!r}")
        if len(parts) > self.max_depth:
            raise CategoryDepthExceededError(
                f"Path depth {len(parts)} > max_depth {self.max_depth}: {path!r}"
            )
        if parts[0] not in self.dictionary:
            raise CategoryNotFoundError(
                f"Top category {parts[0]!r} not in dictionary. "
                f"Available: {list(self.dictionary.keys())}"
            )
        return "/".join(parts)
    
    def classify(self, content: str) -> str:
        """分类内容到顶层 (借鉴 v0.3.0, 简化版只返顶层).
        
        Args:
            content: 文本内容.
            
        Returns:
            顶层分类 (5 选 1), 找不到返 "Knowledge" (默认).
        """
        tokens = self._tokenize(content)
        for top_cat, keywords in self.dictionary.items():
            for kw in keywords:
                if kw.lower() in tokens or kw in content:
                    return top_cat
        return "Knowledge"  # 默认分类
    
    def add_keyword(self, category: str, keyword: str) -> None:
        """加关键词到分类."""
        if category not in self.dictionary:
            raise CategoryNotFoundError(
                f"Category {category!r} not in dictionary. "
                f"Add to DEFAULT_CATEGORIES first or pass custom dictionary."
            )
        if keyword not in self.dictionary[category]:
            self.dictionary[category].append(keyword)
    
    def get_categories(self) -> List[str]:
        """返所有顶层分类."""
        return list(self.dictionary.keys())
    
    def get_all_paths(self) -> List[str]:
        """返所有可能的分类路径 (5 顶层, 不展开)."""
        return list(self.dictionary.keys())


# V 6/7 13:21 借鉴完成, v1.0.0 路径适配
