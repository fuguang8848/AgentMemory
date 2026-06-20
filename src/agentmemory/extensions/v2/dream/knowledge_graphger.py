"""
知识图谱 — KnowledgeGrapher
=============================

参考：anda-brain (ldclabs) 神经符号记忆架构

核心思想：
- 记忆不是孤立的向量点，而是关系网络中的节点
- 支持遍历、合并、矛盾检测、时间线追踪
- LLM 通过自然语言与图谱交互，无需学习图查询语言

节点类型：
- CONCEPT: 概念（人物、项目、技术）
- FACT: 事实（决策、事件）
- PROCEDURE: 程序（工作流、步骤）
- EPISODE: 情景（故事、时间线）

边类型：
- CAUSED_BY / RESULT_OF: 因果关系
- BEFORE / AFTER: 时间顺序
- RELATED_TO: 一般关联
- SUPERSEDES: 替代关系（重要：用于检测矛盾）
- PART_OF: 所属关系
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid
import json
from pathlib import Path


class NodeType(Enum):
    CONCEPT = "concept"
    FACT = "fact"
    PROCEDURE = "procedure"
    EPISODE = "episode"


class EdgeType(Enum):
    CAUSED_BY = "caused_by"
    RESULT_OF = "result_of"
    BEFORE = "before"
    AFTER = "after"
    RELATED_TO = "related_to"
    SUPERSEDES = "supersedes"      # 替代（矛盾检测）
    PART_OF = "part_of"
    LINKED_TO = "linked_to"


@dataclass
class MemoryNode:
    """记忆节点"""
    id: str
    node_type: NodeType
    label: str                    # 自然语言标签
    content: str                  # 原始内容摘要
    properties: dict = field(default_factory=dict)  # 额外属性：置信度、来源、时间等
    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        if isinstance(self.node_type, str):
            self.node_type = NodeType(self.node_type)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "label": self.label,
            "content": self.content,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }


@dataclass
class MemoryEdge:
    """记忆边（关系）"""
    id: str
    source: str                   # 源节点ID
    target: str                   # 目标节点ID
    edge_type: EdgeType
    weight: float = 1.0           # 关系强度 0.0~1.0
    properties: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if isinstance(self.edge_type, str):
            self.edge_type = EdgeType(self.edge_type)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at,
        }


class KnowledgeGrapher:
    """
    知识图谱管理器

    使用示例：
        kg = KnowledgeGrapher("~/.openclaw/workspace/memory")
        kg.load()

        # 添加记忆节点
        node = kg.add_node(
            node_type=NodeType.FACT,
            label="项目X决定使用微服务架构",
            content="2026年6月1日，决定...",
            properties={"project": "X", "decision": "microservice"}
        )

        # 建立关系
        kg.add_edge(source=node_a.id, target=node_b.id, edge_type=EdgeType.RESULT_OF)

        # 查询相关节点
        results = kg.query(label_contains="微服务")
    """

    def __init__(self, storage_dir: str = "~/.openclaw/workspace/memory/graph"):
        self.storage_dir = Path(storage_dir).expanduser()
        self.nodes_file = self.storage_dir / "nodes.json"
        self.edges_file = self.storage_dir / "edges.json"
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: list[MemoryEdge] = []

    def load(self):
        """从磁盘加载图谱"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if self.nodes_file.exists():
            with open(self.nodes_file) as f:
                raw_nodes = json.load(f)
                self._nodes = {
                    k: MemoryNode(**v) for k, v in raw_nodes.items()
                }
        if self.edges_file.exists():
            with open(self.edges_file) as f:
                raw_edges = json.load(f)
                self._edges = [MemoryEdge(**e) for e in raw_edges]

    def save(self):
        """持久化到磁盘"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.nodes_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._nodes.items()}, f, indent=2, ensure_ascii=False)
        with open(self.edges_file, "w") as f:
            json.dump([e.to_dict() for e in self._edges], f, indent=2, ensure_ascii=False)

    def add_node(self, node_type: NodeType, label: str, content: str = "", properties: dict = None) -> MemoryNode:
        """添加节点"""
        node = MemoryNode(
            id=f"node_{uuid.uuid4().hex[:12]}",
            node_type=node_type,
            label=label,
            content=content,
            properties=properties or {},
        )
        self._nodes[node.id] = node
        return node

    def add_edge(self, source: str, target: str, edge_type: EdgeType, weight: float = 1.0, properties: dict = None) -> Optional[MemoryEdge]:
        """添加边（双向）"""
        if source not in self._nodes or target not in self._nodes:
            return None
        edge = MemoryEdge(
            id=f"edge_{uuid.uuid4().hex[:12]}",
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            properties=properties or {},
        )
        self._edges.append(edge)
        return edge

    def query(self, label_contains: str = "", node_type: NodeType = None, limit: int = 20) -> list[MemoryNode]:
        """模糊查询节点"""
        results = []
        for node in self._nodes.values():
            if label_contains and label_contains.lower() not in node.label.lower():
                continue
            if node_type and node.node_type != node_type:
                continue
            results.append(node)
            if len(results) >= limit:
                break
        return results

    def find_contradictions(self) -> list[tuple[MemoryNode, MemoryNode]]:
        """
        查找矛盾记忆

        通过 SUPERSEDES 边找到被替代的记忆对
        例如："我是素食主义者" 被 "我现在吃肉了" 替代
        """
        contradictions = []
        superseding = {}

        for edge in self._edges:
            if edge.edge_type == EdgeType.SUPERSEDES:
                superseding[edge.target] = edge.source

        for node_id, newer_id in superseding.items():
            if node_id in self._nodes and newer_id in self._nodes:
                contradictions.append((self._nodes[node_id], self._nodes[newer_id]))

        return contradictions

    def get_reachability(self, node_id: str, max_depth: int = 3) -> list[str]:
        """从某节点出发，可达的所有节点（带深度限制）"""
        visited = set()
        frontier = [(node_id, 0)]
        result = []

        while frontier:
            current, depth = frontier.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            if current != node_id:
                result.append(current)
            for edge in self._edges:
                if edge.source == current and edge.target not in visited:
                    frontier.append((edge.target, depth + 1))
                if edge.target == current and edge.source not in visited:
                    frontier.append((edge.source, depth + 1))

        return result

    def merge_nodes(self, keep_id: str, merge_ids: list[str]) -> MemoryNode:
        """
        合并多个节点（语义去重）
        参考 openclaw-auto-dream 的语义去重
        """
        keep = self._nodes[keep_id]
        for mid in merge_ids:
            if mid not in self._nodes:
                continue
            node = self._nodes[mid]
            # 合并属性（取较新时间戳）
            if node.updated_at > keep.updated_at:
                keep.updated_at = node.updated_at
                keep.version += 1
            # 更新标签（拼接）
            if node.label != keep.label:
                keep.label = f"{keep.label} | {node.label}"
            # 删除被合并节点
            del self._nodes[mid]
            # 删除相关边（重连到保留节点）
            self._edges = [
                e for e in self._edges
                if e.source != mid and e.target != mid
            ]
        return keep

    def stats(self) -> dict:
        """图谱统计"""
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "by_type": {
                nt.value: sum(1 for n in self._nodes.values() if n.node_type == nt)
                for nt in NodeType
            },
            "contradictions": len(self.find_contradictions()),
        }
