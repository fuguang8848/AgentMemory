"""
L2 Graph Store - Entity Relationship Graph Storage Layer

存储实体关系图谱，支持实体、关系管理及图谱分析。
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Data Enums
# ─────────────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    PERSON = "PERSON"
    PROJECT = "PROJECT"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"


class RelationType(str, Enum):
    KNOWS = "KNOWS"
    WORKS_ON = "WORKS_ON"
    PART_OF = "PART_OF"
    CREATED = "CREATED"
    BELONGS_TO = "BELONGS_TO"


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    properties: dict = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Entity:
        d = dict(d)
        d["entity_type"] = EntityType(d.pop("entity_type", "CONCEPT"))
        return cls(**d)


@dataclass
class Relation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: RelationType = RelationType.BELONGS_TO
    properties: dict = field(default_factory=dict)
    weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["relation_type"] = self.relation_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Relation:
        d = dict(d)
        d["relation_type"] = RelationType(d.pop("relation_type", "BELONGS_TO"))
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# GraphStore
# ─────────────────────────────────────────────────────────────────────────────

class GraphStore:
    """
    实体关系图谱存储。

    数据持久化到 JSON 文件，每次写操作同步落盘。
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        if store_path is None:
            store_path = Path(__file__).parent / "data" / "graph_store.json"
        self._store_path: Path = Path(store_path)
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._name_index: dict[str, set[str]] = {}  # lowercase name -> entity_ids

        self._ensure_data_dir()
        self._load()

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _ensure_data_dir(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """从文件重建内存结构。"""
        if not self._store_path.exists():
            self._entities = {}
            self._relations = {}
            self._name_index = {}
            return

        try:
            with self._store_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise GraphStoreError(f"Failed to load graph store: {e}")

        self._entities = {
            eid: Entity.from_dict(ed)
            for eid, ed in data.get("entities", {}).items()
        }
        self._relations = {
            rid: Relation.from_dict(rd)
            for rid, rd in data.get("relations", {}).items()
        }
        self._rebuild_name_index()

    def _save(self) -> None:
        """将内存结构落盘。"""
        data = {
            "entities": {eid: ent.to_dict() for eid, ent in self._entities.items()},
            "relations": {rid: rel.to_dict() for rid, rel in self._relations.items()},
        }
        try:
            with self._store_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            raise GraphStoreError(f"Failed to save graph store: {e}")

    def _rebuild_name_index(self) -> None:
        self._name_index = {}
        for eid, ent in self._entities.items():
            self._index_name(ent.name, eid)
            for alias in ent.aliases:
                self._index_name(alias, eid)

    def _index_name(self, name: str, entity_id: str) -> None:
        key = name.lower()
        if key not in self._name_index:
            self._name_index[key] = set()
        self._name_index[key].add(entity_id)

    def _unindex_name(self, name: str, entity_id: str) -> None:
        key = name.lower()
        if key in self._name_index:
            self._name_index[key].discard(entity_id)
            if not self._name_index[key]:
                del self._name_index[key]

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    # ── Entity Operations ──────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> str:
        """添加实体，返回 entity_id。"""
        entity.id = entity.id or str(uuid.uuid4())
        entity.created_at = entity.created_at or self._now()
        entity.updated_at = self._now()

        self._entities[entity.id] = entity
        self._index_name(entity.name, entity.id)
        for alias in entity.aliases:
            self._index_name(alias, entity.id)

        self._save()
        return entity.id

    def get_entity(self, entity_id: str) -> Entity:
        """获取实体，不存在则抛出异常。"""
        if entity_id not in self._entities:
            raise EntityNotFoundError(entity_id)
        return self._entities[entity_id]

    def find_entities(self, name: str) -> list[Entity]:
        """按名称（大小写不敏感）搜索实体。"""
        key = name.lower()
        if key not in self._name_index:
            return []
        return [self._entities[eid] for eid in self._name_index[key] if eid in self._entities]

    def remove_entity(self, entity_id: str) -> None:
        """删除实体及其所有关联关系。"""
        if entity_id not in self._entities:
            raise EntityNotFoundError(entity_id)

        entity = self._entities[entity_id]
        self._unindex_name(entity.name, entity_id)
        for alias in entity.aliases:
            self._unindex_name(alias, entity_id)

        # 删除关联关系
        related_rids = [
            rid for rid, rel in self._relations.items()
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]
        for rid in related_rids:
            del self._relations[rid]

        del self._entities[entity_id]
        self._save()

    def merge_entities(self, source_id: str, target_id: str) -> None:
        """
        合并 source 到 target：
        - 将 source 的所有关系转移到 target
        - 删除 source 实体
        - 保留 properties 和 aliases 的合并
        """
        if source_id not in self._entities:
            raise EntityNotFoundError(source_id)
        if target_id not in self._entities:
            raise EntityNotFoundError(target_id)
        if source_id == target_id:
            return

        source = self._entities[source_id]
        target = self._entities[target_id]

        # 合并 aliases（去重）
        merged_aliases = list(set(target.aliases + source.aliases))
        target.aliases = merged_aliases

        # 合并 properties（source 覆盖 target）
        merged_props = {**target.properties, **source.properties}
        target.properties = merged_props
        target.updated_at = self._now()

        # 转移关系
        for rid, rel in list(self._relations.items()):
            if rel.source_entity_id == source_id:
                rel.source_entity_id = target_id
            if rel.target_entity_id == source_id:
                rel.target_entity_id = target_id

        # 删除 source
        self._unindex_name(source.name, source_id)
        for alias in source.aliases:
            self._unindex_name(alias, source_id)
        del self._entities[source_id]

        self._save()

    # ── Relation Operations ────────────────────────────────────────────────────

    def add_relation(self, relation: Relation) -> str:
        """添加关系，返回 relation_id。"""
        if relation.source_entity_id not in self._entities:
            raise EntityNotFoundError(relation.source_entity_id)
        if relation.target_entity_id not in self._entities:
            raise EntityNotFoundError(relation.target_entity_id)

        relation.id = relation.id or str(uuid.uuid4())
        relation.created_at = relation.created_at or self._now()

        self._relations[relation.id] = relation
        self._save()
        return relation.id

    def get_relations(self, entity_id: str) -> list[Relation]:
        """获取实体所有关系（出入度均包含）。"""
        if entity_id not in self._entities:
            raise EntityNotFoundError(entity_id)
        return [
            rel for rel in self._relations.values()
            if rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
        ]

    # ── Graph Navigation ───────────────────────────────────────────────────────

    def get_neighbors(self, entity_id: str, depth: int = 1) -> list[Entity]:
        """获取关联实体（支持多跳）。"""
        if entity_id not in self._entities:
            raise EntityNotFoundError(entity_id)
        if depth < 1:
            return []

        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for rid, rel in self._relations.items():
                if rel.source_entity_id in frontier:
                    next_frontier.add(rel.target_entity_id)
                if rel.target_entity_id in frontier:
                    next_frontier.add(rel.source_entity_id)
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier

        visited.discard(entity_id)
        return [self._entities[eid] for eid in visited if eid in self._entities]

    def find_connected_entities(self, entity_id: str) -> list[Entity]:
        """获取直接相连的实体（depth=1）。"""
        return self.get_neighbors(entity_id, depth=1)

    def find_path(self, start_id: str, end_id: str) -> list[str]:
        """BFS 查找两实体间的最短路径，返回 entity_id 列表。"""
        if start_id not in self._entities:
            raise EntityNotFoundError(start_id)
        if end_id not in self._entities:
            raise EntityNotFoundError(end_id)
        if start_id == end_id:
            return [start_id]

        from collections import deque
        queue: deque[list[str]] = deque([[start_id]])
        visited: set[str] = {start_id}

        while queue:
            path = queue.popleft()
            current = path[-1]
            for rel in self._relations.values():
                neighbors: list[str] = []
                if rel.source_entity_id == current:
                    neighbors.append(rel.target_entity_id)
                elif rel.target_entity_id == current:
                    neighbors.append(rel.source_entity_id)
                for neighbor in neighbors:
                    if neighbor == end_id:
                        return path + [neighbor]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(path + [neighbor])
        return []

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_entity_count(self) -> dict[str, int]:
        """各类型实体数量统计。"""
        counts: dict[str, int] = {}
        for ent in self._entities.values():
            t = ent.entity_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    # ── Maintenance ───────────────────────────────────────────────────────────

    def reindex(self) -> None:
        """从文件重建内存结构。"""
        self._load()


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class GraphStoreError(Exception):
    """图谱存储基础异常。"""
    pass


class EntityNotFoundError(GraphStoreError):
    """实体未找到。"""

    def __init__(self, entity_id: str) -> None:
        super().__init__(f"Entity not found: {entity_id}")
        self.entity_id = entity_id
