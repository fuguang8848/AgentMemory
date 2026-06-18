"""GraphStore ABC.

References:
    - ARCHITECTURE.md §5.3.5 (GraphStore ABC)
"""

from __future__ import annotations

__all__ = ["GraphStore", "GraphNode", "GraphEdge"]

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Graph node representation."""
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphEdge(BaseModel):
    """Graph edge representation."""
    id: str
    source_id: str
    target_id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphStore(ABC):
    """Abstract base class for graph storage backends.

    All graph store implementations (Neo4j, NetworkX, etc.)
    must inherit from this class.
    """

    @abstractmethod
    async def add_node(self, node: GraphNode) -> str:
        """Add a node. Returns the node ID."""
        ...

    @abstractmethod
    async def add_edge(self, edge: GraphEdge) -> str:
        """Add an edge. Returns the edge ID."""
        ...

    @abstractmethod
    async def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        ...

    @abstractmethod
    async def traverse(
        self,
        start_id: str,
        max_hops: int = 2,
        edge_types: list[str] | None = None,
    ) -> list[GraphNode]:
        """Traverse graph from start node."""
        ...

    @abstractmethod
    async def query(self, cypher_like: str, params: dict) -> list[dict]:
        """Query graph with a Cypher-like language."""
        ...

    @abstractmethod
    async def delete(self, node_id: str) -> int:
        """Delete a node and its edges. Returns number of affected edges."""
        ...
