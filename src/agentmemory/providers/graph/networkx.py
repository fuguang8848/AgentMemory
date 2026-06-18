"""
NetworkX Graph Store
Implements GraphStore ABC
M1 Default Graph Library
"""

from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    """Graph node representation"""
    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Graph edge representation"""
    source: str
    target: str
    relation: str
    properties: dict[str, Any] = field(default_factory=dict)


class GraphStore(ABC):
    """Abstract base class for graph stores"""

    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph"""
        raise NotImplementedError

    @abstractmethod
    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID"""
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        depth: int = 1
    ) -> list[GraphNode]:
        """Get neighboring nodes"""
        raise NotImplementedError

    @abstractmethod
    def query(self, cypher: str, **kwargs) -> list[dict[str, Any]]:
        """Query the graph (if supported)"""
        raise NotImplementedError

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges"""
        raise NotImplementedError

    @abstractmethod
    def delete_edge(self, source: str, target: str, relation: str) -> None:
        """Delete an edge"""
        raise NotImplementedError


class NetworkXGraphStore(GraphStore):
    """
    NetworkX in-memory graph store.
    M1 Default Graph Library (zero-dependency, in-memory).
    """

    def __init__(self, directed: bool = False, **kwargs):
        self.directed = directed
        self.kwargs = kwargs
        self._graph = None

    def _get_graph(self):
        """Lazy load the graph"""
        if self._graph is None:
            import networkx as nx
            if self.directed:
                self._graph = nx.DiGraph()
            else:
                self._graph = nx.Graph()
        return self._graph

    @property
    def graph(self):
        """Get the underlying NetworkX graph"""
        return self._get_graph()

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        g = self._get_graph()
        g.add_node(node.id, label=node.label, **node.properties)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph"""
        g = self._get_graph()
        g.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            **edge.properties
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID"""
        g = self._get_graph()
        if node_id not in g:
            return None

        data = g.nodes[node_id]
        return GraphNode(
            id=node_id,
            label=data.get("label", ""),
            properties={k: v for k, v in data.items() if k != "label"}
        )

    def get_neighbors(
        self,
        node_id: str,
        relation: str | None = None,
        depth: int = 1
    ) -> list[GraphNode]:
        """Get neighboring nodes, optionally filtered by relation and depth"""
        g = self._get_graph()
        if node_id not in g:
            return []

        nodes = []

        if depth == 1:
            # Direct neighbors
            for neighbor in g.neighbors(node_id):
                data = g.nodes[neighbor]
                edge_data = g.edges[node_id, neighbor]

                # Filter by relation if specified
                if relation and edge_data.get("relation") != relation:
                    continue

                nodes.append(GraphNode(
                    id=neighbor,
                    label=data.get("label", ""),
                    properties={k: v for k, v in data.items() if k != "label"}
                ))
        else:
            # Multi-hop neighbors using BFS
            import networkx as nx
            paths = nx.single_source_shortest_path_length(g, node_id, cutoff=depth)
            for target_id, distance in paths.items():
                if distance == 0:
                    continue
                data = g.nodes[target_id]
                nodes.append(GraphNode(
                    id=target_id,
                    label=data.get("label", ""),
                    properties={k: v for k, v in data.items() if k != "label"}
                ))

        return nodes

    def query(self, cypher: str, **kwargs) -> list[dict[str, Any]]:
        """
        Query the graph using Python NetworkX API.
        Note: Full Cypher support requires a Cypher query engine.
        This provides a simple Pythonic interface.
        """
        # For now, we don't support raw Cypher without additional libraries
        # Return empty list with a warning
        import warnings
        warnings.warn(
            "NetworkX does not natively support Cypher. "
            "Use built-in methods (get_neighbors, etc.) instead."
        )
        return []

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges"""
        g = self._get_graph()
        if node_id in g:
            g.remove_node(node_id)

    def delete_edge(self, source: str, target: str, relation: str) -> None:
        """Delete an edge"""
        g = self._get_graph()
        if g.has_edge(source, target):
            # Check if relation matches (if specified)
            edge_data = g.edges[source, target]
            if relation and edge_data.get("relation") != relation:
                return
            g.remove_edge(source, target)
