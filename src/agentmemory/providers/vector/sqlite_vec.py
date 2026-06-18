"""
SQLite-Vec Vector Store (Incremental Upsert)
Implements VectorStore ABC
M1 Default Vector Library
"""

import sqlite3
import json
import uuid
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VectorSearchResult:
    """Vector search result"""
    id: str
    vector: list[float]
    score: float
    payload: dict[str, Any] | None = None


class VectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def upsert(self, vectors: list[tuple[str, list[float], dict[str, Any]]]) -> list[str]:
        """Insert or update vectors with incremental upsert"""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        **kwargs
    ) -> list[VectorSearchResult]:
        """Search for nearest neighbors"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by IDs"""
        raise NotImplementedError

    @abstractmethod
    def get(self, ids: list[str]) -> list[VectorSearchResult | None]:
        """Get vectors by IDs"""
        raise NotImplementedError


class SQLiteVecStore(VectorStore):
    """
    SQLite-Vec Vector Store with incremental upsert support.
    Uses sqlite-vec extension for efficient vector storage.
    M1 Default Vector Library (zero-dependency fallback).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS vectors (
        id TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        payload TEXT,
        created_at INTEGER DEFAULT (unixepoch()),
        updated_at INTEGER DEFAULT (unixepoch())
    );
    CREATE INDEX IF NOT EXISTS idx_vectors_updated ON vectors(updated_at);
    """

    def __init__(self, db_path: str = ":memory:", table: str = "vectors", **kwargs):
        self.db_path = db_path
        self.table = table
        self.kwargs = kwargs
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema"""
        conn = self._get_conn()
        conn.executescript(self.SCHEMA)
        conn.commit()

    def _vector_to_bytes(self, vector: list[float]) -> bytes:
        """Convert vector to bytes for storage"""
        return json.dumps(vector).encode()

    def _bytes_to_vector(self, data: bytes) -> list[float]:
        """Convert bytes back to vector"""
        return json.loads(data.decode())

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def upsert(self, vectors: list[tuple[str, list[float], dict[str, Any]]]) -> list[str]:
        """
        Insert or update vectors with incremental upsert.
        Each tuple: (id, vector, payload)
        Returns list of IDs.
        """
        conn = self._get_conn()
        ids = []

        for id_val, vector, payload in vectors:
            if not id_val:
                id_val = str(uuid.uuid4())
            ids.append(id_val)

            payload_json = json.dumps(payload) if payload else None
            vector_bytes = self._vector_to_bytes(vector)

            # Upsert: insert or replace on conflict
            conn.execute(
                f"""INSERT INTO {self.table} (id, vector, payload, updated_at)
                    VALUES (?, ?, ?, unixepoch())
                    ON CONFLICT(id) DO UPDATE SET
                        vector = excluded.vector,
                        payload = excluded.payload,
                        updated_at = excluded.updated_at""",
                (id_val, vector_bytes, payload_json)
            )

        conn.commit()
        return ids

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        **kwargs
    ) -> list[VectorSearchResult]:
        """
        Search for nearest neighbors using cosine similarity.
        Falls back to full scan if sqlite-vec not available.
        """
        conn = self._get_conn()

        # Try to use sqlite-vec if available (faster)
        try:
            cursor = conn.execute(
                f"""SELECT id, vector, payload, 
                    vec_distance_cosine(vector, ?) as score
                    FROM {self.table}
                    ORDER BY score
                    LIMIT ?""",
                (self._vector_to_bytes(query_vector), k)
            )
            rows = cursor.fetchall()
        except Exception:
            # Fallback: manual cosine similarity computation
            cursor = conn.execute(f"SELECT id, vector, payload FROM {self.table}")
            rows = cursor.fetchall()

            # Compute similarities manually
            scored = []
            for row in rows:
                vector = self._bytes_to_vector(row["vector"])
                score = self._cosine_similarity(query_vector, vector)
                scored.append((row["id"], row["vector"], row["payload"], score))

            # Sort by score descending and take top k
            scored.sort(key=lambda x: x[3], reverse=True)
            rows = scored[:k]
            return [
                VectorSearchResult(
                    id=row[0],
                    vector=self._bytes_to_vector(row[1]) if isinstance(row[1], bytes) else row[1],
                    score=row[3],
                    payload=json.loads(row[2]) if row[2] else None
                )
                for row in rows
            ]

        results = []
        for row in rows:
            results.append(VectorSearchResult(
                id=row["id"],
                vector=self._bytes_to_vector(row["vector"]),
                score=row["score"] if "score" in row.keys() else 0.0,
                payload=json.loads(row["payload"]) if row["payload"] else None
            ))

        return results[:k]

    def delete(self, ids: list[str]) -> None:
        """Delete vectors by IDs"""
        if not ids:
            return
        conn = self._get_conn()
        placeholders = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM {self.table} WHERE id IN ({placeholders})", ids)
        conn.commit()

    def get(self, ids: list[str]) -> list[VectorSearchResult | None]:
        """Get vectors by IDs"""
        if not ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(ids))
        cursor = conn.execute(
            f"SELECT id, vector, payload FROM {self.table} WHERE id IN ({placeholders})",
            ids
        )
        rows = cursor.fetchall()

        # Maintain order of requested IDs
        results = [None] * len(ids)
        for row in rows:
            idx = ids.index(row["id"])
            results[idx] = VectorSearchResult(
                id=row["id"],
                vector=self._bytes_to_vector(row["vector"]),
                score=1.0,  # Exact match
                payload=json.loads(row["payload"]) if row["payload"] else None
            )
        return results
