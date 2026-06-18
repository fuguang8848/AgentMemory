"""
SQLite WAL Storage (meta + audit)
Implements Storage ABC
M1 Default Storage
"""

import sqlite3
import json
import time
from typing import Any, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StorageRecord:
    """Storage record representation"""
    key: str
    value: Any
    metadata: dict[str, Any] | None = None
    created_at: float | None = None
    updated_at: float | None = None


class Storage(ABC):
    """Abstract base class for storage backends"""

    @abstractmethod
    def set(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Store a key-value pair with optional metadata"""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> StorageRecord | None:
        """Retrieve a record by key"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a record by key"""
        raise NotImplementedError

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix"""
        raise NotImplementedError

    @abstractmethod
    def iter_records(self, prefix: str = "") -> Iterator[StorageRecord]:
        """Iterate over records"""
        raise NotImplementedError


class SQLiteStorage(Storage):
    """
    SQLite WAL Storage with metadata and audit support.
    M1 Default Storage (zero-dependency, ACID compliant).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS storage (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        metadata TEXT,
        created_at REAL DEFAULT (unixepoch()),
        updated_at REAL DEFAULT (unixepoch())
    );

    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL,
        action TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        timestamp REAL DEFAULT (unixepoch())
    );

    CREATE INDEX IF NOT EXISTS idx_audit_key ON audit_log(key);
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_storage_key ON storage(key);
    """

    def __init__(self, db_path: str = "agentmemory.db", audit: bool = True, **kwargs):
        self.db_path = db_path
        self.audit = audit
        self.kwargs = kwargs
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency and durability
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema"""
        conn = self._get_conn()
        conn.executescript(self.SCHEMA)
        conn.commit()

    def set(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """Store a key-value pair with optional metadata"""
        conn = self._get_conn()

        # Serialize value
        if isinstance(value, (dict, list)):
            value_json = json.dumps(value)
        else:
            value_json = json.dumps({"_raw": value})

        metadata_json = json.dumps(metadata) if metadata else None
        now = time.time()

        # Get old value for audit
        old_value = None
        if self.audit:
            cursor = conn.execute("SELECT value FROM storage WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                old_value = row["value"]

        # Upsert
        conn.execute(
            """INSERT INTO storage (key, value, metadata, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   metadata = excluded.metadata,
                   updated_at = ?""",
            (key, value_json, metadata_json, now, now)
        )

        # Audit log
        if self.audit:
            conn.execute(
                """INSERT INTO audit_log (key, action, old_value, new_value)
                   VALUES (?, ?, ?, ?)""",
                (key, "UPDATE" if old_value else "INSERT", old_value, value_json)
            )

        conn.commit()

    def get(self, key: str) -> StorageRecord | None:
        """Retrieve a record by key"""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT key, value, metadata, created_at, updated_at FROM storage WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize value
        try:
            value = json.loads(row["value"])
            if "_raw" in value:
                value = value["_raw"]
        except json.JSONDecodeError:
            value = row["value"]

        # Deserialize metadata
        metadata = None
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                metadata = {"raw": row["metadata"]}

        return StorageRecord(
            key=row["key"],
            value=value,
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def delete(self, key: str) -> None:
        """Delete a record by key"""
        conn = self._get_conn()

        # Get old value for audit
        old_value = None
        if self.audit:
            cursor = conn.execute("SELECT value FROM storage WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                old_value = row["value"]

        conn.execute("DELETE FROM storage WHERE key = ?", (key,))

        # Audit log
        if self.audit and old_value:
            conn.execute(
                "INSERT INTO audit_log (key, action, old_value) VALUES (?, ?, ?)",
                (key, "DELETE", old_value)
            )

        conn.commit()

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix"""
        conn = self._get_conn()
        if prefix:
            cursor = conn.execute(
                "SELECT key FROM storage WHERE key LIKE ? ORDER BY key",
                (f"{prefix}%",)
            )
        else:
            cursor = conn.execute("SELECT key FROM storage ORDER BY key")

        return [row["key"] for row in cursor.fetchall()]

    def iter_records(self, prefix: str = "") -> Iterator[StorageRecord]:
        """Iterate over records"""
        keys = self.list_keys(prefix)
        for key in keys:
            record = self.get(key)
            if record:
                yield record

    def get_audit_log(self, key: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get audit log entries, optionally filtered by key"""
        conn = self._get_conn()
        if key:
            cursor = conn.execute(
                """SELECT * FROM audit_log WHERE key = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (key, limit)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )

        return [dict(row) for row in cursor.fetchall()]
