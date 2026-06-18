"""1.x to 2.x migration utilities.

Per ARCHITECTURE.md §6.1 line 891 and §6.2:
    agentmemory-migrate = "agentmemory.compat.migration:main"

Scans old 1.x JSON files (memory_manager.metadata_path, etc.) and converts
them to the new SQLite + FAISS format used by 2.0.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Placeholder imports - in production these would come from the actual providers
# We'll use lazy imports to avoid hard dependencies during migration scanning

__all__ = ["migrate_1x_to_2x", "main"]


def _load_1x_json(file_path: str | Path) -> list[dict[str, Any]]:
    """Load a 1.x JSON memory file."""
    path = Path(file_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "memories" in data:
        return data["memories"]
    return [data]


def _infer_memory_type(entry: dict[str, Any]) -> str:
    """Infer 2.0 MemoryType from 1.x entry."""
    t = entry.get("type", "").lower()
    if t in ("fact", "semantic"):
        return "semantic"
    if t in ("procedure", "procedural"):
        return "procedural"
    if t in ("reflection", "reflective"):
        return "reflective"
    return "user"


def _infer_memory_layer(entry: dict[str, Any]) -> str:
    """Infer 2.0 MemoryLayer from 1.x entry."""
    layer = entry.get("layer", "")
    layer_map = {
        "L0": "L0",
        "L1": "L1",
        "L2": "L2",
        "L3": "L3",
        "L4": "L4",
        "L5": "L5",
        0: "L0",
        1: "L1",
        2: "L2",
        3: "L3",
        4: "L4",
        5: "L5",
    }
    return layer_map.get(layer, "L3")


def _convert_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a single 1.x entry to 2.0 MemoryItem-compatible dict."""
    return {
        "id": entry.get("id") or entry.get("memory_id") or str(hash(entry.get("content", ""))),
        "content": entry.get("content", "") or entry.get("text", ""),
        "type": _infer_memory_type(entry),
        "layer": _infer_memory_layer(entry),
        "importance": entry.get("importance", entry.get("weight", 0.5)),
        "confidence": entry.get("confidence", 1.0),
        "entities": entry.get("entities", entry.get("subjects", [])),
        "tags": entry.get("tags", []),
        "source": entry.get("source", "user"),
        "metadata": entry.get("metadata", {}),
        "created_at": entry.get("created_at", datetime.utcnow().isoformat()),
        "updated_at": entry.get("updated_at", datetime.utcnow().isoformat()),
    }


def _scan_1x_dir(input_dir: str | Path) -> list[dict[str, Any]]:
    """Scan a 1.x memory directory for all JSON files."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    entries = []

    # Known 1.x file names to scan
    candidates = [
        "memories.json",
        "memory_manager.metadata.json",
        "metadata.json",
        "facts.json",
        "entries.json",
        "memory.json",
    ]

    for candidate in candidates:
        file_path = input_path / candidate
        if file_path.exists():
            entries.extend(_load_1x_json(file_path))

    # Also scan all .json files recursively
    for json_file in input_path.rglob("*.json"):
        if json_file.is_file():
            try:
                entries.extend(_load_1x_json(json_file))
            except Exception:
                pass

    return entries


def migrate_1x_to_2x(input_dir: str | Path, output_db: str | Path) -> dict[str, int]:
    """Migrate 1.x JSON memories to 2.x SQLite + FAISS.

    Args:
        input_dir: Directory containing 1.x JSON files
        output_db: Output SQLite database path

    Returns:
        Statistics dict with counts of migrated items
    """
    input_path = Path(input_dir)
    output_path = Path(output_db)

    print(f"Scanning {input_path} for 1.x JSON files...")
    entries = _scan_1x_dir(input_path)
    print(f"Found {len(entries)} entries in 1.x format")

    converted = [_convert_entry(e) for e in entries]

    # Import here to avoid circular dependencies
    try:
        import sqlite3
    except ImportError:
        print("ERROR: sqlite3 not available", file=sys.stderr)
        return {"found": len(entries), "converted": len(converted), "written": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create 2.0 schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'semantic',
            layer TEXT NOT NULL DEFAULT 'L3',
            importance REAL NOT NULL DEFAULT 0.5,
            confidence REAL NOT NULL DEFAULT 1.0,
            entities TEXT NOT NULL DEFAULT '[]',
            tags TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL DEFAULT 'user',
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding BLOB,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_accessed_at TEXT,
            access_count INTEGER NOT NULL DEFAULT 0,
            decay_score REAL,
            tenant_id TEXT NOT NULL DEFAULT 'default',
            namespace TEXT NOT NULL DEFAULT 'default'
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(tenant_id)")

    written = 0
    for item in converted:
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, content, type, layer, importance, confidence, entities, tags,
                 source, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["id"],
                    item["content"],
                    item["type"],
                    item["layer"],
                    item["importance"],
                    item["confidence"],
                    json.dumps(item["entities"]),
                    json.dumps(item["tags"]),
                    item["source"],
                    json.dumps(item["metadata"]),
                    item["created_at"],
                    item["updated_at"],
                ),
            )
            written += 1
        except Exception as e:
            print(f"Warning: failed to insert {item['id']}: {e}", file=sys.stderr)

    conn.commit()
    conn.close()

    stats = {"found": len(entries), "converted": len(converted), "written": written}
    print(f"Migration complete: {written}/{len(entries)} entries written to {output_path}")
    return stats


def main():
    """CLI entry point for the migration tool."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate AgentMemory 1.x data to 2.x")
    parser.add_argument("input_dir", help="Directory containing 1.x JSON files")
    parser.add_argument("output_db", help="Output SQLite database path")
    args = parser.parse_args()

    try:
        stats = migrate_1x_to_2x(args.input_dir, args.output_db)
        print(f"Success! Migrated {stats['written']} entries.")
    except Exception as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
