"""Pipeline module - data processing pipelines for AgentMemory 2.0.

Pipelines:
- IngestPipeline: dedupe -> PII redact -> chunk
- ExtractPipeline: LLM fact extraction
- EmbedPipeline: batch vectorization with caching
- IndexPipeline: async storage to VectorStore/GraphStore/FileStore
- RetrievePipeline: hybrid retrieval (vector + BM25 + importance)
- DecayPipeline: half-life forgetting with DecayPolicy

References:
    - ARCHITECTURE.md §10 (lines 1498-1614)
"""

from __future__ import annotations

__all__ = [
    "IngestPipeline",
    "ExtractPipeline",
    "EmbedPipeline",
    "IndexPipeline",
    "RetrievePipeline",
    "DecayPipeline",
]

from .ingest import IngestPipeline
from .extract import ExtractPipeline
from .embed import EmbedPipeline
from .index import IndexPipeline
from .retrieve import RetrievePipeline
from .decay import DecayPipeline
