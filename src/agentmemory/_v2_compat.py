"""Compatibility shim: expose extensions/v2 modules at agentmemory root level.

This allows older code that imports `from agentmemory.memory_manager import ...`
to work with the v2 architecture where these modules live in
`agentmemory.extensions.v2.memory_manager`.
"""
import importlib
import sys

_V2_MODULES = [
    "memory_manager",
    "search.search_engine",
    "search.hybrid_retriever",
    "search.rrf_fusion",
    "search.search_coordinator",
    "bm25",
    "reranker",
    "L3_vector_store",
    "L4_file_persist",
    "decay_engine",
    "injection",
    "library",
]

def install():
    for mod_name in _V2_MODULES:
        try:
            full_name = f"agentmemory.extensions.v2.{mod_name}"
            mod = importlib.import_module(full_name)
            # Also expose at agentmemory.<mod_name>
            sys.modules[f"agentmemory.{mod_name}"] = mod
        except ImportError:
            pass

install()
