"""
FAISS 优化后端 — 梁文峰工程极限视角

问题: IndexFlatIP 是 O(N) 暴力搜索，10万记忆 = 每查询10万次内积
解决: 用 IVF 聚类索引，O(N/M * log(M)) 近似最优

JEPA 对比式视角:
- 不是生成向量(生成式)
- 而是学习"哪些向量属于同一簇"(对比式聚类)
"""

from __future__ import annotations

import threading
from typing import Any

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSOptimizedBackend:
    """
    优化版 FAISS 后端 — 用 IVF 聚类替代 Flat

    梁文峰工程极限: 用有限内存做最强性能

    优化策略:
    1. IVF 倒排文件索引: 将向量聚类到 M 个桶
       - 查询时只搜索最近的 nprobe 桶
       - 时间复杂度: O(N/M * nprobe + k*log(M))
       - vs Flat: O(N)

    2. PQ 向量量化(可选): 将 384维 → 96维
       - 内存减少 4x
       - 精度损失 < 5%

    3. HNSW 图索引(备选): 构建分层可导航小世界图
       - 查询 O(log N)
       - 内存 2-3x
    """

    # 梁文峰工程极限参数
    DEFAULT_NLIST = 100          # 聚类中心数 (经验: N/1000 到 N/100)
    DEFAULT_NPROBE = 10          # 查询时探测的桶数 (nlist 的 10%)
    DEFAULT_M = 32               # HNSW M参数 (每层边数)
    DEFAULT_EF_CONSTRUCTION = 40 # HNSW 建设参数

    def __init__(
        self,
        dimension: int = 384,
        index_path: str = None,
        nlist: int = None,
        nprobe: int = None,
        use_pq: bool = False,
        pq_m: int = 32,
        use_hnsw: bool = False,
        **kwargs
    ):
        """
        初始化优化 FAISS 索引

        Args:
            dimension: 向量维度
            index_path: 持久化路径
            nlist: IVF 聚类数 (默认 N/1000)
            nprobe: 查询探测数 (默认 nlist 的 10%)
            use_pq: 是否使用 PQ 量化
            pq_m: PQ 分区数
            use_hnsw: 是否使用 HNSW (替代 IVF)
        """
        self.dimension = dimension
        self.index_path = index_path
        self.nlist = nlist or self.DEFAULT_NLIST
        self.nprobe = nprobe or self.DEFAULT_NPROBE
        self.use_pq = use_pq
        self.pq_m = pq_m
        self.use_hnsw = use_hnsw
        self.kwargs = kwargs

        self._index = None
        self._id_map: dict[str, int] = {}
        self._reverse_map: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._next_idx = 0
        self._lock = threading.Lock()
        self._total_vectors = 0

        self._lazy_init()

    def _lazy_init(self) -> None:
        """延迟初始化 FAISS 索引"""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")

        if self.use_hnsw:
            # HNSW 图索引
            self._index = faiss.IndexHNSWFlat(self.dimension, self.DEFAULT_M)
            self._index.hnsw.efConstruction = self.DEFAULT_EF_CONSTRUCTION
        elif self.use_pq:
            # PQ 量化索引
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.nlist,
                self.pq_m,  # m: 每个向量被分成 m 个子向量
                8           # nbits: 每个子向量用 8 bits 编码
            )
        else:
            # IVF 聚类索引 (不用 PQ)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVF(
                quantizer,
                self.dimension,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self._index.nprobe = self.nprobe

    def _normalize(self, vectors) -> Any:
        """L2 归一化"""
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vectors / norm

    def add(self, texts: list[str], embeddings: list[list[float]], metadata: list[dict[str, Any]]) -> list[str]:
        """
        添加向量到索引

        梁文峰工程极限: 批量添加减少 API 调用
        """
        if not FAISS_AVAILABLE:
            return []

        self._lazy_init()

        with self._lock:
            ids = []
            embeddings = np.array(embeddings, dtype='float32')
            normalized = self._normalize(embeddings)

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                vector_id = metadata[i].get("id", f"faiss_{self._next_idx}")
                ids.append(vector_id)

                self._id_map[vector_id] = self._next_idx
                self._reverse_map[self._next_idx] = vector_id
                self._metadata[vector_id] = {
                    "text": text,
                    **metadata[i]
                }
                self._next_idx += 1

            # 批量添加
            if len(normalized) > 0:
                self._index.add(normalized)
                self._total_vectors += len(normalized)

            return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        tenant_id: str = None,
        namespace: str = None,
        filter_metadata: dict[str, Any] = None,
    ) -> list[dict]:
        """搜索最近邻"""
        self._lazy_init()

        with self._lock:
            if self._index.ntotal == 0:
                return []

            query_np = np.array(query_embedding, dtype='float32').reshape(1, -1)
            query_np = self._normalize(query_np)

            # 搜索
            search_k = min(k * 10, self._index.ntotal)
            scores, indices = self._index.search(query_np, search_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    break
                vector_id = self._reverse_map.get(int(idx))
                if not vector_id:
                    continue
                meta = self._metadata.get(vector_id, {})

                # 应用过滤
                if tenant_id and meta.get("tenant_id") != tenant_id:
                    continue
                if namespace and meta.get("namespace") != namespace:
                    continue
                if filter_metadata:
                    skip = False
                    for fk, fv in filter_metadata.items():
                        if meta.get(fk) != fv:
                            skip = True
                            break
                    if skip:
                        continue

                results.append({
                    "id": vector_id,
                    "score": float(score),
                    "text": meta.get("text", ""),
                    "metadata": meta
                })

                if len(results) >= k:
                    break

            return results

    def set_nprobe(self, nprobe: int) -> None:
        """
        动态调整 nprobe — 梁文峰工程极限

        nprobe↑ = 更高精度 + 更慢查询
        nprobe↓ = 更低精度 + 更快查询

        运行时可根据负载动态调整
        """
        if hasattr(self._index, 'nprobe'):
            self._index.nprobe = nprobe

    def save(self, path: str = None) -> None:
        """保存索引到磁盘"""
        if self._index is None:
            return
        path = path or self.index_path
        if path:
            faiss.write_index(self._index, path)

    def load(self, path: str = None) -> None:
        """从磁盘加载索引"""
        if not FAISS_AVAILABLE:
            return
        path = path or self.index_path
        if path:
            self._index = faiss.read_index(path)

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计"""
        return {
            "backend": "faiss_optimized",
            "total_vectors": self._total_vectors,
            "dimension": self.dimension,
            "nlist": self.nlist if not self.use_hnsw else "N/A (HNSW)",
            "nprobe": self.nprobe if not self.use_hnsw else "N/A (HNSW)",
            "index_type": "HNSW" if self.use_hnsw else ("PQ" if self.use_pq else "IVF"),
            "memory_estimate_mb": self._estimate_memory(),
        }

    def _estimate_memory(self) -> float:
        """估算内存使用(MB)"""
        if self._total_vectors == 0:
            return 0.0

        if self.use_hnsw:
            # HNSW: 每个向量 + 边
            return self._total_vectors * self.dimension * 4 / 1024 / 1024 * 2
        elif self.use_pq:
            # PQ: 量化后的向量
            return self._total_vectors * self.pq_m / 8 / 1024 / 1024
        else:
            # IVF: 向量 + 聚类中心
            return self._total_vectors * self.dimension * 4 / 1024 / 1024


class FAISSIVFTuner:
    """
    FAISS IVF 参数调优器 — 梁文峰工程极限

    自动找到 nlist/nprobe 的最优组合
    在精度和速度之间找平衡
    """

    def __init__(self, index: FAISSOptimizedBackend):
        self.index = index

    def tune(self, queries: list[tuple[list[float], list[str]]], target_recall: float = 0.95) -> dict:
        """
        调优 nprobe 参数

        Args:
            queries: [(query_embedding, expected_relevant_ids)]
            target_recall: 目标召回率

        Returns:
            {"nprobe": N, "qps": speed, "recall": actual_recall}
        """
        import time

        best_nprobe = 1
        best_score = 0.0

        for nprobe in [1, 5, 10, 20, 50]:
            self.index.set_nprobe(nprobe)

            recalls = []
            start = time.time()
            for query_emb, expected_ids in queries:
                results = self.index.search(query_emb, k=20)
                result_ids = {r["id"] for r in results}
                recall = len(result_ids & set(expected_ids)) / max(len(expected_ids), 1)
                recalls.append(recall)
            elapsed = time.time() - start

            avg_recall = sum(recalls) / len(recalls)
            qps = len(queries) / elapsed

            score = avg_recall / elapsed  # recall per second

            if avg_recall >= target_recall and score > best_score:
                best_nprobe = nprobe
                best_score = score

        return {
            "nprobe": best_nprobe,
            "estimated_qps": best_score,
            "target_recall": target_recall,
        }
