"""
L3 Vector Store Layer
混合检索层：向量检索 + BM25 混合检索，参考 Mem0 的混合排序
"""

import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    content: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)
    importance: float = 0.5  # 0-1
    access_count: int = 0
    last_accessed: Optional[float] = None  # Unix timestamp
    created_at: float = field(default_factory=lambda: time.time())
    tags: list[str] = field(default_factory=list)
    fact_type: str = "general"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


# ============================================================================
# BM25 Indexer (Classic algorithm, no external library)
# ============================================================================


class BM25Indexer:
    """经典 BM25 算法实现，不依赖外部库"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[str] = []
        self.doc_ids: list[str] = []
        self.avg_doc_len: float = 0.0
        self.doc_lens: list[int] = []
        self.doc_freqs: dict[str, int] = {}  # term -> doc frequency
        self.idf: dict[str, float] = {}
        self.term_index: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc_id, freq)]

    def _tokenize(self, text: str) -> list[str]:
        """简单分词：小写化 + 分割"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def index(self, documents: list[str], doc_ids: Optional[list[str]] = None) -> None:
        """
        构建 BM25 索引
        :param documents: 文档内容列表
        :param doc_ids: 可选的文档ID列表
        """
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]

        # 计算平均文档长度
        tokenized = [self._tokenize(doc) for doc in documents]
        self.doc_lens = [len(tokens) for tokens in tokenized]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 1

        # 构建倒排索引
        self.term_index: dict[str, list[tuple[int, int]]] = {}
        self.doc_freqs = {}

        for doc_idx, tokens in enumerate(tokenized):
            seen = set()
            for term_idx, term in enumerate(tokens):
                if term not in self.term_index:
                    self.term_index[term] = []
                    self.doc_freqs[term] = 0
                self.term_index[term].append((doc_idx, term_idx))
                if term not in seen:
                    self.doc_freqs[term] += 1
                    seen.add(term)

        # 计算 IDF
        n = len(documents)
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # BM25 IDF 公式：log((n - df + 0.5) / (df + 0.5))
            self.idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1e-8)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """
        BM25 搜索
        :param query: 查询字符串
        :param k: 返回前 k 个结果
        :return: [(doc_id, score), ...] 按分数降序
        """
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        scores: dict[int, float] = {}

        for term in query_tokens:
            if term not in self.term_index:
                continue

            idf = self.idf.get(term, 0)
            for doc_idx, _ in self.term_index[term]:
                doc_len = self.doc_lens[doc_idx]
                # BM25 公式
                term_freq = sum(1 for t, _ in self.term_index[term] if t == doc_idx)
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score = idf * numerator / denominator

                if doc_idx not in scores:
                    scores[doc_idx] = 0.0
                scores[doc_idx] += score

        # 按分数排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # 转换为 (doc_id, score) 格式
        result = [(self.doc_ids[idx], score) for idx, score in sorted_scores[:k]]

        return result

    def get_doc_by_idx(self, idx: int) -> Optional[str]:
        """根据索引获取文档"""
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None


# ============================================================================
# Vector Store
# ============================================================================


class VectorStore:
    """向量存储 + 混合检索"""

    def __init__(
        self,
        storage_path: str = "src/data/vectors.json",
        embedding_model: str = "text-embedding-v3",
        embedding_dims: int = 1024,
        embedding_batch_size: int = 16,
    ):
        self.storage_path = storage_path
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims
        self.embedding_batch_size = embedding_batch_size

        # 确保存储目录存在
        os.makedirs(os.path.dirname(storage_path) or ".", exist_ok=True)

        # 加载已有数据
        self.entries: dict[str, MemoryEntry] = {}
        self._load()

        # BM25 索引器
        self.bm25 = BM25Indexer(k1=1.5, b=0.75)
        self._rebuild_bm25()

    def _load(self) -> None:
        """从文件加载数据"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry_data in data.values():
                        entry = MemoryEntry.from_dict(entry_data)
                        self.entries[entry.id] = entry
            except (json.JSONDecodeError, KeyError):
                self.entries = {}

    def _save(self) -> None:
        """保存数据到文件"""
        data = {eid: entry.to_dict() for eid, entry in self.entries.items()}
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _rebuild_bm25(self) -> None:
        """重建 BM25 索引"""
        if not self.entries:
            self.bm25.index([])
            return

        docs = [entry.content for entry in self.entries.values()]
        ids = [entry.id for entry in self.entries.values()]
        self.bm25.index(docs, ids)

    def _embed_single(self, text: str) -> list[float]:
        """
        调用 DashScope embedding API（单条）
        """
        import httpx

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            # 模拟向量（实际使用时替换为真实 API 调用）
            import random
            random.seed(hash(text) % (2**32))
            return [random.random() for _ in range(self.embedding_dims)]

        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_model,
            "input": {"text": text},
            "parameters": {"dimension": self.embedding_dims},
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return embedding

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        批量调用 DashScope embedding API
        提升效率，减少 API 调用次数
        """
        if not texts:
            return []

        import httpx

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            # 模拟向量
            import random
            random.seed(42)
            return [[random.random() for _ in range(self.embedding_dims)] for _ in texts]

        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        all_embeddings: list[list[float]] = []

        # 批量处理
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]
            payload = {
                "model": self.embedding_model,
                "input": {"texts": batch},
                "parameters": {"dimension": self.embedding_dims},
            }

            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                for item in result["data"]:
                    all_embeddings.append(item["embedding"])

        return all_embeddings

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def store(
        self,
        content: str,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        fact_type: str = "general",
    ) -> str:
        """
        存储记忆
        :return: memory_id
        """
        # 生成 ID
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        # 计算 embedding
        vector = self._embed_single(content)

        # 创建条目
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            vector=vector,
            metadata=metadata or {},
            importance=importance,
            tags=tags or [],
            fact_type=fact_type,
        )

        self.entries[memory_id] = entry
        self._save()
        self._rebuild_bm25()

        return memory_id

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        混合检索
        :param query: 查询字符串
        :param limit: 返回数量
        :param filters: 过滤条件 {fact_type, tags, date_range}
        :return: [{"id": ..., "content": ..., "score": 0.xx, "metadata": {...}}, ...]
        """
        if not self.entries:
            return []

        # 获取查询向量
        query_vector = self._embed_single(query)

        # BM25 搜索
        bm25_results = self.bm25.search(query, k=len(self.entries))
        bm25_scores: dict[str, float] = {doc_id: score for doc_id, score in bm25_results}
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0

        # 计算每个条目的混合分数
        candidate_scores: dict[str, float] = {}

        for memory_id, entry in self.entries.items():
            # 应用过滤
            if filters:
                if "fact_type" in filters and entry.fact_type != filters["fact_type"]:
                    continue
                if "tags" in filters:
                    if not any(t in entry.tags for t in filters["tags"]):
                        continue
                if "date_range" in filters:
                    dr = filters["date_range"]
                    if "start" in dr and entry.created_at < dr["start"]:
                        continue
                    if "end" in dr and entry.created_at > dr["end"]:
                        continue

            # 向量相似度
            vector_sim = self._cosine_similarity(query_vector, entry.vector)

            # BM25 归一化分数
            bm25_score = bm25_scores.get(memory_id, 0.0) / max_bm25 if max_bm25 > 0 else 0.0

            # 混合评分：vector * 0.6 + bm25 * 0.3 + importance * 0.1
            hybrid_score = vector_sim * 0.6 + bm25_score * 0.3 + entry.importance * 0.1

            candidate_scores[memory_id] = hybrid_score

        # 排序并返回 top N
        sorted_ids = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for memory_id, score in sorted_ids[:limit]:
            entry = self.entries[memory_id]
            results.append({
                "id": memory_id,
                "content": entry.content,
                "score": round(score, 4),
                "metadata": entry.metadata,
                "tags": entry.tags,
                "fact_type": entry.fact_type,
                "importance": entry.importance,
                "created_at": entry.created_at,
            })

        return results

    def update_importance(self, memory_id: str, importance: float) -> bool:
        """更新记忆重要性分数"""
        if memory_id not in self.entries:
            return False
        self.entries[memory_id].importance = max(0.0, min(1.0, importance))
        self._save()
        return True

    def increment_access(self, memory_id: str) -> bool:
        """增加访问计数"""
        if memory_id not in self.entries:
            return False
        entry = self.entries[memory_id]
        entry.access_count += 1
        entry.last_accessed = time.time()
        self._save()
        return True

    def get_stats(self) -> dict:
        """获取统计信息"""
        entries_list = list(self.entries.values())
        if not entries_list:
            return {
                "total": 0,
                "avg_importance": 0.0,
                "avg_access_count": 0.0,
                "fact_types": {},
                "tags": {},
            }

        import collections

        total = len(entries_list)
        avg_importance = sum(e.importance for e in entries_list) / total
        avg_access_count = sum(e.access_count for e in entries_list) / total

        fact_types = collections.Counter(e.fact_type for e in entries_list)
        all_tags: list[str] = []
        for e in entries_list:
            all_tags.extend(e.tags)
        tags = dict(collections.Counter(all_tags).most_common(20))

        return {
            "total": total,
            "avg_importance": round(avg_importance, 4),
            "avg_access_count": round(avg_access_count, 2),
            "fact_types": dict(fact_types),
            "tags": tags,
        }

    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        if memory_id not in self.entries:
            return False
        del self.entries[memory_id]
        self._save()
        self._rebuild_bm25()
        return True

    def get(self, memory_id: str) -> Optional[dict]:
        """获取单条记忆"""
        if memory_id not in self.entries:
            return None
        entry = self.entries[memory_id]
        return {
            "id": entry.id,
            "content": entry.content,
            "metadata": entry.metadata,
            "importance": entry.importance,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed,
            "created_at": entry.created_at,
            "tags": entry.tags,
            "fact_type": entry.fact_type,
        }


# ============================================================================
# Hybrid Retriever
# ============================================================================


class HybridRetriever:
    """混合检索器，支持过滤和重排"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        storage_path: str = "src/data/vectors.json",
        rerank_enabled: bool = False,
        rerank_model: str = "jina-reranker-v2",
    ):
        self.vector_store = vector_store or VectorStore(storage_path=storage_path)
        self.rerank_enabled = rerank_enabled
        self.rerank_model = rerank_model

    def _simple_rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        简单的重排策略（当没有 rerank API 时使用）
        基于位置衰减和分数调整
        """
        reranked = []
        for i, item in enumerate(results[:top_k * 2]):
            # 位置衰减
            position_boost = 1.0 / (1 + 0.1 * i)
            # 综合分数
            adjusted_score = item["score"] * position_boost

            # 内容相似度增强：如果 query 关键词在 content 中出现多次
            query_terms = set(query.lower().split())
            content_lower = item["content"].lower()
            term_overlap = sum(content_lower.count(t) for t in query_terms)

            adjusted_score += 0.01 * min(term_overlap, 5)

            reranked.append((adjusted_score, item))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in reranked[:top_k]]

    def _rerank_with_api(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        使用 Jina Rerank API 进行重排
        """
        import httpx

        api_key = os.environ.get("JINA_API_KEY", "")
        if not api_key:
            return self._simple_rerank(query, results, top_k)

        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        docs = [item["content"] for item in results]
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": docs,
            "top_n": top_k,
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

        # 重新组织结果
        reranked = []
        for item in result["results"]:
            idx = item["index"]
            original = results[idx]
            original["rerank_score"] = item["score"]
            reranked.append(original)

        return reranked

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[dict] = None,
        fact_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        date_range: Optional[dict] = None,
        use_rerank: bool = False,
    ) -> list[dict]:
        """
        检索记忆
        :param query: 查询字符串
        :param limit: 返回数量
        :param filters: 过滤条件（兼容格式）
        :param fact_type: 按事实类型过滤
        :param tags: 按标签过滤（任一匹配）
        :param date_range: 按日期范围过滤 {"start": timestamp, "end": timestamp}
        :param use_rerank: 是否使用重排
        :return: 检索结果列表
        """
        # 合并过滤条件
        combined_filters = filters or {}
        if fact_type:
            combined_filters["fact_type"] = fact_type
        if tags:
            combined_filters["tags"] = tags
        if date_range:
            combined_filters["date_range"] = date_range

        # 初始检索（多取一些用于 rerank）
        search_limit = limit * 3 if use_rerank else limit
        results = self.vector_store.search(query, limit=search_limit, filters=combined_filters)

        # 重排
        if use_rerank and results:
            if self.rerank_enabled:
                results = self._rerank_with_api(query, results, top_k=limit)
            else:
                results = self._simple_rerank(query, results, top_k=limit)

        return results[:limit]

    def retrieve_with_context(
        self,
        query: str,
        limit: int = 5,
        time_decay: bool = True,
    ) -> list[dict]:
        """
        带时间衰减的检索（近期记忆权重更高）
        """
        results = self.vector_store.search(query, limit=limit * 2)

        if time_decay:
            current_time = time.time()
            for item in results:
                # 时间衰减因子（30天为半衰期）
                age_days = (current_time - item.get("created_at", current_time)) / 86400
                decay = math.exp(-0.023 * age_days)  # ln(0.5) / 30 ≈ -0.023
                item["score"] = item["score"] * (0.7 + 0.3 * decay)

            results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]


# ============================================================================
# Convenience Factory
# ============================================================================


def create_vector_store(
    storage_path: str = "src/data/vectors.json",
    embedding_model: str = "text-embedding-v3",
    embedding_dims: int = 1024,
) -> VectorStore:
    """创建 VectorStore 实例"""
    return VectorStore(
        storage_path=storage_path,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
    )


def create_retriever(
    storage_path: str = "src/data/vectors.json",
    rerank_enabled: bool = False,
) -> HybridRetriever:
    """创建 HybridRetriever 实例"""
    return HybridRetriever(
        storage_path=storage_path,
        rerank_enabled=rerank_enabled,
    )


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=== L3 Vector Store 测试 ===\n")

    # 创建实例
    store = VectorStore(storage_path="src/data/vectors.json")
    retriever = HybridRetriever(vector_store=store)

    # 存储测试
    print("1. 存储测试")
    id1 = store.store("今天天气很好，阳光明媚", metadata={"source": "user"}, importance=0.8, tags=["weather", "daily"], fact_type="observation")
    id2 = store.store("人工智能技术发展迅速，大模型不断涌现", metadata={"source": "system"}, importance=0.9, tags=["AI", "tech"], fact_type="knowledge")
    id3 = store.store("用户偏好阅读技术文档", metadata={"source": "inference"}, importance=0.6, tags=["preference"], fact_type="preference")
    print(f"   存储了 3 条记忆: {id1}, {id2}, {id3}\n")

    # 检索测试
    print("2. 检索测试 (查询: '天气如何')")
    results = retriever.retrieve("天气如何", limit=3)
    for r in results:
        print(f"   - [{r['score']:.3f}] {r['content'][:40]}...")

    print("\n3. 检索测试 (查询: 'AI 技术')")
    results = retriever.retrieve("AI 技术", limit=3)
    for r in results:
        print(f"   - [{r['score']:.3f}] {r['content'][:40]}...")

    # 过滤测试
    print("\n4. 过滤测试 (fact_type='knowledge')")
    results = retriever.retrieve("技术", limit=3, fact_type="knowledge")
    for r in results:
        print(f"   - [{r['score']:.3f}] {r['content'][:40]}...")

    # 统计
    print("\n5. 统计信息")
    stats = store.get_stats()
    print(f"   总记忆数: {stats['total']}")
    print(f"   平均重要性: {stats['avg_importance']}")
    print(f"   事实类型分布: {stats['fact_types']}")
    print(f"   热门标签: {list(stats['tags'].keys())[:5]}")

    # 访问计数
    print("\n6. 访问计数测试")
    store.increment_access(id1)
    store.increment_access(id1)
    entry = store.get(id1)
    print(f"   {id1} 访问次数: {entry['access_count']}")

    print("\n=== 测试完成 ===")
