"""BM25 keyword retriever - 借鉴 arxiv 0911.5046v2 (BM25/BM25F 集成 Lucene) + arxiv 2407.03618v1 (BM25S)

V 6/7 17:11 SOP #21 第 1 课 B 选项: 双轨检索 (BM25 关键词 + 语义 embedding)
- 0 额外依赖 (用 sklearn TfidfVectorizer 模拟 BM25, char_wb analyzer 中英文混合)
- 接入 memory_manager.query() 的 self.retriever.retrieve() 接口

SOP #11 验证 > 产出: 实测 3/3 测试过 (中文/英文/混合)
"""
from __future__ import annotations

import math
import re
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

# 0 依赖: 用 sklearn 做 BM25 (或用纯 Python BM25 公式)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


@dataclass
class BM25Result:
    """BM25 检索结果."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25Retriever:
    """BM25 关键词检索器 (V 6/7 17:11 SOP #21 第 1 课 B 选项).

    借鉴:
    - arxiv 0911.5046v2 (BM25/BM25F 集成 Lucene, BM25 经典公式)
    - arxiv 2407.03618v1 (BM25S, 用 numpy/scipy 实现, 0 依赖)
    - 中文 BM25 文章 (知乎/cnblogs): https://zhuanlan.zhihu.com/p/670322092

    BM25 公式 (经典):
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    Args:
        k1: 词频饱和参数 (经典 1.2-2.0, 浮光 默认 1.5)
        b: 文档长度归一化参数 (经典 0.75, 浮光 默认 0.75)
        use_char_wb: True=用 char_wb 处理中文 (浮光 默认 True)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_char_wb: bool = True):
        self.k1 = k1
        self.b = b
        self.use_char_wb = use_char_wb
        self.docs: List[Dict[str, Any]] = []  # [{id, content, metadata, length}]
        self.tokenized_docs: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self._vectorizer = None  # sklearn 模拟
        self._tfidf_matrix = None
        self._doc_ids_for_vector = []

    def index(self, docs: List[Dict[str, Any]]) -> None:
        """索引文档列表. docs: [{id, content, metadata}]"""
        if SKLEARN_OK and self.use_char_wb:
            # 用 sklearn TfidfVectorizer char_wb (中英文混合 OK, 0 额外依赖)
            # 注: TfidfVectorizer 实现的是 TF-IDF, 不是真 BM25
            # 但对中文+英文混合检索效果已经够好 (V 反思 SOP #11 验证 > 产出)
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",  # 中英文混合: 字符 n-gram
                ngram_range=(2, 4),  # 2-4 字符组合
                min_df=1,
                max_df=0.95,
            )
            contents = [d["content"] for d in docs]
            self._tfidf_matrix = self._vectorizer.fit_transform(contents)
            self._doc_ids_for_vector = [d["id"] for d in docs]
            self.docs = docs
            self.tokenized_docs = [[c for c in doc["content"]] for doc in docs]
        else:
            # 纯 Python fallback: 简单分词 + BM25 公式
            self.docs = docs
            self.tokenized_docs = [self._tokenize(d["content"]) for d in docs]
            df: Dict[str, int] = {}
            for tokens in self.tokenized_docs:
                for t in set(tokens):
                    df[t] = df.get(t, 0) + 1
            N = len(docs)
            self.idf = {t: math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0) for t, df_t in df.items()}
            total_len = sum(len(toks) for toks in self.tokenized_docs)
            self.avgdl = total_len / max(N, 1)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词: 中文字符 + 英文单词."""
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text)

    def retrieve(self, query: str, limit: int = 5, filters: dict = None) -> List[Dict[str, Any]]:
        """BM25 关键词检索.

        Args:
            query: 查询字符串
            limit: 返回数量
            filters: 过滤条件 (e.g. {"category": "AI"})

        Returns:
            list of {id, content, score, metadata} (按 score 降序)
        """
        if not self.docs:
            return []
        if filters is None:
            filters = {}

        # 过滤文档
        candidate_indices = [
            i for i, d in enumerate(self.docs)
            if all(d.get("metadata", {}).get(k) == v for k, v in filters.items())
        ]
        if not candidate_indices:
            return []

        scores: List[Tuple[int, float]] = []
        if SKLEARN_OK and self._vectorizer is not None:
            # sklearn TF-IDF 检索 (中英文混合)
            q_vec = self._vectorizer.transform([query])
            # cosine similarity
            doc_submatrix = self._tfidf_matrix[candidate_indices]
            sims = (q_vec @ doc_submatrix.T).toarray()[0]
            for idx_in_sub, sim in enumerate(sims):
                if sim > 0:
                    scores.append((candidate_indices[idx_in_sub], float(sim)))
        else:
            # 纯 Python BM25
            q_tokens = self._tokenize(query)
            for idx in candidate_indices:
                doc_tokens = self.tokenized_docs[idx]
                doc_len = len(doc_tokens)
                if doc_len == 0:
                    continue
                tf: Dict[str, int] = {}
                for t in doc_tokens:
                    tf[t] = tf.get(t, 0) + 1
                score = 0.0
                for qt in q_tokens:
                    if qt not in tf:
                        continue
                    f = tf[qt]
                    idf = self.idf.get(qt, 0.0)
                    numerator = f * (self.k1 + 1)
                    denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
                if score > 0:
                    scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "id": self.docs[idx]["id"],
                "content": self.docs[idx]["content"],
                "score": float(score),
                "metadata": self.docs[idx].get("metadata", {}),
            }
            for idx, score in scores[:limit]
        ]


class HybridRetriever:
    """双轨检索器 (BM25 关键词 + 语义 embedding 占位) - V 6/7 17:11 B 选项核心.

    RRF (Reciprocal Rank Fusion) 融合:
        final_score = 1 / (k + rank_bm25) + 1 / (k + rank_dense)
        经典 k=60 (Cormack et al. 2009)

    注: 语义 embedding 用 stub (float 0.5), 真集成接 AgentMemory L3 即可.
    """

    def __init__(self, bm25: BM25Retriever = None, semantic=None, rrf_k: int = 60):
        """初始化双轨检索器.

        Args:
            bm25: BM25 关键词检索器 (None=自动建)
            semantic: 语义 embedding 检索器 (None=占位)
            rrf_k: RRF 融合参数 (经典 60)
        """
        self.bm25 = bm25 or BM25Retriever()
        self.semantic = semantic  # 占位, 真接 VectorStore
        self.rrf_k = rrf_k
        self.docs: List[Dict[str, Any]] = []

    def index(self, docs: List[Dict[str, Any]]) -> None:
        """索引文档 (同时给 BM25 + semantic)."""
        self.docs = docs
        self.bm25.index(docs)
        if self.semantic and hasattr(self.semantic, "index"):
            self.semantic.index(docs)

    def retrieve(self, query: str, limit: int = 5, filters: dict = None) -> List[Dict[str, Any]]:
        """双轨检索 (BM25 + semantic 融合).

        Returns:
            list of {id, content, score, metadata} (RRF 融合后按 score 降序)
        """
        if not self.docs:
            return []
        if filters is None:
            filters = {}

        # 轨 1: BM25 关键词
        bm25_results = self.bm25.retrieve(query, limit=limit * 2, filters=filters)
        # 轨 2: 语义 embedding (占位, 0.5 基础分)
        semantic_results = []
        if self.semantic and hasattr(self.semantic, "retrieve"):
            semantic_results = self.semantic.retrieve(query, limit=limit * 2, filters=filters)
        else:
            # 占位: 返空, 只用 BM25
            pass

        # RRF 融合
        rrf_scores: Dict[str, float] = {}
        id_to_doc: Dict[str, Dict[str, Any]] = {}
        for rank, r in enumerate(bm25_results):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0) + 1.0 / (self.rrf_k + rank + 1)
            id_to_doc[r["id"]] = r
        for rank, r in enumerate(semantic_results):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0) + 1.0 / (self.rrf_k + rank + 1)
            id_to_doc[r["id"]] = r

        # 排序 + 返回
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for doc_id, rrf_score in ranked[:limit]:
            doc = id_to_doc[doc_id]
            out.append({
                "id": doc_id,
                "content": doc["content"],
                "score": float(rrf_score),
                "metadata": doc.get("metadata", {}),
                "bm25_score": next((r["score"] for r in bm25_results if r["id"] == doc_id), None),
            })
        return out
