import os
from itertools import repeat

from keyword_search import InvertedIndex
from semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def normalize_scores(self, scores: list[float]) -> list[float] | None:
        if len(scores) == 0:
            return None

        scores.sort(reverse=True)
        max_score = scores[0]
        min_score = scores[-1]

        repeat_num = 1.0
        if max_score == min_score:
            return list(repeat(repeat_num, len(scores)))

        for i, score in enumerate(scores):
            scores[i] = (score - min_score) / (max_score - min_score)

        return scores

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        search_limit = limit * 500
        bm25_search_res = self._bm25_search(query, search_limit)

        semantic_search_res = self.semantic_search.search_chunks(query, search_limit)

        bm25_scores = self._normalize_result_scores(bm25_search_res)
        semantic_scores = self._normalize_result_scores(semantic_search_res)

        docs_scores = []
        doc_ids = set(bm25_scores) | set(semantic_scores)
        documents_by_id = {doc["id"]: doc for doc in self.documents}

        for doc_id in doc_ids:
            keyword_score = bm25_scores.get(doc_id, 0.0)
            semantic_score = semantic_scores.get(doc_id, 0.0)

            docs_scores.append(
                {
                    "document": documents_by_id[doc_id],
                    "keyword_score": keyword_score,
                    "semantic_score": semantic_score,
                    "hybrid_score": hybrid_score(keyword_score, semantic_score, alpha),
                }
            )

        doc_scores_sorted = sorted(
            docs_scores, key=lambda item: item["hybrid_score"], reverse=True
        )

        return doc_scores_sorted[:limit]

    def _normalize_result_scores(self, results: list[dict]) -> dict[int, float]:
        if len(results) == 0:
            return {}

        scores = [result["score"] for result in results]
        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:
            return {self._result_doc_id(result): 1.0 for result in results}

        normalized_scores = {}
        for result in results:
            normalized_scores[self._result_doc_id(result)] = (
                result["score"] - min_score
            ) / (max_score - min_score)

        return normalized_scores

    def _result_doc_id(self, result: dict) -> int:
        doc_id = result["id"]
        if isinstance(doc_id, dict):
            return doc_id["id"]
        return doc_id


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score
