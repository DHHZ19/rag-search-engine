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

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

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
