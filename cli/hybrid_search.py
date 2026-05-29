import os

from keyword_search import InvertedIndex
from llm_reranking import batch_rerank_scores, rerank_scores
from search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
)
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

    def rrf_search(
        self, query: str, k: int, limit: int = 10, rerank_method: str = "batch"
    ) -> list[dict]:
        bm25_search_result = self._bm25_search(query, limit * 500)
        semantic_search_result = self.semantic_search.search_chunks(query, limit * 500)

        bm25_search_result = rank_scores(bm25_search_result)
        semantic_search_result = rank_scores(semantic_search_result)

        movie_ids_rankings = {}

        for score in bm25_search_result:
            if score["id"] not in movie_ids_rankings:
                movie_ids_rankings[score["id"]] = {
                    "title": score["title"],
                    "document": score["document"],
                    "bm25_rank": rrf_score(score["score"]),
                    "semantic_rank": 0.0,
                    "rrf_score": 0.0,
                }

        for score in semantic_search_result:
            if score["id"] not in movie_ids_rankings:
                movie_ids_rankings[score["id"]] = {
                    "title": score["title"],
                    "document": score["document"],
                    "bm25_rank": 0.0,
                    "semantic_rank": rrf_score(score["score"]),
                    "rrf_score": 0.0,
                }
            else:
                movie_ids_rankings[score["id"]]["semantic_rank"] = rrf_score(
                    score["score"]
                )

        for key, movie_rank in movie_ids_rankings.items():
            if movie_rank["bm25_rank"] != 0.0 and movie_rank["semantic_rank"] != 0.0:
                movie_rank["rrf_score"] = (
                    movie_rank["bm25_rank"] + movie_rank["semantic_rank"]
                )
            elif movie_rank["bm25_rank"] != 0.0 and movie_rank["semantic_rank"] == 0.0:
                movie_rank["rrf_score"] = movie_rank["bm25_rank"]
            elif movie_rank["bm25_rank"] == 0.0 and movie_rank["semantic_rank"] != 0.0:
                movie_rank["rrf_score"] = movie_rank["semantic_rank"]

        hybrid_ranks = []
        for doc_id, data in movie_ids_rankings.items():
            result = format_search_result(
                doc_id=doc_id,
                title=data["title"],
                document=data["document"],
                score=data["rrf_score"],
                bm25_score=data["bm25_rank"],
                semantic_score=data["semantic_rank"],
            )
            hybrid_ranks.append(result)

        if rerank_method == "batch":
            batch_rerank_scores(hybrid_ranks[:limit], query)
        else:
            rerank_scores(hybrid_ranks[:limit], query)

        sorted_movie_ids_rankings = sorted(
            hybrid_ranks[:limit],
            key=lambda item: item["rerank_score"],
        )

        return sorted_movie_ids_rankings

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
            if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
                combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
            if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
                combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_search_command(query: str, k=60, limit=10, rerank_method="batch"):
    movies = load_movies()
    hy = HybridSearch(movies)

    return hy.rrf_search(query, 8, limit, rerank_method)


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def rank_scores(scores: list[dict]):
    for i, score in enumerate(scores, start=1):
        score["score"] = i

    return scores
