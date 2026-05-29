import argparse

from hybrid_search import (
    HybridSearch,
    rrf_search_command,
    weighted_search_command,
)
from query_enhancement import enhance_query
from search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="Search movies using BM25")
    normalize.add_argument(
        "normalize", nargs="+", type=float, help="One or more numbers"
    )

    weighted_search = subparsers.add_parser(
        "weighted-search", help="Search movies using BM25"
    )
    weighted_search.add_argument("query", type=str, help="Must Enter a Query")
    weighted_search.add_argument(
        "--alpha", type=float, help="Alpha how much weight you want on a keyword search"
    )
    weighted_search.add_argument("--limit", type=int, default=5, help="the limit")

    rrf_search = subparsers.add_parser(
        "rrf-search",
        help="rrf search used for searching and adds the scores of the semantic search and keyword search together using Reciprocal rank fusion",
    )
    rrf_search.add_argument("query", type=str, help="Must Enter a Query")
    rrf_search.add_argument(
        "-k", type=int, default=60, help="enter k paramater defaults to 60"
    )
    rrf_search.add_argument(
        "--limit", type=int, default=5, help="Enter the limit defaults to 5"
    )
    rrf_search.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        required=False,
        help="enter rerank method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.normalize
            movies = load_movies()
            hs = HybridSearch(movies)

            print(hs.normalize_scores(scores))

        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)

            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()

        case "rrf-search":
            query = args.query
            k = args.k
            limit = args.limit
            enhance = args.enhance
            rerank_method = args.rerank_method
            METHOD = enhance
            QUERY = query
            ENHANCED_QUERY = ""
            DEFULT_LIMIT = limit

            if rerank_method:
                limit = limit * 5

            query = enhance_query(QUERY, enhance)
            ENHANCED_QUERY = query

            print(f"Enhanced query ({METHOD}): '{QUERY}' -> '{ENHANCED_QUERY}'\n")

            res = rrf_search_command(query, k, limit, rerank_method)

            for i, doc_score in enumerate(res[:DEFULT_LIMIT], start=1):
                metadata = doc_score.get("metadata", {})
                print(f"{i}. {doc_score['title']}")
                print(f"Re-rank Score: {doc_score['rerank_score']}")
                print(f"RRF Score: {doc_score['score']}")
                print(
                    f"BM25 Rank: {metadata['bm25_score']} Semantic Rank: {metadata['semantic_score']}"
                )
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
