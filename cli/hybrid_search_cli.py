import argparse

from hybrid_search import HybridSearch, hybrid_score
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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.normalize
            movies = load_movies()
            hs = HybridSearch(movies)

            print(hs.normalize_scores(scores))

        case "weighted-search":
            query = args.query
            alpha = args.alpha
            limit = args.limit
            movies = load_movies()
            hs = HybridSearch(movies)

            hs.semantic_search.load_or_create_embeddings(movies)
            docs_with_scores = hs.weighted_search(query, alpha, limit)

            for i, doc in enumerate(docs_with_scores, start=1):
                print(f"{i}\n")
                print(f"{doc['document']['title']}")
                print(
                    f"BM25: {doc['keyword_score']}, Semantic: {doc['semantic_score']} "
                )
                print(f"{doc['document']['description']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
