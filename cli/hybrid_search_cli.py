import argparse

from hybrid_search import HybridSearch, hybrid_score, weighted_search_command
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
