import argparse

from hybrid_search import HybridSearch
from search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="Search movies using BM25")
    normalize.add_argument(
        "normalize", nargs="+", type=float, help="One or more numbers"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.normalize
            movies = load_movies()
            hs = HybridSearch(movies)

            print(hs.normalize_scores(scores))

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
