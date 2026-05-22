#!/usr/bin/env python3

import argparse
import json

from lib.semantic_search import verify_model
from lib.semantic_search import embed_text
from lib.semantic_search import verify_embeddings
from lib.semantic_search import SemanticSearch


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="verify that the semantic-search model loads")

    embed_command = subparsers.add_parser("embed_text", help="embed")
    embed_command.add_argument("query", type=str, help="Search query")

    subparsers.add_parser(
        "verify_embeddings", help="verify that the semantic-search model loads"
    )

    embed_command = subparsers.add_parser("embed_query", help="embed")
    embed_command.add_argument("query", type=str, help="Search query")

    search_command = subparsers.add_parser("search", help="add a search term")
    search_command.add_argument("search", type=str, help="Search query")
    search_command.add_argument(
        "--limit", type=int, required=False, help="Add a limit to your search"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            text = args.query
            embed_text(text)

        case "embed_query":
            text = args.query
            embed_text(text)

        case "search":
            search = args.search
            limit = args.limit

            ss = SemanticSearch()

            with open("data/movies.json", "rb") as file:
                data = json.load(file)

                documents = list(data["movies"])

                ss.embeddings = ss.load_or_create_embeddings(documents)

                res = ss.search(search)

                for i, r in enumerate(res, start=1):
                    print(
                        f"{i}. {r['info']['title']} (score: {r['score']}) \n {r['info']['description']} \n"
                    )

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
