#!/usr/bin/env python3

import argparse
import json
from lzma import CHECK_UNKNOWN

from lib.semantic_search import (
    SemanticSearch,
    embed_text,
    verify_embeddings,
    verify_model,
)


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

    chunk_command = subparsers.add_parser("chunk", help="add a search term")
    chunk_command.add_argument("chunk", type=str, help="add positional argument")
    chunk_command.add_argument(
        "--chunk-size", type=int, required=False, help="optional chunk size"
    )
    chunk_command.add_argument(
        "--overlap", type=int, required=False, help="optional chunk size"
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
        case "chunk":
            text = args.chunk
            overlap = args.overlap
            words = text.split()
            chunk_size = args.chunk_size
            chunks = []

            n_words = len(words)
            i = 0
            o = 0

            while i < n_words:
                chunk_words = words[i : i + chunk_size]

                if chunks and len(chunk_words) <= overlap:
                    break

                chunks.append(" ".join(chunk_words))
                i += chunk_size - overlap

            print(f"Chunking {len(text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
