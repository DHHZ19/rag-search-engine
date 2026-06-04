#!/usr/bin/env python3

import argparse
from pathlib import Path

from multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image = subparsers.add_parser(
        "verify_image_embedding", help="Verify the image embedding"
    )
    verify_image.add_argument("verify_image_embedding", type=str, help="add image path")

    image_search = subparsers.add_parser(
        "image_search", help="search for a movie via an image"
    )
    image_search.add_argument("image_search", type=str, help="add image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.verify_image_embedding
            verify_image_embedding(image_path)

        case "image_search":
            image_path = args.image_search
            scores = image_search_command(image_path)

            for i, s in enumerate(scores, start=1):
                print(f"{i} {s['title']} (similarity: {s['score'][0][0].item():.1f})")
                print(f"  {s['description']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
