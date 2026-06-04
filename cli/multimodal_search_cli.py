#!/usr/bin/env python3

import argparse
from pathlib import Path

from multimodal_search import verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image = subparsers.add_parser(
        "verify_image_embedding", help="Verify the image embedding"
    )
    verify_image.add_argument("verify_image_embedding", type=str, help="add image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.verify_image_embedding
            verify_image_embedding(image_path)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
