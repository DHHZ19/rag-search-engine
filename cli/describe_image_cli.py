import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from hybrid_search import HybridSearch, rrf_search_command
from search_utils import load_movies

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--query", type=str, required=True, help="question")
    parser.add_argument("--image", type=str, required=True, help="path to the image")

    args = parser.parse_args()

    query = args.query
    image = args.image
    # movies = load_movies()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    response = None
    with open(image, "rb") as f:
        img = f.read()
        system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                - Synthesize visual and textual information
                - Focus on movie-specific details (actors, scenes, style, etc.)
                - Return only the rewritten query, without any additional commentary"""

        response = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=[
                system_prompt,
                query.strip(),
                types.Part.from_bytes(data=img, mime_type=mime),
            ],
        )

    print(f"Rewritten query: {response.text}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
