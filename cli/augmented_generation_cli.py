import argparse
import os

from dotenv import load_dotenv
from google import genai
from hybrid_search import HybridSearch, rrf_search_command
from search_utils import load_movies

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sumarize_parser = subparsers.add_parser(
        "summarize", help="Perform search and then summarize the results"
    )
    sumarize_parser.add_argument("query", type=str, help="Search query for RAG")
    sumarize_parser.add_argument(
        "--limit", type=int, required=False, help="limit the search"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            movies = load_movies()
            hs = HybridSearch(movies)

            docs = hs.rrf_search(query)

            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {docs}

            Answer:"""

            response = client.models.generate_content(
                model="gemma-4-31b-it", contents=prompt
            )

            for doc in docs:
                print(doc["title"])

            print("RAG Response:")
            print(response.text)

        case "summarize":
            query = args.query

            movies = load_movies()
            hy = HybridSearch(movies)

            results = hy.rrf_search(query)

            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Search results:
            {results}

            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

            response = client.models.generate_content(
                model="gemma-4-31b-it", contents=prompt
            )

            print("Search Results:")
            for doc in results:
                print(f"    - {doc['title']}\n")
            print("LLM Summary:")
            print(response.text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
