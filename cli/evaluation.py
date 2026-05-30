import json
import math
import os

from dotenv import load_dotenv
from google import genai
from hybrid_search import rrf_search_command
from search_utils import GOLDEN_DATASET_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)


def f1_score(precision: float, recall: float) -> float:
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def eval(limit: int):
    results = []
    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

        for movie in golden_dataset["test_cases"]:
            query = movie["query"]
            res = rrf_search_command(query, limit=limit)
            relevant_retrieved = 0
            total_retrieved = len(res["results"][:limit])

            relevant_retrieved_str = ""
            total_retrieved_str = ""
            for r in res["results"][:limit]:
                total_retrieved_str += r["title"] + ", "

            for reterieved in res["results"][:limit]:
                for relevant_doc in movie["relevant_docs"]:
                    if reterieved["title"] == relevant_doc:
                        relevant_retrieved += 1
                        relevant_retrieved_str += relevant_doc + ", "
            precision = relevant_retrieved / total_retrieved
            recall = relevant_retrieved / len(movie["relevant_docs"])
            f1 = f1_score(precision, recall)

            results.append(
                {
                    "query": query,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "retrieved": total_retrieved_str,
                    "relevant": relevant_retrieved_str,
                }
            )
    return results


def eval_results(result):
    query = result["query"]
    formatted_results = []
    for i, res in enumerate(result["results"], start=1):
        formatted_results.append(f"{i}. {res['title']}")

    contents = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(model="gemma-4-31b-it", contents=contents)

    if response.text:
        data = json.loads(response.text)
        for i, res in enumerate(data):
            i_display = i + 1
            print(f"{i_display}. {result['results'][i]['title']} {res}/3")
