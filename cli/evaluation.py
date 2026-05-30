import json
import math

from hybrid_search import rrf_search_command
from search_utils import GOLDEN_DATASET_PATH


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
