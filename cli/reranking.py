import json
import os
from time import sleep
from typing import Literal

from dotenv import load_dotenv
from google import genai
from search_utils import SearchResult
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


def llm_rerank_individual(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[dict[str, object]]:
    scored_docs: list[dict[str, object]] = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def llm_rerank_batch(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[dict[str, object]]:
    if not documents:
        return []

    doc_map: dict[int, SearchResult] = {}
    doc_list: list[str] = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return the movie IDs in order of relevance, best match first.

    Your response must be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    response = client.models.generate_content(model=model, contents=prompt)
    ranking_text = (response.text or "").strip()

    parsed_ids = json.loads(ranking_text)

    reranked: list[dict[str, object]] = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    return reranked[:limit]


def cross_encoder_rerank(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[dict[str, object]]:
    pairs: list[list[str]] = []
    reranked_documents: list[dict[str, object]] = [dict(doc) for doc in documents]
    for doc in reranked_documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(reranked_documents, scores):
        doc["crossencoder_score"] = float(score)

    reranked_documents.sort(key=lambda x: float(x["crossencoder_score"]), reverse=True)
    return reranked_documents[:limit]


def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual", "batch", "cross_encoder"] = "batch",
    limit: int = 5,
) -> list[dict[str, object]]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    if method == "batch":
        return llm_rerank_batch(query, documents, limit)
    if method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)
    else:
        return documents[:limit]
