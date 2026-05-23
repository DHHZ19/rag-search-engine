import json

import numpy as np
from numpy._core.multiarray import ndarray
from sentence_transformers import SentenceTransformer


def verify_model():
    ss = SemanticSearch()
    model = ss.model
    max_sequence = model.max_seq_length
    print(f"Model Loaded: {model}")
    print(f"Max sequence length: {max_sequence}")


def embed_text(text):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()

    with open("data/movies.json", "rb") as file:
        data = json.load(file)

        documents = list(data["movies"])

        embeddings = ss.load_or_create_embeddings(documents)

        print(f"Number of docs: {len(documents)}")
        print(
            f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
        )


def buld_query_text(query):
    ss = SemanticSearch()

    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        t = text.strip()
        if len(t) == 0:
            raise ValueError("text can't only be whitespace")

        embed_text_list = []

        embed_text_list.append(t)
        embedding = self.model.encode(embed_text_list, show_progress_bar=True)

        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doc_strings = []
        for i, doc in enumerate(self.documents, start=1):
            self.document_map[i] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")

            self.embeddings = self.generate_embedding(doc_strings)

        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        # set docuents, and docmap
        self.documents = documents
        doc_strings = []
        for i, doc in enumerate(self.documents, start=1):
            self.document_map[i] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")

        # load embeddings if present
        with open("cache/movie_embeddings.npy", "rb") as f:
            if f is not None:
                embeddings = np.load("cache/movie_embeddings.npy")
                if len(embeddings) == len(documents):
                    return embeddings
            return self.build_embeddings(documents=documents)

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        embedding = self.generate_embedding(query)

        similarity_scores = []

        for i, e in enumerate(self.embeddings, start=1):
            similarity_score = cosine_similarity(embedding, e)
            touple = (self.document_map[i], similarity_score)
            similarity_scores.append(touple)

        sorted_similarity_scores = sorted(
            similarity_scores, key=lambda item: item[1], reverse=True
        )

        # sorted = np.sort(similarity_scores, axis=1)

        # sorted_similarity_scores = sorted[::-1]

        dic_similarities = []
        for touples_similarites in sorted_similarity_scores[:limit]:
            dic_similarities.append(
                {
                    "score": touples_similarites[1],
                    "info": self.document_map[int(touples_similarites[0]["id"])],
                }
            )

        return dic_similarities


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for i, doc in enumerate(documents, start=1):
            if i not in self.document_map:
                self.document_map[i] = doc
