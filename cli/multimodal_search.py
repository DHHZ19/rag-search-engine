from PIL import Image
from search_utils import load_movies
from sentence_transformers import SentenceTransformer


def verify_image_embedding(image_path):
    movies = load_movies()
    ms = MultimodalSearch(documents=movies)
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path):
    movies = load_movies()
    ms = MultimodalSearch(documents=movies)
    scores = ms.search_with_image(image_path)
    return scores


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.documents = load_movies()
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.st = SentenceTransformer(model_name)
        self.text_embeddings = self.st.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image_file = Image.open(image_path)

        embedding = self.st.encode(image_file)
        return embedding

    def search_with_image(self, image_path: str):
        embedding = self.embed_image(image_path)

        res = []

        for doc_embedding in self.text_embeddings:
            res.append(self.st.similarity(embedding, doc_embedding))

        for i, doc in enumerate(self.documents):
            res[i] = {"score": res[i], **doc}

        res = sorted(res, key=lambda item: item["score"], reverse=True)

        return res[:5]
