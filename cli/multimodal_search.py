from PIL import Image
from sentence_transformers import SentenceTransformer


def verify_image_embedding(image_path):
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.st = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        image_file = Image.open(image_path)

        embedding = self.st.encode(image_file)
        return embedding
