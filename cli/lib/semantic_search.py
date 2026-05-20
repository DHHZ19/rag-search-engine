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


class SemanticSearch:
    def generate_embedding(self, text):
        if len(text) == 0:
            raise ValueError("text can't only be whitespace")

        embed_text_list = []

        embed_text_list.append(text)
        embedding = self.model.encode(embed_text_list)

        return embedding[0]

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
