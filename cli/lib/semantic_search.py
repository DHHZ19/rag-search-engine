from sentence_transformers import SentenceTransformer


def verify_model():
    ss = SemanticSearch()
    model = ss.model
    max_sequence = model.max_seq_length
    print(f"Model Loaded: {model}")
    print(f"Max sequence length: {max_sequence}")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
