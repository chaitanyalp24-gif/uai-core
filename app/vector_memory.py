import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorMemory:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.texts = []

    def add(self, text: str):
        embedding = self.embed(text)
        self.index.add(np.array([embedding]))
        self.texts.append(text)

    def search(self, query: str, k: int = 3):
        if len(self.texts) == 0:
            return []

        query_embedding = self.embed(query)
        distances, indices = self.index.search(
            np.array([query_embedding]), k
        )

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results

    def embed(self, text: str):
        return self.model.encode(text).astype("float32")
