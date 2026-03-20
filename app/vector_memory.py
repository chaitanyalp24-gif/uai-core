import faiss
import numpy as np


class VectorMemory:
    def __init__(self):
        self.model = None
        self.index = None
        self.texts = []
        self.dim = 384

    def _load_model(self):
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                trust_remote_code=True
            )
            self.index = faiss.IndexFlatL2(self.dim)
        except Exception as e:
            print("⚠️ Vector memory disabled:", str(e))
            self.model = None
            self.index = None

    def add(self, text: str):
        self._load_model()
        if not self.model:
            return

        emb = self.model.encode(text).astype("float32")
        self.index.add(np.array([emb]))
        self.texts.append(text)

    def search(self, query: str, k: int = 3):
        self._load_model()
        if not self.model or not self.texts:
            return []

        emb = self.model.encode(query).astype("float32")
        _, idxs = self.index.search(np.array([emb]), k)

        return [
            self.texts[i]
            for i in idxs[0]
            if i < len(self.texts)
        ]
