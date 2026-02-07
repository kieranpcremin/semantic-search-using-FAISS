"""Embedding model wrapper for semantic search."""

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer for generating text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch encode a list of texts into embedding vectors."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Encode a single query string into an embedding vector."""
        embedding = self.model.encode(query, show_progress_bar=False)
        return embedding.tolist()
