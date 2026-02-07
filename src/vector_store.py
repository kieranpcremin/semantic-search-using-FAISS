"""FAISS vector store wrapper for document indexing and search."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from .embeddings import EmbeddingModel


class VectorStore:
    """Manages document embeddings in FAISS for similarity search."""

    INDEX_FILE = "index.faiss"
    METADATA_FILE = "metadata.json"

    def __init__(
        self,
        persist_dir: str = "data/faiss",
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.dimension = 384  # all-MiniLM-L6-v2 output dimension

        self._load_or_create()

    def _load_or_create(self):
        """Load existing index from disk or create a new one."""
        index_path = self.persist_dir / self.INDEX_FILE
        meta_path = self.persist_dir / self.METADATA_FILE

        if index_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            # Use IndexFlatIP (inner product) on normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def _save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, str(self.persist_dir / self.INDEX_FILE))
        with open(self.persist_dir / self.METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def index_documents(self, chunks: List[Dict]) -> None:
        """Add document chunks to the vector store.

        Each chunk dict must have: text, source, chunk_id
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        # Normalize for cosine similarity via inner product
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        for chunk in chunks:
            self.metadata.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
            })

        self._save()

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Query the index and return top-N results with similarity scores.

        Returns list of dicts with: text, source, score, chunk_id
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        k = min(n_results, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "text": meta["text"],
                "source": meta["source"],
                "score": round(float(score), 4),
                "chunk_id": meta["chunk_id"],
            })

        return results

    def get_stats(self) -> Dict:
        """Return index statistics."""
        sources = {m["source"] for m in self.metadata}
        return {
            "total_chunks": self.index.ntotal,
            "document_count": len(sources),
            "documents": sorted(sources),
        }

    def clear(self) -> None:
        """Reset the index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self._save()
