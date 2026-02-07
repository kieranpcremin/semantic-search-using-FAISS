"""Search pipeline orchestrator for semantic document search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


@dataclass
class SearchResult:
    """A single search result with relevance score and metadata."""

    text: str
    source: str
    score: float
    chunk_id: str


class SearchPipeline:
    """Orchestrates document processing, embedding, and search."""

    def __init__(
        self,
        documents_dir: str = "documents",
        persist_dir: str = "data/faiss",
    ):
        self.documents_dir = documents_dir
        self.persist_dir = persist_dir
        self.processor = DocumentProcessor()
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(
            persist_dir=persist_dir,
            embedding_model=self.embedding_model,
        )

    def index(self, force_reindex: bool = False) -> int:
        """Process and index all documents in the documents directory.

        Returns the number of chunks indexed.
        Skips indexing if documents are already indexed, unless force_reindex is True.
        """
        stats = self.vector_store.get_stats()

        if stats["total_chunks"] > 0 and not force_reindex:
            return stats["total_chunks"]

        if force_reindex:
            self.vector_store.clear()

        chunks = self.processor.process_directory(self.documents_dir)
        self.vector_store.index_documents(chunks)
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Run a semantic search query and return ranked results."""
        if not query.strip():
            return []

        results = self.vector_store.search(query, n_results=n_results)

        return [
            SearchResult(
                text=r["text"],
                source=r["source"],
                score=r["score"],
                chunk_id=r["chunk_id"],
            )
            for r in results
        ]

    def add_document(self, file_path: str) -> int:
        """Add a single document to the existing index.

        Returns the number of new chunks added.
        """
        text = self.processor.load_document(file_path)
        source = Path(file_path).name
        chunks_text = self.processor.chunk_text(text)

        chunks = [
            {
                "text": chunk,
                "source": source,
                "chunk_id": f"{Path(file_path).stem}_chunk_{i}",
            }
            for i, chunk in enumerate(chunks_text)
        ]

        self.vector_store.index_documents(chunks)
        return len(chunks)

    def get_stats(self) -> Dict:
        """Return index statistics."""
        return self.vector_store.get_stats()
