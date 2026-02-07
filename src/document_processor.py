"""Document loading and text chunking for the semantic search pipeline."""

import os
import re
from pathlib import Path
from typing import Dict, List

from PyPDF2 import PdfReader


class DocumentProcessor:
    """Loads documents from files and splits them into searchable chunks."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

    def load_document(self, file_path: str) -> str:
        """Load a document file and return its text content.

        Supports .md, .txt, and .pdf files.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".pdf":
            return self._load_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    def _load_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks, respecting paragraph boundaries.

        Strategy:
        1. Split on double newlines (paragraphs)
        2. If a paragraph exceeds chunk_size, split on sentences
        3. Combine small paragraphs until chunk_size is reached
        4. Add overlap between chunks
        """
        paragraphs = re.split(r"\n{2,}", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Break oversized paragraphs into sentences
        segments = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                segments.append(para)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                segments.extend(sentences)

        # Combine segments into chunks
        chunks = []
        current_chunk = ""

        for segment in segments:
            candidate = (
                f"{current_chunk}\n\n{segment}" if current_chunk else segment
            )
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = segment

        if current_chunk:
            chunks.append(current_chunk)

        # Add overlap between chunks
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_text = chunks[i - 1]
                overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
                # Find a word boundary in the overlap
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1 :]
                overlapped.append(f"{overlap_text} {chunks[i]}")
            chunks = overlapped

        return chunks

    def process_directory(self, dir_path: str) -> List[Dict]:
        """Walk a directory and process all supported documents.

        Returns a list of dicts with keys: text, source, chunk_id
        """
        results = []
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        files = sorted(
            f
            for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        for file_path in files:
            text = self.load_document(str(file_path))
            chunks = self.chunk_text(text)

            for i, chunk in enumerate(chunks):
                results.append(
                    {
                        "text": chunk,
                        "source": file_path.name,
                        "chunk_id": f"{file_path.stem}_chunk_{i}",
                    }
                )

        return results
