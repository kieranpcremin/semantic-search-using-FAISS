# Semantic Search Engine for Technical Documents

**A semantic search engine that understands the *meaning* of your queries — not just the keywords — to find the most relevant sections across technical document collections, built with SentenceTransformers and FAISS.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.2+-green)](https://www.sbert.net/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7+-orange)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

<img width="1862" height="967" alt="image" src="https://github.com/user-attachments/assets/0ebc424c-f853-496b-b258-e9ef14dc6170" />


## 🎯 What This Project Does

Type a natural language query — in your own words — and get the most relevant paragraphs from a collection of technical documents, ranked by semantic similarity.

- **Semantic search** — finds results by meaning, not keyword matching
- **Cross-document discovery** — surfaces related content across different documents you didn't know were connected
- **Upload & search** — add your own documents (.md, .txt, .pdf) through the web UI
- **RAG-ready** — implements the retrieval half of Retrieval-Augmented Generation

---

## 🖥️ Try These Searches

| Query | What It Finds |
|-------|--------------|
| `fire resistance requirements for steel structures` | Results from **both** fire protection and structural steel docs |
| `personal protective equipment compliance` | Safety standards content — even paragraphs that say "hard hats" not "PPE" |
| `hazardous waste disposal procedures` | Environmental compliance content |
| `lockout tagout electrical safety` | Electrical safety procedures |
| `stormwater management and erosion control` | Environmental content about water management |

---

## 🤔 The Problem: Why Keyword Search Fails

Traditional keyword search (like `Ctrl+F` or SQL `LIKE '%term%'`) has fundamental limitations when searching technical documents:

### 1. The Synonym Problem
A construction engineer searching for **"PPE requirements"** won't find paragraphs that say *"hard hats, safety glasses, and high-visibility vests must be worn"* — even though that's exactly what they need. Keyword search demands the exact term.

### 2. The Context Problem
Searching for **"fire resistance"** in a structural steel document and a fire protection document requires understanding that both topics are relevant. A keyword search treats each match identically — it has no concept of *relevance* or *meaning*.

### 3. The Vocabulary Mismatch Problem
Technical documents use varied terminology. **"Fall protection"** might appear as *"fall arrest systems"*, *"guardrails"*, *"safety harnesses"*, or *"edge protection"*. A keyword search needs every variant spelled out; semantic search handles this automatically.

### 4. The Natural Language Problem
Engineers ask questions in natural language: *"What are the requirements for working at height?"*. Keyword search can't interpret this as a question — it just looks for the literal words "requirements", "working", "height" with no understanding of intent.

### How Semantic Search Solves This

Instead of matching words, semantic search converts both the query and every document chunk into **embedding vectors** — numerical representations that capture meaning. Similar meanings produce similar vectors, regardless of the exact words used.

```
Query: "safety gear for workers"         Document chunk: "PPE including hard hats
         │                                and high-vis vests are mandatory"
         ▼                                         │
   ┌──────────┐                              ┌──────────┐
   │ [0.12,   │    cosine similarity         │ [0.15,   │
   │  0.84,   │ ◄──── 0.82 (high!) ────►    │  0.79,   │
   │  0.31,   │                              │  0.28,   │
   │  ...384] │                              │  ...384] │
   └──────────┘                              └──────────┘
   Query vector                              Document vector
```

The vectors are close in 384-dimensional space because the **meanings** are related, even though the **words** are different.

---

## 🧠 How Embeddings Work

This is the core concept. If you understand embeddings, you understand modern NLP.

### What Is an Embedding?

An embedding is a list of numbers (a **vector**) that represents the meaning of a piece of text. The model `all-MiniLM-L6-v2` produces 384 numbers for any input text.

```python
"construction safety"  →  [0.12, -0.45, 0.78, 0.03, ..., -0.21]  # 384 numbers
"building site hazards" → [0.11, -0.42, 0.75, 0.05, ..., -0.19]  # similar numbers!
"Italian cooking recipes" → [0.89, 0.12, -0.67, 0.44, ..., 0.56]  # very different
```

### Why Do Similar Meanings Get Similar Vectors?

The model was trained on millions of text pairs where humans labelled whether two sentences mean similar things. Through training, the model learned to place related concepts close together in vector space and unrelated concepts far apart.

### Measuring Similarity: Cosine Similarity

To find how similar two embeddings are, we calculate the **cosine of the angle** between them:

| Score | Meaning |
|-------|---------|
| **1.0** | Identical meaning |
| **0.7–0.9** | Strongly related |
| **0.4–0.7** | Somewhat related |
| **0.0–0.3** | Unrelated |
| **-1.0** | Opposite meaning |

**Implementation trick:** If you normalise vectors to unit length first, cosine similarity becomes a simple dot product — which FAISS computes extremely efficiently.

### Why all-MiniLM-L6-v2?

| Property | Value |
|----------|-------|
| **Parameters** | 22 million |
| **Output dimensions** | 384 |
| **Model size** | ~88 MB |
| **Speed** | ~14,000 sentences/second on GPU |
| **Quality** | Excellent for its size — trained on 1B+ sentence pairs |
| **Runs on** | CPU or GPU (this project uses CPU) |

It's the sweet spot: small enough to run on any machine, good enough to capture nuanced meaning. Larger models (like `all-mpnet-base-v2` at 768 dimensions) are more accurate but slower and heavier.

---

## 🏗️ Architecture

### System Overview

```
                        ┌──────────────────────────────────────────┐
                        │           Streamlit Web UI               │
                        │  ┌─────────────┐  ┌──────────────────┐  │
                        │  │ Search Bar  │  │  File Upload     │  │
                        │  └──────┬──────┘  └────────┬─────────┘  │
                        └────────│───────────────────│────────────┘
                                 │                   │
                        ┌────────▼───────────────────▼────────────┐
                        │          SearchPipeline                  │
                        │         (Orchestrator)                   │
                        └──┬─────────────┬──────────────┬─────────┘
                           │             │              │
                  ┌────────▼──────┐ ┌────▼──────┐ ┌────▼──────────┐
                  │ Document      │ │ Embedding │ │  VectorStore  │
                  │ Processor     │ │ Model     │ │  (FAISS)      │
                  │               │ │           │ │               │
                  │ Load files    │ │ MiniLM    │ │ Index vectors │
                  │ Chunk text    │ │ 384-dim   │ │ Search cosine │
                  │ .md .txt .pdf │ │ vectors   │ │ Persist disk  │
                  └───────────────┘ └───────────┘ └───────────────┘
```

### Indexing Flow (One-Time)

```
Documents (.md, .txt, .pdf)
    │
    ▼
1. Load text content
    │
    ▼
2. Split into chunks (500 chars, 50 char overlap)
    │  - Respect paragraph boundaries
    │  - Break long paragraphs by sentence
    │  - Overlap prevents losing context at edges
    │
    ▼
3. Generate embeddings (384-dim vectors per chunk)
    │
    ▼
4. Normalise vectors (unit length for cosine similarity)
    │
    ▼
5. Add to FAISS index + save metadata to disk
```

### Search Flow (Per Query)

```
User query: "fire resistance for steel"
    │
    ▼
1. Embed query → 384-dim vector
    │
    ▼
2. Normalise query vector
    │
    ▼
3. FAISS finds top-N closest document vectors (dot product)
    │
    ▼
4. Return chunks + similarity scores
    │
    ▼
5. Display ranked results in UI
```

---

## 📄 Text Chunking Strategy

Chunking — how you split documents into searchable pieces — is one of the most important design decisions in a semantic search system.

### Why Chunk at All?

Embedding models have **token limits** (MiniLM maxes out at 256 tokens ≈ 400 words). Even if they didn't, embedding an entire 10-page document into a single vector would dilute the meaning — the vector would be an average of everything, matching nothing well.

### The Chunking Algorithm

```
Step 1: Split on paragraph boundaries (\n\n)
        → Preserves logical units of thought

Step 2: If paragraph > 500 chars, split on sentences
        → Prevents oversized chunks

Step 3: Combine small segments until chunk_size reached
        → Avoids tiny, meaningless chunks

Step 4: Add 50-char overlap between adjacent chunks
        → Prevents losing context at boundaries
```

### Chunk Size Trade-offs

| Size | Pros | Cons |
|------|------|------|
| **Too small** (< 200 chars) | Precise matches | Loses context; many fragments |
| **Too large** (> 1000 chars) | Rich context | Diluted embedding; less precise |
| **Sweet spot** (400–600 chars) | Good precision + context | Requires tuning per domain |

### Why Overlap Matters

Without overlap, a sentence split across two chunks would be lost from both embeddings. A 50-character overlap means the end of chunk N appears at the start of chunk N+1, preserving context at boundaries.

---

## 🔍 Keyword Search vs Semantic Search

| Feature | Keyword Search | Semantic Search |
|---------|---------------|-----------------|
| **Matching** | Exact word matching | Meaning-based matching |
| **Synonyms** | Misses them — "PPE" won't find "safety gear" | Finds them automatically |
| **Context** | No understanding of what words mean together | Understands phrases and intent |
| **Cross-topic** | Only finds documents with exact terms | Surfaces related content across documents |
| **Typo tolerance** | Fails on misspellings | Handles variations naturally |
| **Natural language** | Can't interpret questions | Understands query intent |
| **Setup** | Simple string matching | Requires embedding model + vector store |
| **Speed** | Faster for exact lookups | Slightly slower (embedding computation) |
| **Scalability** | Degrades with corpus size | FAISS handles millions of vectors |

**Example:** Searching **"personal protective equipment"** with keyword search won't find paragraphs that say *"safety gear"*, *"hard hats and safety glasses"*, or *"high-visibility vests must be worn."* Semantic search understands these all refer to the same concept.

---

## 🧠 Key Concepts Demonstrated

| Concept | Where | What I Learned |
|---------|-------|---------------|
| **Text Embeddings** | `embeddings.py` | Converting text into 384-dimensional vectors that capture meaning, not just words |
| **Vector Similarity** | `vector_store.py` | Using cosine similarity (via normalised dot product) to find semantically related content |
| **Text Chunking** | `document_processor.py` | Splitting documents into searchable chunks with paragraph-aware boundaries and overlap |
| **FAISS Vector Store** | `vector_store.py` | Efficient similarity search with persistence — FAISS stores vectors, metadata stored separately |
| **RAG Pipeline** | `search.py` | The retrieval half of Retrieval-Augmented Generation — add an LLM and you have a full RAG system |
| **Pipeline Orchestration** | `search.py` | Composing modular components (processor, embeddings, store) into a clean pipeline |
| **Web Deployment** | `streamlit_app.py` | Serving an ML-powered search engine through an interactive UI with file upload |

---

## 📁 Project Structure

```
semantic-search-using-FAISS/
├── app/
│   └── streamlit_app.py           # Web UI — search interface + file upload
├── src/
│   ├── __init__.py
│   ├── document_processor.py      # Load files (.md, .txt, .pdf) + chunk text
│   ├── embeddings.py              # SentenceTransformer wrapper (all-MiniLM-L6-v2)
│   ├── vector_store.py            # FAISS index — store, search, persist vectors
│   └── search.py                  # Pipeline orchestrator + SearchResult dataclass
├── documents/                     # 5 sample engineering/construction documents
│   ├── construction_safety_standards.md
│   ├── structural_steel_requirements.md
│   ├── fire_protection_guidelines.md
│   ├── electrical_safety_procedures.md
│   └── environmental_compliance.md
├── data/                          # FAISS index + metadata (generated, not in repo)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔍 Known Limitations & Honest Reflection

| Limitation | Impact | What I'd Do Differently |
|-----------|--------|------------------------|
| **Small corpus** | Only 5 sample documents (62 chunks) — too small to stress-test ranking | Test with 100+ real technical documents |
| **No hybrid search** | Pure semantic search misses exact term matches (e.g. part numbers, codes) | Combine with BM25 keyword search (hybrid approach) |
| **Fixed chunk size** | 500-char chunks may split important sections awkwardly | Use recursive or semantic chunking that respects document structure |
| **No re-ranking** | FAISS returns results in one pass — no refinement | Add a cross-encoder re-ranker for the top-N results |
| **No evaluation metrics** | No way to measure search quality objectively | Build a test set with labelled relevant results and measure MRR/NDCG |
| **Single embedding model** | all-MiniLM-L6-v2 is general-purpose, not domain-specific | Fine-tune on construction/engineering text or try domain-specific models |

### How I'd Improve It

- ✅ **Hybrid search** — combine FAISS semantic search with BM25 keyword search, merge results with reciprocal rank fusion
- ✅ **Cross-encoder re-ranking** — use a more expensive model to re-rank just the top 20 results for better precision
- ✅ **Add an LLM** — connect to an LLM to build a full RAG system that answers questions using retrieved context
- ✅ **Better chunking** — use recursive text splitters that respect markdown headers and section boundaries
- ✅ **Evaluation** — create a test set of queries with known relevant documents and measure retrieval quality

> This project implements the **retrieval** half of RAG. The architecture is designed so adding an LLM for the **generation** half would require minimal changes — feed the top-N search results as context to the LLM prompt.

---

## 🚀 Setup

### 1. Clone & Create Environment

```bash
git clone https://github.com/kieranpcremin/semantic-search-using-FAISS.git
cd semantic-search-using-FAISS
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` model (~88 MB) from Hugging Face. Subsequent runs use the cached version.

### 2. Index Sample Documents

```bash
python -c "from src.search import SearchPipeline; p = SearchPipeline(); print(f'Indexed {p.index()} chunks')"
```

This processes the 5 sample technical documents, chunks them into ~62 paragraphs, generates embeddings, and stores everything in a FAISS index.

### 3. Run the Web App

```bash
streamlit run app/streamlit_app.py
```

### 4. Upload Your Own Documents

Use the file upload in the sidebar to add your own `.md`, `.txt`, or `.pdf` documents. They're indexed immediately and searchable alongside the sample docs.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) | Convert text to 384-dim meaning vectors |
| **Vector Store** | FAISS (`IndexFlatIP`) | Fast cosine similarity search on normalised vectors |
| **Web UI** | Streamlit | Interactive search interface with file upload |
| **PDF Parsing** | PyPDF2 | Extract text from uploaded PDF documents |
| **ML Backend** | PyTorch | Runtime for the SentenceTransformer model |

---

## 📚 Data Types And Tech Stacks

| Project | Data Type | ML Type | Key Tech |
|---------|----------|---------|----------|
| [Safety Detector](https://github.com/kieranpcremin/hard-hat-detector) | Images | Classification (CNN) | PyTorch, ResNet18 |
| [Safety Detector (.NET)](https://github.com/kieranpcremin/safety-detector-dotnet) | Images | Classification (CNN) | .NET, TensorFlow, ML.NET |
| **Semantic Search** | **Text** | **Embeddings + Search** | **SentenceTransformers, FAISS** |
| [Timeline Predictor](https://github.com/kieranpcremin/project-timeline-predictor) | Tabular | Regression | scikit-learn, XGBoost |

---

## 👨‍💻 Author

**Kieran Cremin**
Built with assistance from Claude (Anthropic)

---

## 📄 License

MIT License — Free to use, modify, and distribute.
