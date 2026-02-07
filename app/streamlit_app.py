"""Streamlit web UI for semantic search over technical documents."""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path so src package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.search import SearchPipeline


def get_pipeline() -> SearchPipeline:
    """Get or create the cached search pipeline."""
    if "pipeline" not in st.session_state:
        documents_dir = str(PROJECT_ROOT / "documents")
        persist_dir = str(PROJECT_ROOT / "data" / "faiss")
        st.session_state.pipeline = SearchPipeline(
            documents_dir=documents_dir,
            persist_dir=persist_dir,
        )
    return st.session_state.pipeline


def render_sidebar():
    """Render the sidebar with stats, re-index button, and file uploader."""
    with st.sidebar:
        st.header("Index Management")

        pipeline = get_pipeline()
        stats = pipeline.get_stats()

        # Collection stats
        st.metric("Total Chunks", stats["total_chunks"])
        st.metric("Documents Indexed", stats["document_count"])

        if stats["documents"]:
            with st.expander("Indexed Documents"):
                for doc in stats["documents"]:
                    st.text(f"  {doc}")

        st.divider()

        # Re-index button
        if st.button("Re-index All Documents", use_container_width=True):
            with st.spinner("Re-indexing documents..."):
                count = pipeline.index(force_reindex=True)
            st.success(f"Indexed {count} chunks")
            st.rerun()

        # Index if empty
        if stats["total_chunks"] == 0:
            with st.spinner("Indexing sample documents..."):
                count = pipeline.index()
            st.success(f"Indexed {count} chunks from sample documents")
            st.rerun()

        st.divider()

        # File uploader
        st.subheader("Add Document")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["md", "txt", "pdf"],
            help="Upload .md, .txt, or .pdf files to add to the search index",
        )

        if uploaded_file is not None:
            # Save to documents directory
            documents_dir = PROJECT_ROOT / "documents"
            save_path = documents_dir / uploaded_file.name

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexing {uploaded_file.name}..."):
                chunks_added = pipeline.add_document(str(save_path))

            st.success(f"Added {uploaded_file.name} ({chunks_added} chunks)")
            st.rerun()


def render_results(results):
    """Render search results as styled cards."""
    for i, result in enumerate(results):
        # Top result gets highlighted
        if i == 0:
            st.markdown("##### Best Match")

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**{result.source}**")
            with col2:
                st.markdown(
                    f"Score: **{result.score:.1%}**"
                )

            # Relevance bar
            st.progress(max(0.0, min(1.0, result.score)))

            # Text excerpt
            st.markdown(result.text)


def main():
    st.set_page_config(
        page_title="Semantic Search",
        page_icon="magnifying glass",
        layout="wide",
    )

    st.title("Semantic Search for Technical Documents")
    st.markdown(
        "Search across technical documents using natural language. "
        "Results are ranked by meaning similarity, not just keyword matching."
    )

    render_sidebar()

    # Search box
    query = st.text_input(
        "Search query",
        placeholder="e.g., fire resistance requirements for steel structures",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([4, 1])
    with col2:
        n_results = st.selectbox(
            "Results",
            options=[3, 5, 10],
            index=1,
            label_visibility="collapsed",
        )

    if query:
        pipeline = get_pipeline()

        stats = pipeline.get_stats()
        if stats["total_chunks"] == 0:
            st.warning("No documents indexed yet. Use the sidebar to index documents.")
            return

        with st.spinner("Searching..."):
            results = pipeline.search(query, n_results=n_results)

        if results:
            st.markdown(f"**{len(results)} results** for: *{query}*")
            st.divider()
            render_results(results)
        else:
            st.info("No results found. Try a different query.")
    else:
        # Show example queries when no search is active
        st.markdown("---")
        st.markdown("**Try these example queries:**")
        examples = [
            "fire resistance requirements for steel structures",
            "personal protective equipment compliance",
            "hazardous waste disposal procedures",
            "lockout tagout electrical safety",
            "stormwater management and erosion control",
        ]
        for example in examples:
            st.markdown(f"- {example}")


if __name__ == "__main__":
    main()
