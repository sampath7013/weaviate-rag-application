import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st


# ============================================================
# Project root setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.15rem;
        border-radius: 999px;
        background-color: rgba(100, 116, 139, 0.12);
        font-size: 0.85rem;
    }

    .document-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: rgba(100, 116, 139, 0.08);
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .source-card {
        padding: 0.8rem;
        border-radius: 8px;
        background-color: rgba(100, 116, 139, 0.06);
        margin-bottom: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Streamlit Cloud secrets
# ============================================================

def load_streamlit_secrets() -> None:
    secret_names = [
        "APP_ENV",
        "OPENAI_API_KEY",
        "WEAVIATE_HOST",
        "WEAVIATE_HTTP_PORT",
        "WEAVIATE_GRPC_PORT",
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "WEAVIATE_COLLECTION",
        "EMBEDDING_MODEL",
        "LLM_MODEL",
    ]

    for secret_name in secret_names:
        try:
            value = st.secrets.get(secret_name)

            if value is not None:
                os.environ[secret_name] = str(value)

        except Exception:
            pass


load_streamlit_secrets()


# ============================================================
# Import application modules
# ============================================================

from app.generation.rag_chain import ask_rag
from app.ingestion.ingest import ingest_pdf


# ============================================================
# Session state
# ============================================================

defaults = {
    "document_id": None,
    "document_name": None,
    "document_ready": False,
    "messages": [],
    "pages": None,
    "chunks": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a PDF to build a searchable knowledge base.",
    )

    process_document = st.button(
        "Process Document",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.divider()

    st.subheader("Retrieval Settings")

    top_k = st.slider(
        "Top K chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Maximum number of candidate chunks retrieved.",
    )

    max_distance = st.slider(
        "Maximum vector distance",
        min_value=0.10,
        max_value=1.00,
        value=0.55,
        step=0.05,
        help=(
            "Lower values require stronger semantic similarity. "
            "Chunks above this distance are filtered out."
        ),
    )

    st.divider()

    st.subheader("Current Document")

    if st.session_state.document_ready:

        st.success("Ready")

        st.markdown(
            f"""
            <div class="document-card">
                <strong>{st.session_state.document_name}</strong>
                <br>
                Pages: {st.session_state.pages or "N/A"}
                <br>
                Chunks: {st.session_state.chunks or "N/A"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Clear Document",
            use_container_width=True,
        ):
            st.session_state.document_id = None
            st.session_state.document_name = None
            st.session_state.document_ready = False
            st.session_state.messages = []
            st.session_state.pages = None
            st.session_state.chunks = None

            st.rerun()

    else:
        st.info(
            "Upload and process a PDF to begin."
        )

    st.divider()

    with st.expander("About this project"):

        st.write(
            "A production-style Retrieval-Augmented Generation "
            "application that combines semantic search with "
            "grounded LLM responses."
        )

        st.markdown(
            """
            **Key features**

            - PDF ingestion
            - SHA-256 duplicate detection
            - Batch embeddings
            - Weaviate vector search
            - Document-specific filtering
            - Relevance thresholding
            - Source citations
            - OpenAI answer generation
            """
        )


# ============================================================
# Main header
# ============================================================

st.markdown(
    '<div class="main-title">📚 Document RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload a PDF and ask questions grounded only in the
        document content.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Technology badges
# ============================================================

st.markdown(
    """
    <span class="tech-badge">Python</span>
    <span class="tech-badge">Streamlit</span>
    <span class="tech-badge">OpenAI</span>
    <span class="tech-badge">Weaviate</span>
    <span class="tech-badge">FastAPI</span>
    <span class="tech-badge">RAG</span>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# PDF ingestion
# ============================================================

if process_document and uploaded_file is not None:

    document_id = str(uuid.uuid4())

    original_filename = Path(
        uploaded_file.name
    ).name

    temporary_path = None

    try:

        with st.status(
            "Processing document...",
            expanded=True,
        ) as status:

            st.write("Saving PDF...")

            suffix = Path(
                original_filename
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary_file:

                temporary_file.write(
                    uploaded_file.getbuffer()
                )

                temporary_path = (
                    temporary_file.name
                )

            st.write(
                "Checking for duplicate document..."
            )

            st.write(
                "Extracting and chunking text..."
            )

            st.write(
                "Creating embeddings..."
            )

            st.write(
                "Indexing vectors in Weaviate..."
            )

            result = ingest_pdf(
                file_path=temporary_path,
                document_id=document_id,
                document_name=original_filename,
            )

            st.session_state.document_id = (
                result["document_id"]
            )

            st.session_state.document_name = (
                result["filename"]
            )

            st.session_state.document_ready = True

            st.session_state.messages = []

            st.session_state.pages = (
                result.get("pages")
            )

            st.session_state.chunks = (
                result.get("chunks")
            )

            if result["duplicate"]:

                status.update(
                    label=(
                        "Document already indexed — "
                        "using existing vectors."
                    ),
                    state="complete",
                    expanded=False,
                )

                st.info(
                    "This PDF already exists in the knowledge base. "
                    "The existing index was reused."
                )

            else:

                status.update(
                    label="Document processed successfully.",
                    state="complete",
                    expanded=False,
                )

                st.success(
                    f"Indexed {result['pages']} pages "
                    f"into {result['chunks']} chunks."
                )

    except Exception:

        st.session_state.document_ready = False

        st.error(
            "The document could not be processed. "
            "Please try again."
        )

    finally:

        if (
            temporary_path
            and Path(temporary_path).exists()
        ):
            Path(
                temporary_path
            ).unlink()


# ============================================================
# Landing state
# ============================================================

if not st.session_state.document_ready:

    st.subheader(
        "How it works"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 1. Upload")

        st.write(
            "Upload a PDF from the sidebar."
        )

    with col2:

        st.markdown("### 2. Index")

        st.write(
            "The document is chunked, embedded, "
            "and stored in Weaviate."
        )

    with col3:

        st.markdown("### 3. Ask")

        st.write(
            "Ask natural-language questions and "
            "receive grounded answers with sources."
        )

    st.info(
        "Upload a PDF from the sidebar to start."
    )


# ============================================================
# Chat section
# ============================================================

else:

    st.subheader(
        "💬 Ask Your Document"
    )

    if not st.session_state.messages:

        st.info(
            "Try asking a question about the uploaded PDF."
        )

        st.caption(
            "Example: “Summarize the main points of this document.”"
        )


# ============================================================
# Render previous messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            sources = message[
                "sources"
            ]

            with st.expander(
                f"Sources ({len(sources)})"
            ):

                for index, source in enumerate(
                    sources,
                    start=1,
                ):

                    distance = source.get(
                        "distance"
                    )

                    st.markdown(
                        f"#### Source {index}"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    with col1:
                        st.caption("Page")
                        st.write(
                            source.get(
                                "page_number",
                                "N/A",
                            )
                        )

                    with col2:
                        st.caption("Chunk")
                        st.write(
                            source.get(
                                "chunk_index",
                                "N/A",
                            )
                        )

                    with col3:
                        st.caption("Distance")

                        if distance is not None:
                            st.write(
                                f"{distance:.4f}"
                            )
                        else:
                            st.write("N/A")

                    st.caption(
                        source.get(
                            "document_name",
                            "Unknown document",
                        )
                    )

                    st.write(
                        source.get(
                            "text",
                            "",
                        )
                    )

                    st.divider()


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    (
        "Ask a question about the uploaded document..."
        if st.session_state.document_ready
        else "Upload a document first..."
    ),
    disabled=not st.session_state.document_ready,
)


# ============================================================
# Handle question
# ============================================================

if question:

    cleaned_question = question.strip()

    if cleaned_question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": cleaned_question,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                cleaned_question
            )

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching document..."
                ):

                    result = ask_rag(
                        question=cleaned_question,
                        top_k=top_k,
                        document_id=(
                            st.session_state.document_id
                        ),
                        max_distance=max_distance,
                    )

                answer = result.get(
                    "answer",
                    (
                        "The document does not contain "
                        "enough information to answer "
                        "this question."
                    ),
                )

                sources = result.get(
                    "sources",
                    [],
                )

                st.markdown(
                    answer
                )

                if sources:

                    with st.expander(
                        f"Sources ({len(sources)})"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):

                            distance = source.get(
                                "distance"
                            )

                            st.markdown(
                                f"#### Source {index}"
                            )

                            col1, col2, col3 = (
                                st.columns(3)
                            )

                            with col1:

                                st.caption(
                                    "Page"
                                )

                                st.write(
                                    source.get(
                                        "page_number",
                                        "N/A",
                                    )
                                )

                            with col2:

                                st.caption(
                                    "Chunk"
                                )

                                st.write(
                                    source.get(
                                        "chunk_index",
                                        "N/A",
                                    )
                                )

                            with col3:

                                st.caption(
                                    "Distance"
                                )

                                if distance is not None:

                                    st.write(
                                        f"{distance:.4f}"
                                    )

                                else:

                                    st.write(
                                        "N/A"
                                    )

                            st.caption(
                                source.get(
                                    "document_name",
                                    "Unknown document",
                                )
                            )

                            st.write(
                                source.get(
                                    "text",
                                    "",
                                )
                            )

                            st.divider()

                else:

                    st.info(
                        "No document chunks passed "
                        "the relevance threshold."
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception:

                error_message = (
                    "I couldn't process that question. "
                    "Please try again."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Built with Python • Streamlit • FastAPI • OpenAI • "
    "Weaviate • Retrieval-Augmented Generation"
)