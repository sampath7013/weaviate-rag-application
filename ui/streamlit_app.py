import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger(__name__)


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
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load Streamlit Cloud secrets
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

            value = st.secrets.get(
                secret_name
            )

            if value is not None:

                os.environ[
                    secret_name
                ] = str(value)

        except Exception:
            pass


load_streamlit_secrets()


# ============================================================
# Application imports
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
    "processing_message": None,
    "processing_message_type": None,
}


for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "📄 Document"
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help=(
            "Upload a PDF to build a searchable "
            "knowledge base."
        ),
    )

    process_document = st.button(
        "Process Document",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.divider()


    # ========================================================
    # Retrieval settings
    # ========================================================

    st.subheader(
        "Retrieval Settings"
    )

    top_k = st.slider(
        "Top K chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help=(
            "Maximum number of candidate chunks "
            "retrieved from Weaviate."
        ),
    )

    alpha = st.slider(
        "Semantic vs keyword balance",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help=(
            "Controls hybrid retrieval. "
            "0 = BM25 keyword search, "
            "1 = semantic vector search. "
            "0.50 gives equal weight to both."
        ),
    )

    min_score = st.slider(
        "Minimum relevance score",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help=(
            "Retrieved chunks below this hybrid "
            "relevance score are discarded."
        ),
    )

    st.divider()


    # ========================================================
    # Current document
    # ========================================================

    st.subheader(
        "Current Document"
    )

    if st.session_state.document_ready:

        st.success(
            "Ready"
        )

        pages_display = (
            st.session_state.pages
            if st.session_state.pages is not None
            else "N/A"
        )

        chunks_display = (
            st.session_state.chunks
            if st.session_state.chunks is not None
            else "N/A"
        )

        document_card = (
            '<div class="document-card">'
            f'<strong>{st.session_state.document_name}</strong><br>'
            f'Pages: {pages_display}<br>'
            f'Chunks: {chunks_display}'
            '</div>'
        )

        st.markdown(
            document_card,
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

            st.session_state.processing_message = None
            st.session_state.processing_message_type = None

            st.rerun()

    else:

        st.info(
            "Upload and process a PDF to begin."
        )

    st.divider()


    # ========================================================
    # About project
    # ========================================================

    with st.expander(
        "About this project"
    ):

        st.write(
            "A production-style Retrieval-Augmented "
            "Generation application combining semantic "
            "vector search and BM25 keyword search."
        )

        st.markdown(
            """
            **Key features**

            - PDF ingestion
            - SHA-256 duplicate detection
            - Batch embeddings
            - Weaviate hybrid retrieval
            - BM25 + semantic search
            - Document-specific filtering
            - Relevance score filtering
            - LLM relevance gate
            - Grounded answer generation
            - Source citations
            """
        )


# ============================================================
# Main header
# ============================================================

st.markdown(
    (
        '<div class="main-title">'
        '📚 Document RAG Assistant'
        '</div>'
    ),
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload a PDF and ask questions grounded only
        in the document content.
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
    <span class="tech-badge">Hybrid RAG</span>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# Processing result / flash message
# ============================================================

if st.session_state.processing_message:

    if (
        st.session_state.processing_message_type
        == "success"
    ):

        st.success(
            st.session_state.processing_message
        )

    else:

        st.info(
            st.session_state.processing_message
        )

    st.session_state.processing_message = None
    st.session_state.processing_message_type = None


# ============================================================
# PDF ingestion
# ============================================================

if (
    process_document
    and uploaded_file is not None
):

    document_id = str(
        uuid.uuid4()
    )

    original_filename = Path(
        uploaded_file.name
    ).name

    temporary_path = None

    try:

        with st.status(
            "Processing document...",
            expanded=True,
        ) as status:

            st.write(
                "Saving PDF..."
            )

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


            # =================================================
            # Store document state
            # =================================================

            st.session_state.document_id = (
                result["document_id"]
            )

            st.session_state.document_name = (
                result["filename"]
            )

            st.session_state.document_ready = True

            st.session_state.messages = []

            st.session_state.pages = (
                result.get(
                    "pages"
                )
            )

            st.session_state.chunks = (
                result.get(
                    "chunks"
                )
            )


            # =================================================
            # Duplicate document
            # =================================================

            if result.get(
                "duplicate",
                False,
            ):

                status.update(
                    label=(
                        "Document already indexed — "
                        "using existing vectors."
                    ),
                    state="complete",
                    expanded=False,
                )

                st.session_state.processing_message = (
                    "Document already indexed — "
                    "using existing vectors."
                )

                st.session_state.processing_message_type = (
                    "info"
                )


            # =================================================
            # New document
            # =================================================

            else:

                status.update(
                    label=(
                        "Document processed successfully."
                    ),
                    state="complete",
                    expanded=False,
                )

                st.session_state.processing_message = (
                    f"Indexed {result['pages']} pages "
                    f"into {result['chunks']} chunks."
                )

                st.session_state.processing_message_type = (
                    "success"
                )


        st.rerun()


    except Exception:

        logger.exception(
            "document_ingestion_failed | filename=%s",
            original_filename,
        )

        st.session_state.document_ready = False

        st.error(
            "The document could not be processed. "
            "Please try again."
        )


    finally:

        if (
            temporary_path
            and Path(
                temporary_path
            ).exists()
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

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.markdown(
            "### 1. Upload"
        )

        st.write(
            "Upload a PDF from the sidebar."
        )


    with col2:

        st.markdown(
            "### 2. Index"
        )

        st.write(
            "The document is chunked, embedded, "
            "and indexed in Weaviate."
        )


    with col3:

        st.markdown(
            "### 3. Ask"
        )

        st.write(
            "Ask natural-language questions "
            "and receive grounded answers "
            "with source citations."
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
            "Try asking a question about "
            "the uploaded PDF."
        )

        st.caption(
            "Example: "
            "“Summarize the main points "
            "of this document.”"
        )


# ============================================================
# Source renderer
# ============================================================

def render_sources(
    sources: list[dict],
) -> None:

    if not sources:
        return

    with st.expander(
        f"Sources ({len(sources)})"
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

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
                    "Relevance"
                )

                score = source.get(
                    "score"
                )

                if score is not None:

                    st.write(
                        f"{score:.4f}"
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


# ============================================================
# Render existing conversation
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message[
            "role"
        ]
    ):

        st.markdown(
            message[
                "content"
            ]
        )

        if (
            message["role"] == "assistant"
            and message.get(
                "sources"
            )
        ):

            render_sources(
                message[
                    "sources"
                ]
            )


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    (
        "Ask a question about the uploaded document..."
        if st.session_state.document_ready
        else "Upload a document first..."
    ),
    disabled=(
        not st.session_state.document_ready
    ),
)


# ============================================================
# Process question
# ============================================================

if question:

    cleaned_question = (
        question.strip()
    )

    if cleaned_question:

        st.session_state.messages.append(
            {
                "role":
                    "user",

                "content":
                    cleaned_question,
            }
        )


        # ====================================================
        # User message
        # ====================================================

        with st.chat_message(
            "user"
        ):

            st.markdown(
                cleaned_question
            )


        # ====================================================
        # Assistant response
        # ====================================================

        with st.chat_message(
            "assistant"
        ):

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
                        alpha=alpha,
                        min_score=min_score,
                    )


                answer = result.get(
                    "answer",
                    (
                        "The available document does not "
                        "contain enough relevant information "
                        "to answer this question."
                    ),
                )

                sources = result.get(
                    "sources",
                    [],
                )


                st.markdown(
                    answer
                )


                # =================================================
                # Sources
                # =================================================

                if sources:

                    render_sources(
                        sources
                    )

                else:

                    st.info(
                        "No sufficiently relevant "
                        "document context was found."
                    )


                # =================================================
                # Save assistant response
                # =================================================

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            answer,

                        "sources":
                            sources,
                    }
                )


            except Exception:

                logger.exception(
                    (
                        "rag_query_failed | "
                        "document_id=%s | "
                        "question=%s"
                    ),
                    st.session_state.document_id,
                    cleaned_question,
                )

                error_message = (
                    "I couldn't process that question. "
                    "Please try again."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role":
                            "assistant",

                        "content":
                            error_message,

                        "sources":
                            [],
                    }
                )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Built with Python • Streamlit • FastAPI • OpenAI • "
    "Weaviate • Hybrid Retrieval-Augmented Generation"
)