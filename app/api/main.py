from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import logging
import shutil
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from app.generation.rag_chain import ask_rag
from app.ingestion.ingest import ingest_pdf
from app.logging_config import configure_logging
from app.retrieval.weaviate_client import (
    close_shared_weaviate_client,
)


# ============================================================
# Logging
# ============================================================

configure_logging()

logger = logging.getLogger(__name__)


# ============================================================
# FastAPI lifespan
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Manage process-level application resources.

    The shared Weaviate retrieval client is created lazily
    when the first retrieval request is made.

    On application shutdown, the shared Weaviate connection
    is closed cleanly.
    """

    logger.info(
        "fastapi_application_started"
    )

    try:
        yield

    finally:

        logger.info(
            "fastapi_application_shutting_down"
        )

        close_shared_weaviate_client()

        logger.info(
            "fastapi_application_stopped"
        )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Weaviate RAG API",
    description=(
        "Production-style Retrieval-Augmented Generation "
        "application using OpenAI and Weaviate."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# ============================================================
# Upload directory
# ============================================================

UPLOAD_DIR = Path(
    "data/uploads"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Request model
# ============================================================

class QueryRequest(
    BaseModel,
):

    question: str = Field(
        min_length=1,
        description=(
            "Question to ask the document knowledge base."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Maximum number of candidate chunks "
            "to retrieve from Weaviate."
        ),
    )

    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document ID. "
            "If provided, retrieval is restricted "
            "to that document."
        ),
    )

    page_start: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional first page of the retrieval range. "
            "If provided, chunks before this page "
            "are excluded."
        ),
    )

    page_end: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional last page of the retrieval range. "
            "If provided, chunks after this page "
            "are excluded."
        ),
    )

    alpha: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Hybrid search balance. "
            "0 means keyword/BM25 only, "
            "1 means vector/semantic only."
        ),
    )

    min_score: float | None = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum hybrid relevance score required "
            "for a retrieved chunk."
        ),
    )

    rerank: bool = Field(
        default=False,
        description=(
            "Whether to apply the experimental reranker "
            "after retrieval."
        ),
    )

    candidate_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description=(
            "Number of retrieval candidates to consider "
            "before reranking. Used only when rerank=true."
        ),
    )

    @model_validator(
        mode="after"
    )
    def validate_page_range(
        self,
    ) -> "QueryRequest":
        """
        Validate relationships between page filters.
        """

        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_start > self.page_end
        ):
            raise ValueError(
                "page_start cannot be greater than page_end."
            )

        return self


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Weaviate RAG API is running"
        ),
        "version": "1.2.0",
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Ask endpoint
# ============================================================

@app.post("/ask")
def ask(
    request: QueryRequest,
):

    logger.info(
        (
            "ask_request | "
            "document_id=%s | "
            "page_start=%s | "
            "page_end=%s | "
            "top_k=%s | "
            "candidate_k=%s | "
            "rerank=%s | "
            "alpha=%s | "
            "min_score=%s"
        ),
        request.document_id,
        request.page_start,
        request.page_end,
        request.top_k,
        request.candidate_k,
        request.rerank,
        request.alpha,
        request.min_score,
    )

    try:

        result = ask_rag(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            page_start=request.page_start,
            page_end=request.page_end,
            alpha=request.alpha,
            min_score=request.min_score,
            rerank=request.rerank,
            candidate_k=request.candidate_k,
        )

        trace = result.get(
            "trace",
            {},
        )

        metadata = trace.get(
            "metadata",
            {},
        )

        logger.info(
            (
                "ask_completed | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s | "
                "sources=%s | "
                "reranked=%s | "
                "request_id=%s | "
                "total_duration_ms=%s"
            ),
            request.document_id,
            request.page_start,
            request.page_end,
            len(
                result.get(
                    "sources",
                    [],
                )
            ),
            result.get(
                "reranked"
            ),
            trace.get(
                "request_id"
            ),
            trace.get(
                "total_duration_ms"
            ),
        )

        logger.info(
            (
                "ask_trace_summary | "
                "page_filtered=%s | "
                "retrieved_pages=%s | "
                "embedding_cache_hit=%s | "
                "relevance_gate_called=%s | "
                "relevant=%s | "
                "fallback_reason=%s"
            ),
            metadata.get(
                "retrieval_page_filtered"
            ),
            metadata.get(
                "retrieved_pages"
            ),
            metadata.get(
                "embedding_cache_hit"
            ),
            metadata.get(
                "relevance_gate_called"
            ),
            metadata.get(
                "relevant"
            ),
            metadata.get(
                "fallback_reason"
            ),
        )

        return result

    except ValueError as error:

        logger.warning(
            (
                "ask_validation_failed | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s | "
                "error=%s"
            ),
            request.document_id,
            request.page_start,
            request.page_end,
            error,
        )

        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        )

    except Exception:

        logger.exception(
            (
                "ask_failed | "
                "document_id=%s | "
                "page_start=%s | "
                "page_end=%s"
            ),
            request.document_id,
            request.page_start,
            request.page_end,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process the question "
                "at this time."
            ),
        )


# ============================================================
# Upload endpoint
# ============================================================

@app.post(
    "/documents/upload"
)
async def upload_document(
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "File name is missing"
            ),
        )

    # Prevent path traversal such as:
    #
    # ../../malicious.pdf

    original_filename = Path(
        file.filename
    ).name

    # --------------------------------------------------------
    # Validate PDF extension
    # --------------------------------------------------------

    if not original_filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files are supported"
            ),
        )

    # --------------------------------------------------------
    # Generate unique document ID
    # --------------------------------------------------------

    document_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # Create local storage filename
    # --------------------------------------------------------

    stored_filename = (
        f"{document_id}_{original_filename}"
    )

    file_path = (
        UPLOAD_DIR
        / stored_filename
    )

    logger.info(
        (
            "upload_started | "
            "document_id=%s | "
            "filename=%s"
        ),
        document_id,
        original_filename,
    )

    try:

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        with file_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # ----------------------------------------------------
        # Ingest PDF
        # ----------------------------------------------------

        ingestion_result = ingest_pdf(
            file_path=str(
                file_path
            ),
            document_id=document_id,
            document_name=original_filename,
        )

        # ----------------------------------------------------
        # Duplicate document
        # ----------------------------------------------------

        if ingestion_result[
            "duplicate"
        ]:

            # Remove unnecessary duplicate physical file.

            if file_path.exists():

                file_path.unlink()

            logger.info(
                (
                    "duplicate_document | "
                    "document_id=%s | "
                    "filename=%s"
                ),
                ingestion_result[
                    "document_id"
                ],
                ingestion_result[
                    "filename"
                ],
            )

            return {
                "message": (
                    "Document already exists"
                ),

                "document_id": (
                    ingestion_result[
                        "document_id"
                    ]
                ),

                "filename": (
                    ingestion_result[
                        "filename"
                    ]
                ),

                "pages": (
                    ingestion_result.get(
                        "pages"
                    )
                ),

                "chunks": (
                    ingestion_result.get(
                        "chunks"
                    )
                ),

                "duplicate": True,
            }

        # ----------------------------------------------------
        # Successful ingestion
        # ----------------------------------------------------

        logger.info(
            (
                "upload_completed | "
                "document_id=%s | "
                "filename=%s | "
                "pages=%s | "
                "chunks=%s"
            ),
            ingestion_result[
                "document_id"
            ],
            ingestion_result[
                "filename"
            ],
            ingestion_result[
                "pages"
            ],
            ingestion_result[
                "chunks"
            ],
        )

        return {
            "message": (
                "Document ingested successfully"
            ),

            "document_id": (
                ingestion_result[
                    "document_id"
                ]
            ),

            "filename": (
                ingestion_result[
                    "filename"
                ]
            ),

            "pages": (
                ingestion_result[
                    "pages"
                ]
            ),

            "chunks": (
                ingestion_result[
                    "chunks"
                ]
            ),

            "duplicate": False,
        }

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            (
                "upload_failed | "
                "document_id=%s | "
                "filename=%s"
            ),
            document_id,
            original_filename,
        )

        # ----------------------------------------------------
        # Clean up failed upload
        # ----------------------------------------------------

        if file_path.exists():

            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process this document "
                "at this time."
            ),
        )

    finally:

        await file.close()