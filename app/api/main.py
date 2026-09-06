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
)

from app.generation.rag_chain import ask_rag
from app.ingestion.ingest import ingest_pdf
from app.logging_config import configure_logging


configure_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Weaviate RAG API",
    description=(
        "Production-style RAG application "
        "using Weaviate and OpenAI"
    ),
    version="1.0.0",
)


UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class QueryRequest(BaseModel):

    question: str = Field(
        min_length=1,
        description=(
            "Question to ask the document knowledge base"
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of document chunks to retrieve"
        ),
    )

    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document ID. "
            "If provided, retrieval searches "
            "only that document."
        ),
    )

    max_distance: float | None = Field(
        default=0.55,
        ge=0.0,
        le=2.0,
        description=(
            "Maximum vector distance allowed. "
            "Lower values require stronger "
            "semantic similarity."
        ),
    )


@app.get("/")
def root():
    return {
        "message": "Weaviate RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask(
    request: QueryRequest,
):
    logger.info(
        "ask_request | document_id=%s | top_k=%s | max_distance=%s",
        request.document_id,
        request.top_k,
        request.max_distance,
    )

    try:
        result = ask_rag(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            max_distance=request.max_distance,
        )

        logger.info(
            "ask_completed | document_id=%s | sources=%s",
            request.document_id,
            len(result.get("sources", [])),
        )

        return result

    except Exception:
        logger.exception(
            "ask_failed | document_id=%s",
            request.document_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process the question "
                "at this time."
            ),
        )


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is missing",
        )

    original_filename = Path(
        file.filename
    ).name

    if not original_filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    document_id = str(
        uuid.uuid4()
    )

    stored_filename = (
        f"{document_id}_{original_filename}"
    )

    file_path = (
        UPLOAD_DIR /
        stored_filename
    )

    logger.info(
        "upload_started | document_id=%s | filename=%s",
        document_id,
        original_filename,
    )

    try:
        with file_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        ingestion_result = ingest_pdf(
            file_path=str(file_path),
            document_id=document_id,
            document_name=original_filename,
        )

        if ingestion_result["duplicate"]:

            if file_path.exists():
                file_path.unlink()

            logger.info(
                "duplicate_document | document_id=%s | filename=%s",
                ingestion_result["document_id"],
                ingestion_result["filename"],
            )

            return {
                "message":
                    "Document already exists",

                "document_id":
                    ingestion_result[
                        "document_id"
                    ],

                "filename":
                    ingestion_result[
                        "filename"
                    ],

                "duplicate":
                    True,
            }

        logger.info(
            "upload_completed | document_id=%s | filename=%s | pages=%s | chunks=%s",
            ingestion_result["document_id"],
            ingestion_result["filename"],
            ingestion_result["pages"],
            ingestion_result["chunks"],
        )

        return {
            "message":
                "Document ingested successfully",

            "document_id":
                ingestion_result[
                    "document_id"
                ],

            "filename":
                ingestion_result[
                    "filename"
                ],

            "pages":
                ingestion_result[
                    "pages"
                ],

            "chunks":
                ingestion_result[
                    "chunks"
                ],

            "duplicate":
                False,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "upload_failed | document_id=%s | filename=%s",
            document_id,
            original_filename,
        )

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