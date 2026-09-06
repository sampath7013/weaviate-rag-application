from pathlib import Path
import shutil
import uuid

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)
from pydantic import BaseModel, Field

from app.generation.rag_chain import ask_rag
from app.ingestion.ingest import ingest_pdf


app = FastAPI(
    title="Weaviate RAG API",
    description="RAG application using Weaviate and OpenAI",
    version="1.0.0",
)


# Folder where uploaded PDFs are stored
UPLOAD_DIR = Path("data/uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Request model for /ask endpoint
class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask the document knowledge base",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve",
    )

    document_id: str | None = Field(
        default=None,
        description=(
            "Optional document ID. "
            "If provided, retrieval will search only that document."
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
def ask(request: QueryRequest):
    try:
        return ask_rag(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(exc)}",
        )


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is missing",
        )

    # Allow only PDF files
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    # Create a unique ID for this document
    document_id = str(uuid.uuid4())

    # Create a unique filename on disk
    stored_filename = (
        f"{document_id}_{file.filename}"
    )

    file_path = UPLOAD_DIR / stored_filename

    try:
        # Save uploaded PDF
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        # Ingest PDF into Weaviate
        ingestion_result = ingest_pdf(
            str(file_path),
            document_id=document_id,
        )

        return {
            "message": "Document ingested successfully",
            "document_id": document_id,
            "filename": file.filename,
            "pages": ingestion_result["pages"],
            "chunks": ingestion_result["chunks"],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(exc)}",
        )

    finally:
        await file.close()