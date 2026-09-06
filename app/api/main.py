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


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask the document knowledge base",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
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
    return ask_rag(
        question=request.question,
        top_k=request.top_k,
    )


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is missing",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    unique_name = (
        f"{uuid.uuid4()}_{file.filename}"
    )

    file_path = UPLOAD_DIR / unique_name

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        ingestion_result = ingest_pdf(
            str(file_path)
        )

        return {
            "message": "Document ingested successfully",
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