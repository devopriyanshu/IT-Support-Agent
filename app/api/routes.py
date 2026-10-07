import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents, list_indexed_sources, delete_documents_by_source
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)
    thread_id: str | None = Field(default=None, description="Session thread ID for multi-turn conversation memory")


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        thread_id = payload.thread_id or str(uuid.uuid4())
        result = ask(payload.question, thread_id=thread_id)
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
            "thread_id": thread_id,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}


@router.get("/documents")
def list_documents(x_admin_key: str = Header(default="")):
    """List all document sources currently indexed in Pinecone."""
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    try:
        sources = list_indexed_sources()
        return {"indexed_sources": sources, "count": len(sources)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/documents/{filename}")
def delete_document(filename: str, x_admin_key: str = Header(default="")):
    """Delete all Pinecone embeddings whose source path contains the given filename."""
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    try:
        deleted = delete_documents_by_source(filename)
        if deleted == 0:
            raise HTTPException(status_code=404, detail=f"No indexed vectors found for '{filename}'")
        # Also remove the raw file from uploads/ if it exists there
        upload_file = Path(settings.upload_dir) / filename
        removed_file = False
        if upload_file.exists():
            upload_file.unlink()
            removed_file = True
        return {"message": f"Deleted {deleted} vectors for '{filename}'", "file_removed": removed_file}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
