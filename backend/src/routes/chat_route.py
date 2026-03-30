"""
chat_route.py
-------------
FastAPI router for the /api/chat endpoint.
Keeps routing definitions separate from business logic (controller).
"""

from fastapi import APIRouter, HTTPException, status

from src.controllers.chat_controller import ChatRequest, ChatResponse, handle_chat

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/health", summary="Health check")
async def health_check():
    """Returns the operational status of the API and RAG pipeline."""
    from src.services.rag_service import rag_service  # noqa: PLC0415

    return {
        "status": "ok",
        "rag_pipeline_ready": rag_service.is_ready,
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a medical question",
    description=(
        "Submit a medical question. The RAG pipeline retrieves relevant "
        "context from trusted medical guidelines and generates an evidence-based answer."
    ),
)
async def chat(request: ChatRequest):
    """Handle a medical question via the RAG pipeline."""
    try:
        return await handle_chat(request)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG pipeline is not ready: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        ) from exc
