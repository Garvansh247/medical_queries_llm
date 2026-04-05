"""
chat_route.py
-------------
FastAPI router for the /api/chat and /api/clear endpoints.
Keeps routing definitions separate from business logic (controller).
"""

from fastapi import APIRouter, HTTPException, status

from src.controllers.chat_controller import (
    ChatRequest,
    ChatResponse,
    ClearRequest,
    ClearResponse,
    handle_chat,
    handle_clear,
)

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
        "Submit a medical question with an optional session_id for conversational memory. "
        "The RAG pipeline first classifies the question as MEDICAL or NON_MEDICAL. "
        "Non-medical questions receive a static refusal. Medical questions are answered "
        "using retrieved context from trusted medical guidelines, with full chat history."
    ),
)
async def chat(request: ChatRequest):
    """Handle a medical question via the RAG pipeline with session memory."""
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


@router.post(
    "/clear",
    response_model=ClearResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear a chat session",
    description=(
        "Delete the chat history for the given session_id. "
        "The next request with this session_id will start a fresh conversation."
    ),
)
async def clear(request: ClearRequest):
    """Clear the chat history for a specific session."""
    try:
        return await handle_clear(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {exc}",
        ) from exc
