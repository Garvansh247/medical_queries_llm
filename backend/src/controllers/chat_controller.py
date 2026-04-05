"""
chat_controller.py
------------------
Handles the HTTP request/response logic for the /api/chat and /api/clear endpoints.
Separates the transport layer (FastAPI request/response) from the
business/AI logic (RAG service).
"""

import logging
import uuid
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator

from src.services.rag_service import rag_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    # session_id is optional; if omitted the server generates a new one
    session_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty.")
        if len(v) > 2000:
            raise ValueError("Question must not exceed 2000 characters.")
        return v

    @model_validator(mode="after")
    def ensure_session_id(self) -> "ChatRequest":
        """Generate a session_id if the client did not provide one."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        return self


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    # Echo back the session_id so the client can reuse it in the next request
    session_id: str


class ClearRequest(BaseModel):
    session_id: str

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("session_id must not be empty.")
        return v


class ClearResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Controller functions
# ---------------------------------------------------------------------------


async def handle_chat(request: ChatRequest) -> ChatResponse:
    """
    Process a medical question through the RAG pipeline and return
    the AI-generated answer along with the source documents cited.

    The session_id is forwarded to rag_service.query() so that
    RunnableWithMessageHistory can load/save the right chat history.
    """
    logger.info("Received question: %.100s...", request.question)
    logger.info("Session ID: %s", request.session_id)

    try:
        result = rag_service.query(request.question, request.session_id)
    except RuntimeError as exc:
        logger.error("RAG pipeline error: %s", exc)
        raise

    logger.info(
        "Answer generated. Sources: %s", result.get("sources", [])
    )
    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        session_id=request.session_id,
    )


async def handle_clear(request: ClearRequest) -> ClearResponse:
    """
    Delete the chat history for the given session_id from the session_store.
    This effectively starts a fresh conversation for that session.
    """
    logger.info("Clearing session: %s", request.session_id)
    deleted = rag_service.clear_session(request.session_id)
    if deleted:
        return ClearResponse(
            success=True,
            message=f"Session '{request.session_id}' cleared successfully.",
        )
    return ClearResponse(
        success=True,
        message=f"Session '{request.session_id}' was not found (already empty).",
    )
