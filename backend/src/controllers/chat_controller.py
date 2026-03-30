"""
chat_controller.py
------------------
Handles the HTTP request/response logic for the /api/chat endpoint.
Separates the transport layer (FastAPI request/response) from the
business/AI logic (RAG service).
"""

import logging
from pydantic import BaseModel, field_validator

from src.services.rag_service import rag_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty.")
        if len(v) > 2000:
            raise ValueError("Question must not exceed 2000 characters.")
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Controller functions
# ---------------------------------------------------------------------------


async def handle_chat(request: ChatRequest) -> ChatResponse:
    """
    Process a medical question through the RAG pipeline and return
    the AI-generated answer along with the source documents cited.
    """
    logger.info("Received question: %.100s...", request.question)

    try:
        result = rag_service.query(request.question)
    except RuntimeError as exc:
        logger.error("RAG pipeline error: %s", exc)
        raise

    logger.info(
        "Answer generated. Sources: %s", result.get("sources", [])
    )
    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
    )
