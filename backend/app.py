"""
app.py
------
Entry point for the MediQuery AI backend.

Start the server with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes.chat_route import router as chat_router
from src.services.rag_service import rag_service

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: initialize the RAG pipeline once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code before 'yield' runs on startup; code after runs on shutdown.
    """
    logger.info("Starting MediQuery AI backend...")
    try:
        rag_service.initialize()
        logger.info("RAG pipeline is ready. Server is accepting requests.")
    except Exception as exc:
        logger.error(
            "Failed to initialize RAG pipeline: %s. "
            "The /api/chat endpoint will return 503 until this is resolved.",
            exc,
        )
    yield
    logger.info("Shutting down MediQuery AI backend.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MediQuery AI - Clinical Q&A API",
    description=(
        "A RAG-powered medical question-answering API. "
        "Uses LangChain, ChromaDB, and a configurable LLM to retrieve and "
        "answer clinical questions from trusted medical guidelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware (allow React dev server and production frontend)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React port
        "http://localhost:4173",  # Vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(chat_router)


# ---------------------------------------------------------------------------
# Root route
# ---------------------------------------------------------------------------
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Welcome to MediQuery AI API",
        "docs": "/docs",
        "health": "/api/health",
    }
