"""
rag_service.py
--------------
Implements the full Retrieval-Augmented Generation (RAG) pipeline.

Flow:
  1. Load documents from the data directory.
  2. Split documents into chunks.
  3. Embed chunks and store them in ChromaDB (persisted on disk).
  4. On each query: retrieve the top-k relevant chunks, then pass
     them to the LLM to generate a grounded, cited answer.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (reads from .env with sensible defaults)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./src/data")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable clinical assistant. Your role is to answer
medical questions strictly based on the provided context from trusted medical
guidelines. Do not invent or assume any information not present in the context.
If the context does not contain enough information to answer the question, say
"I do not have enough information in the provided guidelines to answer this question."

Context from medical guidelines:
{context}

User's medical question: {question}

Provide a clear, concise, evidence-based answer. Reference the relevant section
or guideline information where appropriate.""",
)

# ---------------------------------------------------------------------------
# RAGService class
# ---------------------------------------------------------------------------


class RAGService:
    """Encapsulates the full RAG pipeline for medical question answering."""

    def __init__(self) -> None:
        self._vectorstore: Optional[Chroma] = None
        self._retriever = None
        self._llm = None
        self._chain = None
        self._is_ready = False

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _build_llm(self):
        """Instantiate the LLM based on the configured provider."""
        if LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is not set. Please add it to your .env file."
                )
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                api_key=OPENAI_API_KEY,
            )

        elif LLM_PROVIDER == "groq":
            from langchain_groq import ChatGroq  # noqa: PLC0415

            if not GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY is not set. Please add it to your .env file."
                )
            return ChatGroq(
                model=GROQ_MODEL,
                temperature=0.2,
                groq_api_key=GROQ_API_KEY,
            )

        else:
            # Default: Ollama (local LLM)
            from langchain_ollama import ChatOllama  # noqa: PLC0415

            return ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

    def _load_documents(self):
        """Load .txt and .pdf files from the data directory."""
        data_path = Path(DATA_DIR)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_path.resolve()}"
            )

        docs = []

        # Load .txt files
        txt_files = list(data_path.rglob("*.txt"))
        for txt_file in txt_files:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs.extend(loader.load())
            logger.info("Loaded text file: %s", txt_file.name)

        # Load .pdf files
        pdf_files = list(data_path.rglob("*.pdf"))
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs.extend(loader.load())
            logger.info("Loaded PDF file: %s", pdf_file.name)

        if not docs:
            raise ValueError(
                f"No .txt or .pdf files found in data directory: {data_path.resolve()}"
            )

        logger.info("Total documents loaded: %d", len(docs))
        return docs

    def _build_vectorstore(self, docs):
        """Chunk documents and store embeddings in ChromaDB."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info("Total chunks created: %d", len(chunks))

        embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("Vector store built and persisted at: %s", CHROMA_PERSIST_DIR)
        return vectorstore

    def _load_existing_vectorstore(self):
        """Load a previously persisted ChromaDB vector store."""
        embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_fn,
        )
        logger.info("Loaded existing vector store from: %s", CHROMA_PERSIST_DIR)
        return vectorstore

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize the RAG pipeline.
        - If a persisted ChromaDB exists, load it (fast start).
        - Otherwise, build it from scratch (first run).
        """
        logger.info("Initializing RAG pipeline...")

        # Check if vector store already exists on disk
        chroma_path = Path(CHROMA_PERSIST_DIR)
        if chroma_path.exists() and any(chroma_path.iterdir()):
            logger.info("Found existing vector store. Loading...")
            self._vectorstore = self._load_existing_vectorstore()
        else:
            logger.info("No existing vector store found. Building from documents...")
            docs = self._load_documents()
            self._vectorstore = self._build_vectorstore(docs)

        # Create retriever (top-4 most relevant chunks)
        self._retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # Build LLM
        self._llm = self._build_llm()

        # Build RAG chain
        def format_docs(retrieved_docs):
            parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get("source", "Unknown source")
                parts.append(f"[{i}] Source: {Path(source).name}\n{doc.page_content}")
            return "\n\n---\n\n".join(parts)

        self._chain = (
            {
                "context": self._retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | MEDICAL_PROMPT
            | self._llm
            | StrOutputParser()
        )

        self._is_ready = True
        logger.info("RAG pipeline initialized successfully.")

    def query(self, question: str) -> dict:
        """
        Run the RAG pipeline for the given medical question.

        Returns:
            {
                "answer": str,           # LLM-generated answer
                "sources": list[str],    # Source file names used
            }
        """
        if not self._is_ready:
            raise RuntimeError(
                "RAG pipeline is not initialized. Call initialize() first."
            )

        # Retrieve source documents for citation
        retrieved_docs = self._retriever.invoke(question)
        sources = list(
            {Path(doc.metadata.get("source", "Unknown")).name for doc in retrieved_docs}
        )

        # Generate answer
        answer = self._chain.invoke(question)

        return {"answer": answer, "sources": sorted(sources)}

    @property
    def is_ready(self) -> bool:
        return self._is_ready


# Singleton instance used across the application lifetime
rag_service = RAGService()
