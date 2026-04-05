"""
rag_service.py
--------------
Implements the full Retrieval-Augmented Generation (RAG) pipeline
with LCEL routing and conversational memory.

Flow:
  1. Load documents from the data directory.
  2. Split documents into chunks.
  3. Embed chunks and store them in ChromaDB (persisted on disk).
  4. On each query:
     a. Classify the question as MEDICAL or NON_MEDICAL using a routing chain.
     b. If NON_MEDICAL, return a static refusal message immediately.
     c. If MEDICAL, retrieve the top-k relevant chunks and pass them with the
        full chat history to the LLM to generate a grounded, cited answer.

Key LangChain concepts used here (CampusX / standard LCEL style):
  - LCEL (LangChain Expression Language): chains built with the `|` pipe operator.
    Classic syntax: chain = prompt | llm | output_parser
  - StrOutputParser: converts the LLM's AIMessage reply into a plain Python string.
  - BaseOutputParser: custom parser to extract a domain label from LLM output.
  - RunnableWithMessageHistory: automatically injects and saves chat history per session.
  - InMemoryChatMessageHistory: stores per-session message history in RAM.
  - ChatGroq: free, cloud-hosted LLM from Groq (llama3-8b-8192) — no GPU needed.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (reads from .env with sensible defaults)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./src/data")
# LLM provider: "groq" (default, free), "openai", or "huggingface"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq (free cloud API — strongly recommended for students)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# OpenAI (paid)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# HuggingFace Endpoint (free via API token)
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")

# ---------------------------------------------------------------------------
# Session store (Conversational Memory)
# ---------------------------------------------------------------------------
# Maps session_id (str) → ChatMessageHistory so each user conversation is
# remembered independently across multiple /chat requests.
session_store: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the chat history for a session, creating it if it doesn't exist."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


# ---------------------------------------------------------------------------
# LCEL Routing – Domain Classification
# ---------------------------------------------------------------------------

# A short prompt that asks the model to output exactly one word: MEDICAL or NON_MEDICAL.
_CLASSIFY_TEMPLATE = (
    "Does the following question relate to medicine, health, symptoms, diseases, "
    "treatments, or drugs?\n"
    "Reply with exactly one word: MEDICAL or NON_MEDICAL.\n\n"
    "Question: {input}\n"
    "Answer:"
)
CLASSIFY_PROMPT = PromptTemplate.from_template(_CLASSIFY_TEMPLATE)


class DomainClassifierParser(BaseOutputParser[str]):
    """
    Custom OutputParser for the domain classification step.

    Parses the raw LLM text (e.g. ' NON_MEDICAL\n') and returns a clean
    string: either "MEDICAL" or "NON_MEDICAL".

    CampusX concept: BaseOutputParser lets you define exactly how to
    transform the LLM's raw text output into any Python type you need.
    """

    def parse(self, text: str) -> str:
        # Normalize to uppercase and strip whitespace / punctuation
        cleaned = text.strip().upper()
        if "NON_MEDICAL" in cleaned:
            return "NON_MEDICAL"
        return "MEDICAL"


# ---------------------------------------------------------------------------
# RAG Prompt Template (with chat history for conversational memory)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a clinical assistant that ONLY answers medical questions.\n\n"
    "Follow these steps strictly:\n"
    "Step 1 - Fix spelling: Silently correct any spelling or typing mistakes "
    "in the question before proceeding "
    "(for example, 'pian in my ice' should be understood as 'pain in my eyes').\n"
    "Step 2 - Answer: Answer the corrected medical question using ONLY the "
    "information provided in the Context below. Do not invent or assume any "
    "information. If the context does not contain enough information to answer "
    "the question, say: 'I do not have enough information in the provided "
    "guidelines to answer this question.'\n\n"
    "Context:\n{context}"
)

# MessagesPlaceholder injects the full chat_history list into the prompt so
# the model can remember previous turns within the same session.
MEDICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),  # injected by RunnableWithMessageHistory
    ("human", "{input}"),
])


def _format_docs(docs: List) -> str:
    """
    Format a list of retrieved Document objects into a single string.
    Each chunk is labelled with its source file so the LLM can cite it.

    This string is passed as the "context" key when invoking the RAG chain,
    filling in the {context} placeholder in MEDICAL_PROMPT.
    """
    return "\n\n".join(
        f"[Source: {Path(doc.metadata.get('source', 'Unknown')).name}]\n{doc.page_content}"
        for doc in docs
    )

# ---------------------------------------------------------------------------
# RAGService class
# ---------------------------------------------------------------------------


class RAGService:
    """
    Encapsulates the full RAG pipeline for medical question answering.

    LangChain concepts used (CampusX LCEL style):
    - _classify_chain    : chain = CLASSIFY_PROMPT | llm | DomainClassifierParser()
    - _chain_with_history: chain = MEDICAL_PROMPT | llm | StrOutputParser()
                           wrapped inside RunnableWithMessageHistory for per-session memory.
                           Context (retrieved docs) and input are passed together on invoke.
    - session_store      : global dict mapping session_id → ChatMessageHistory.
    """

    def __init__(self) -> None:
        self._vectorstore: Optional[Chroma] = None
        self._retriever = None
        self._llm = None
        self._classify_chain = None    # LCEL routing chain
        self._chain_with_history = None  # RAG chain + memory
        self._is_ready = False

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _build_llm(self):
        """
        Instantiate the LLM based on the LLM_PROVIDER environment variable.

        - groq        : ChatGroq (free cloud API, default — recommended for students)
        - openai      : ChatOpenAI (paid cloud API)
        - huggingface : HuggingFaceEndpoint (free cloud API via HF Hub token)

        All three behave identically in an LCEL chain: prompt | llm | output_parser
        """
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

        elif LLM_PROVIDER == "huggingface":
            from langchain_huggingface import HuggingFaceEndpoint  # noqa: PLC0415

            if not HUGGINGFACEHUB_API_TOKEN:
                raise ValueError(
                    "HUGGINGFACEHUB_API_TOKEN is not set. Please add it to your .env file."
                )
            return HuggingFaceEndpoint(
                repo_id=HF_MODEL,
                temperature=0.2,
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            )

        else:
            # Default: Groq (free, cloud, fast — no GPU or local install needed)
            from langchain_groq import ChatGroq  # noqa: PLC0415

            if not GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY is not set. Please add it to your .env file.\n"
                    "Get a free API key at: https://console.groq.com"
                )
            return ChatGroq(
                model=GROQ_MODEL,
                temperature=0.2,
                groq_api_key=GROQ_API_KEY,
            )

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

        # Load .pdf files (PyPDFLoader kept intact as required)
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

        Builds two LCEL chains (CampusX style):
          1. _classify_chain    : CLASSIFY_PROMPT | llm | DomainClassifierParser()
          2. _chain_with_history: MEDICAL_PROMPT | llm | StrOutputParser()
               wrapped with RunnableWithMessageHistory for conversational memory.
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

        # ------------------------------------------------------------------
        # Chain 1: Domain Classification (CampusX LCEL style)
        # chain = prompt | llm | output_parser
        # ------------------------------------------------------------------
        # Input : {"input": "<user question>"}
        # Output: "MEDICAL" or "NON_MEDICAL"
        self._classify_chain = CLASSIFY_PROMPT | self._llm | DomainClassifierParser()

        # ------------------------------------------------------------------
        # Chain 2: RAG chain with Conversational Memory (CampusX LCEL style)
        # chain = prompt | llm | StrOutputParser()
        # ------------------------------------------------------------------
        # The context (retrieved documents) is formatted and injected into
        # the input dict in query() before invoking the chain, so the chain
        # itself is the classic: prompt | llm | output_parser
        # Input dict keys expected by MEDICAL_PROMPT:
        #   - "input"        : the user's question (from invoke call)
        #   - "context"      : formatted retrieved documents (from invoke call)
        #   - "chat_history" : previous messages (injected by RunnableWithMessageHistory)
        rag_chain = MEDICAL_PROMPT | self._llm | StrOutputParser()

        # RunnableWithMessageHistory wraps rag_chain so that every call
        # automatically loads the session's past messages into "chat_history"
        # and saves the new exchange afterwards.
        self._chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,        # function returning the right history object
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        self._is_ready = True
        logger.info("RAG pipeline initialized successfully.")

    def query(self, question: str, session_id: str) -> dict:
        """
        Run the full pipeline for the given question and session.

        Step 1 – LCEL Routing (Domain Check):
            classify_chain.invoke({"input": question})
            → "MEDICAL" or "NON_MEDICAL"
            If NON_MEDICAL, immediately return the refusal message.

        Step 2 – Source Retrieval:
            Fetch the top-k document chunks from ChromaDB for source citation.

        Step 3 – RAG with Memory:
            chain_with_history.invoke({"input": question}, config={...})
            → plain answer string (from StrOutputParser)
            RunnableWithMessageHistory automatically loads and saves chat history.

        Returns:
            {
                "answer" : str,        # LLM-generated answer
                "sources": list[str],  # source file names cited
            }
        """
        if not self._is_ready:
            raise RuntimeError(
                "RAG pipeline is not initialized. Call initialize() first."
            )

        # ------------------------------------------------------------------
        # Step 1: LCEL Routing — classify the question domain
        # ------------------------------------------------------------------
        domain = self._classify_chain.invoke({"input": question})
        logger.info("Domain classification for question: %s", domain)

        # Route NON_MEDICAL questions to a static refusal (no LLM call needed)
        if domain == "NON_MEDICAL":
            return {
                "answer": "I can answer only medical related problems.",
                "sources": [],
            }

        # ------------------------------------------------------------------
        # Step 2: Single retrieval — get docs for both context and citation
        # ------------------------------------------------------------------
        docs = self._retriever.invoke(question)
        sources = sorted(
            {Path(doc.metadata.get("source", "Unknown")).name for doc in docs}
        )

        # ------------------------------------------------------------------
        # Step 3: RAG chain with Conversational Memory
        # ------------------------------------------------------------------
        # We pass "context" (formatted retrieved docs) together with "input"
        # in the same invoke call. RunnableWithMessageHistory then injects
        # "chat_history" automatically, giving MEDICAL_PROMPT all three keys.
        # The chain returns a plain string (StrOutputParser: AIMessage → str).
        answer = self._chain_with_history.invoke(
            {"input": question, "context": _format_docs(docs)},
            config={"configurable": {"session_id": session_id}},
        )

        return {"answer": answer, "sources": sources}

    def clear_session(self, session_id: str) -> bool:
        """
        Delete the chat history for a given session_id from the session_store.
        Returns True if the session existed and was deleted, False otherwise.
        """
        if session_id in session_store:
            del session_store[session_id]
            logger.info("Cleared session: %s", session_id)
            return True
        return False

    @property
    def is_ready(self) -> bool:
        return self._is_ready


# Singleton instance used across the application lifetime
rag_service = RAGService()
