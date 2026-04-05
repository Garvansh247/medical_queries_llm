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

Key LangChain concepts used here:
  - LCEL (LangChain Expression Language): chains built with the `|` pipe operator.
  - BaseOutputParser: custom parser to extract domain label from LLM output.
  - RunnableWithMessageHistory: automatically injects and saves chat history.
  - InMemoryChatMessageHistory: stores per-session message history in RAM.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

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

# A very short prompt optimized for TinyLlama (small models need simple instructions).
# It asks the model to output exactly one word: MEDICAL or NON_MEDICAL.
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
# TinyLlama can remember previous turns within the same session.
MEDICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),  # injected by RunnableWithMessageHistory
    ("human", "{input}"),
])

# Document prompt used by create_stuff_documents_chain to format each retrieved chunk
_DOC_PROMPT = PromptTemplate.from_template("Source: {source}\n{page_content}")

# ---------------------------------------------------------------------------
# RAGService class
# ---------------------------------------------------------------------------


class RAGService:
    """
    Encapsulates the full RAG pipeline for medical question answering.

    Concepts for the student:
    - _classify_chain  : LCEL chain that decides MEDICAL vs NON_MEDICAL (routing).
    - _chain_with_history: RunnableWithMessageHistory wraps the RAG chain so that
      previous messages in a session are automatically injected as chat_history.
    - session_store    : global dict mapping session_id → ChatMessageHistory.
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
        Builds two chains:
          1. _classify_chain  : LCEL routing (domain check).
          2. _chain_with_history: RAG chain wrapped with RunnableWithMessageHistory.
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
        # Chain 1: Domain Classification (LCEL Routing)
        # ------------------------------------------------------------------
        # CLASSIFY_PROMPT → LLM → DomainClassifierParser
        # Input : {"input": "<user question>"}
        # Output: "MEDICAL" or "NON_MEDICAL"
        self._classify_chain = CLASSIFY_PROMPT | self._llm | DomainClassifierParser()

        # ------------------------------------------------------------------
        # Chain 2: RAG chain with Conversational Memory
        # ------------------------------------------------------------------
        # Step A: create_stuff_documents_chain combines the retrieved docs into
        #         the MEDICAL_PROMPT and calls the LLM.
        question_answer_chain = create_stuff_documents_chain(
            self._llm, MEDICAL_PROMPT, document_prompt=_DOC_PROMPT
        )
        # Step B: create_retrieval_chain first retrieves relevant chunks and
        #         then passes them to question_answer_chain.
        rag_chain = create_retrieval_chain(self._retriever, question_answer_chain)

        # Step C: RunnableWithMessageHistory wraps rag_chain so that every
        #         call automatically loads the session's past messages into
        #         "chat_history" and saves the new exchange afterwards.
        self._chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,       # function returning the right history object
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        self._is_ready = True
        logger.info("RAG pipeline initialized successfully.")

    def query(self, question: str, session_id: str) -> dict:
        """
        Run the full pipeline for the given question and session.

        Step 1 – LCEL Routing (Domain Check):
            Invoke _classify_chain to decide MEDICAL or NON_MEDICAL.
            If NON_MEDICAL, immediately return the refusal message.

        Step 2 – RAG with Memory:
            Invoke _chain_with_history, which:
              a) retrieves relevant document chunks from ChromaDB,
              b) injects chat_history for the session,
              c) calls TinyLlama and saves the new turn to history.

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
        # Step 2: RAG chain with Conversational Memory
        # ------------------------------------------------------------------
        # config["configurable"]["session_id"] tells RunnableWithMessageHistory
        # which history bucket in session_store to use.
        result = self._chain_with_history.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )

        answer = result["answer"]
        sources = list(
            {
                Path(doc.metadata.get("source", "Unknown")).name
                for doc in result.get("context", [])
            }
        )

        return {"answer": answer, "sources": sorted(sources)}

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
