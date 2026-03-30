# 🩺 MediQuery AI — Clinical Question-Answering Assistant (LLM + RAG)

A full-stack, evidence-based medical question-answering application built with **React + Vite** on the frontend and **FastAPI + LangChain + ChromaDB** on the backend. Ask any clinical question and get a grounded, cited answer powered by Retrieval-Augmented Generation (RAG).

> ⚠️ **Disclaimer:** This project is for educational purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional.

---

## 📁 Project Structure

```
medical_queries_llm/
│
├── frontend/                        # React + Vite + Tailwind CSS
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.jsx          # Chat message list with auto-scroll
│   │   │   └── Message.jsx          # Individual message bubble with citations
│   │   ├── pages/
│   │   │   └── Home.jsx             # Main ChatGPT-style UI page
│   │   ├── services/
│   │   │   └── api.js               # Axios API client (calls backend)
│   │   ├── utils/
│   │   │   └── constants.js         # App-wide constants
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css                # Tailwind CSS import
│   ├── vite.config.js               # Vite config with Tailwind plugin + proxy
│   └── package.json
│
├── backend/                         # Python + FastAPI + LangChain
│   ├── app.py                       # FastAPI entry point
│   ├── requirements.txt
│   ├── .env.example                 # Environment variable template
│   └── src/
│       ├── controllers/
│       │   └── chat_controller.py   # Request/response logic (Pydantic models)
│       ├── routes/
│       │   └── chat_route.py        # FastAPI router (/api/chat, /api/health)
│       ├── services/
│       │   └── rag_service.py       # Full RAG pipeline (LangChain + ChromaDB)
│       └── data/
│           └── sample_medical_guidelines.txt   # Sample medical data (Diabetes, HTN, etc.)
│
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend UI** | React 18, Vite, Tailwind CSS |
| **HTTP Client** | Axios |
| **Backend Server** | Python, FastAPI, Uvicorn |
| **AI Framework** | LangChain |
| **Vector Database** | ChromaDB (persisted locally) |
| **Embedding Model** | `all-MiniLM-L6-v2` via `sentence-transformers` (runs locally, free) |
| **LLM Options** | Ollama (local) / OpenAI API / Groq API |

---

## ⚙️ Prerequisites

Make sure you have these installed on your system:

- **Node.js** v18 or higher — [Download](https://nodejs.org/)
- **Python** 3.10 or higher — [Download](https://www.python.org/)
- **pip** (comes with Python)
- **One of these LLMs** (choose based on your RAM):
  - **Ollama** (recommended, free, local) — [Download Ollama](https://ollama.ai/) — requires 16GB RAM
  - **OpenAI API key** — [Get one here](https://platform.openai.com/) — works on any laptop
  - **Groq API key** — [Get a free key here](https://console.groq.com/) — works on any laptop, very fast

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Garvansh247/medical_queries_llm.git
cd medical_queries_llm
```

### Step 2: Set Up the Backend

```bash
cd backend
```

**2a. Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**2b. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**2c. Configure your LLM provider**

Copy the example env file and edit it:
```bash
cp .env.example .env
```

Open `.env` in any text editor and choose your LLM option:

```env
# --- Option A: Ollama (local, free, needs 16GB RAM) ---
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3

# --- Option B: OpenAI (cloud, needs API key) ---
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-key-here

# --- Option C: Groq (cloud, free tier, very fast) ---
# LLM_PROVIDER=groq
# GROQ_API_KEY=gsk_your-key-here
# GROQ_MODEL=llama3-8b-8192
```

**2d. (If using Ollama) — Pull the model**
```bash
# In a separate terminal
ollama serve

# Pull the Llama 3 model (one-time download ~4.7GB)
ollama pull llama3
```

**2e. Start the FastAPI backend**
```bash
# Make sure you're in the /backend directory
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

On first run, the backend will:
1. Load the medical guidelines from `src/data/`
2. Split them into chunks
3. Create embeddings using the `sentence-transformers` model
4. Save the vector database to `./chroma_db/` (this takes ~30-60 seconds)
5. Start accepting requests

You should see: `RAG pipeline is ready. Server is accepting requests.`

Visit **http://localhost:8000/docs** to see the interactive API documentation.

---

### Step 3: Set Up the Frontend

Open a **new terminal**:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server will start at **http://localhost:5173**.

> The frontend is pre-configured to proxy `/api` requests to `http://localhost:8000`, so CORS is handled automatically in development.

---

### Step 4: Use the Application

1. Open your browser and go to **http://localhost:5173**
2. Type a medical question in the input box, e.g.:
   - *"What is the first-line treatment for Type 2 Diabetes?"*
   - *"What is the standard dosage of Amoxicillin for adults?"*
   - *"What are the diagnostic criteria for hypertension?"*
3. Press **Enter** or click **Send**
4. The AI will retrieve relevant sections from the medical guidelines and generate an evidence-based answer with source citations.

---

## 🧪 API Endpoints

| Method | URL | Description |
|---|---|---|
| `GET` | `/` | Root welcome message |
| `GET` | `/api/health` | Health check + RAG pipeline status |
| `POST` | `/api/chat` | Ask a medical question |

**POST `/api/chat` — Request body:**
```json
{
  "question": "What is the first-line treatment for Type 2 Diabetes?"
}
```

**Response:**
```json
{
  "answer": "The first-line pharmacological agent for Type 2 Diabetes is Metformin...",
  "sources": ["sample_medical_guidelines.txt"]
}
```

---

## 📖 How RAG Works (Explained Simply)

```
User Question
     │
     ▼
[Embed question]  ──►  Convert question to a vector (numbers)
     │
     ▼
[ChromaDB Search]  ──►  Find the top 4 most relevant medical paragraphs
     │
     ▼
[Build Prompt]  ──►  Combine question + retrieved paragraphs
     │
     ▼
[LLM (Ollama/OpenAI/Groq)]  ──►  Generate a grounded answer from the context
     │
     ▼
[Response]  ──►  Return answer + source document names to the UI
```

---

## ➕ Adding Your Own Medical Documents

To add more medical data:
1. Drop any `.txt` or `.pdf` files into `backend/src/data/`
2. Delete the existing vector store: `rm -rf backend/chroma_db/`
3. Restart the backend — it will rebuild the database from scratch

---

## 👥 Team Division (Suggested for 4 Members)

| Member | Responsibility |
|---|---|
| **Member 1 (You - React)** | Frontend — `Home.jsx`, `ChatBox.jsx`, `Message.jsx`, `api.js` |
| **Member 2 (FastAPI expert)** | Backend — `app.py`, `chat_route.py`, `chat_controller.py` |
| **Member 3** | RAG pipeline — `rag_service.py`, ChromaDB, embeddings |
| **Member 4** | Data collection — medical PDFs/guidelines, testing, README |

---

## 📚 What to Study to Understand This Project

| Topic | Resource |
|---|---|
| **LangChain basics** | CampusX GenAI playlist (Videos 1-5) |
| **Document Loaders & Text Splitters** | LangChain docs + CampusX Videos 10-11 |
| **Vector Stores & Embeddings** | CampusX Video 12 |
| **Retrievers & RAG chains** | CampusX Videos 13-15 |
| **Ollama (local LLM)** | Ollama Masterclass (last video in playlist) |
| **FastAPI** | FastAPI official docs (https://fastapi.tiangolo.com/) |

---

## 🔧 Troubleshooting

**"Connection refused" or "Network Error" in the UI?**
→ Make sure the backend is running on port 8000.

**Backend stuck on first run?**
→ It's downloading the embedding model (~90MB) for the first time. Wait ~2 minutes.

**Ollama errors?**
→ Run `ollama serve` in a separate terminal, then `ollama pull llama3`.

**Low RAM / Slow laptop?**
→ Switch to Groq (free, no local compute needed). Set `LLM_PROVIDER=groq` in `.env`.
