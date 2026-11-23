# Privacy-Preserving Agentic RAG

A modular repository for a hybrid local-and-cloud Agentic RAG pipeline with guaranteed PII protection.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Question                               │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL : Retrieve relevant chunks from ChromaDB                   │
│  (nomic-embed-text embeddings via Ollama)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL: PII Masking with qwen2.5:7b (structured output)         │
│  - Detects: Names, Emails, Phones, Addresses, SSNs, etc.        │
│  - Replaces with tokens: [PERSON_1], [EMAIL_1], [PHONE_1]       │
│  - Returns masked context + PII mapping                          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLOUD: Answer synthesis (Gemini 1.5 Flash / GPT-4o-mini)       │
│  - Receives ONLY masked context (no raw PII)                    │
│  - Generates answer using tokens as-is                          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL: De-anonymization with qwen2.5:7b                         │
│  - Deterministically restores PII from mapping                  │
│  - Final answer with original values                            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Answer + Privacy Audit Panel                            │
└─────────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
privacy-agentic-rag/
├── data/                   # Sample PDFs (mock HR policies, legal contracts)
├── chroma_db/              # Local persistent vector database (auto-generated)
├── src/
│   ├── ingest.py           # PDF chunking + local embeddings
│   ├── graph.py            # LangGraph workflow (Retrieve → Mask → Cloud → Unmask)
│   └── prompts.py          # System prompts for local masking & cloud synthesis
├── app.py                  # Streamlit UI with live "Privacy Audit" panel
├── .env.example            # Template for cloud API keys
├── .gitignore              # Excludes .env, chroma_db, __pycache__
└── requirements.txt        # Python dependencies
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running locally
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
3. **Cloud API Key** (choose one):
   - Google Gemini: https://makersuite.google.com/app/apikey
   - OpenAI: https://platform.openai.com/api-keys

## Quick Start

### 1. Clone and Install

```bash
cd privacy-agentic-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY or OPENAI_API_KEY
```

### 3. Add Documents

Place PDF files in the `data/` directory:
```bash
mkdir -p data
# Copy your PDF files to data/
```

### 4. Ingest Documents

```bash
python src/ingest.py
```

This will:
- Load all PDFs from `data/`
- Split into overlapping chunks (1000 chars, 200 overlap)
- Generate embeddings using `nomic-embed-text` via Ollama
- Store in local ChromaDB at `chroma_db/`

### 5. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Enter a question** about your documents
2. **Click "Query"** - the system will:
   - Retrieve relevant chunks locally
   - Mask PII using local LLM
   - Send masked context to cloud LLM
   - Restore PII locally
   - Display answer + privacy audit
3. **Expand "Privacy Audit Panel"** to see:
   - What PII was detected and masked
   - Tokens used for masking
   - Models used at each step

## Privacy Guarantees

| Data | Location | Protection |
|------|----------|------------|
| Raw documents | Local only | Never leaves machine |
| Embeddings | Local ChromaDB | Local only |
| PII detection | Local (qwen2.5:7b) | Local only |
| **Raw PII values** | **Local only** | **Never sent to cloud** |
| Masked context | Sent to cloud | PII replaced with tokens |
| Cloud answer | Received from cloud | Contains only tokens |
| Final answer | Local de-anonymization | Restored locally |

## Customization

### Change Models

Edit `.env`:
```bash
LOCAL_LLM_MODEL=qwen2.5:7b
EMBEDDING_MODEL=nomic-embed-text
CLOUD_LLM_MODEL=gemini-1.5-flash
```

### Adjust Chunking

Edit `src/ingest.py`:
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### Modify Retrieval

Edit `src/graph.py`:
```python
TOP_K = 5  # Number of chunks to retrieve
```

### Add PII Types

Edit `src/prompts.py` - `PII_MASKING_PROMPT` to add more entity types.

## Troubleshooting

### "Ollama not running"
```bash
ollama serve
```

### "Model not found"
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### "No PDF files found"
Add PDF files to `data/` directory and re-run `python src/ingest.py`

### "Vector database not found"
Run `python src/ingest.py` to create the ChromaDB

### "API key not set"
Add your API key to `.env` file

## License

MIT License - Feel free to use and modify for your projects.