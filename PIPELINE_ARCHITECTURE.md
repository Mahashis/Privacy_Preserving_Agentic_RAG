# Privacy-Preserving Agentic RAG - Pipeline Architecture

## Mermaid Flowchart

```mermaid
flowchart TD
    %% User Interaction
    User[("👤 User Question")] --> Ingest{{"📄 Document Ingestion"}}
    
    %% Ingestion Pipeline
    Ingest --> PDFLoad["📚 Load PDFs from data/"]
    PDFLoad --> TextSplit["✂️ Split into chunks\n(chunk_size=1000, overlap=200)"]
    TextSplit --> Embed["🔢 Generate embeddings\n(nomic-embed-text via Ollama)"]
    Embed --> ChromaDB[("🗄️ Store in ChromaDB\nchroma_db/")]
    
    %% Query Pipeline
    User --> Retrieve["🔍 Retrieve top-k chunks\n(TOP_K=5 from ChromaDB)"]
    ChromaDB -.-> Retrieve
    
    %% Local PII Masking
    Retrieve --> Mask["🛡️ Local PII Masking\n(qwen2.5:7b via Ollama)"]
    Mask --> StructuredOutput["📋 Structured JSON Output"]
    StructuredOutput --> MaskedContext["📝 Masked Context"]
    StructuredOutput --> PIIMap["🔐 PII Mapping Dictionary"]
    
    %% Cloud Synthesis
    MaskedContext --> Cloud["☁️ Cloud LLM Synthesis\n(Gemini 1.5 Flash / GPT-4o-mini)"]
    Cloud --> CloudAnswer["💬 Cloud Answer (with tokens)"]
    
    %% Local De-anonymization
    CloudAnswer --> Deanonymize["🔓 Local De-anonymization\n(qwen2.5:7b via Ollama)"]
    PIIMap --> Deanonymize
    Deanonymize --> FinalAnswer["✅ Final Answer (restored PII)"]
    
    %% Privacy Audit
    PIIMap --> Audit["📊 Privacy Audit Panel"]
    MaskedContext -.-> Audit
    CloudAnswer -.-> Audit
    FinalAnswer -.-> Audit
    
    %% Output
    FinalAnswer --> User
    Audit --> User
    
    %% Styling
    classDef local fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef cloud fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef data fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef user fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef audit fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class Ingest,PDFLoad,TextSplit,Embed,ChromaDB,Retrieve,Mask,StructuredOutput,MaskedContext,PIIMap,Deanonymize,FinalAnswer local
    class Cloud,CloudAnswer cloud
    class User data
    class Audit audit
```

## Pipeline Stages

### 1. Document Ingestion (Local Only)
| Step | Description | Config |
|------|-------------|--------|
| Load PDFs | Load all PDFs from `data/` directory | `paths.data_dir` |
| Text Splitting | RecursiveCharacterTextSplitter with overlap | `ingestion.chunk_size`, `ingestion.chunk_overlap`, `ingestion.text_splitter_separators` |
| Embeddings | nomic-embed-text via Ollama | `ingestion.embedding_model` |
| Vector Store | ChromaDB persistent storage | `paths.chroma_dir` |

### 2. Query Pipeline

#### Stage A: Local Retrieval
- Retrieve top-k relevant chunks from ChromaDB
- Config: `retrieval.top_k`, `retrieval.embedding_model`

#### Stage B: Local PII Masking (Privacy Critical)
- Local LLM (qwen2.5:7b) identifies PII entities
- Replaces with tokens: `[ENTITY_TYPE_N]`
- Returns masked context + PII mapping
- Config: `models.local_llm_model`, `pii_masking.entity_types`, `pii_masking.token_format`

#### Stage C: Cloud Synthesis
- Sends ONLY masked context to cloud (Gemini/OpenAI)
- Raw PII never leaves local machine
- Config: `models.cloud_llm_model`, `cloud_providers`

#### Stage D: Local De-anonymization
- Local LLM restores original PII from mapping
- Deterministic token replacement
- Config: `models.local_llm_model`

### 3. Privacy Audit
- Shows detected PII, tokens used, models at each step
- Visual confirmation of privacy protection

## Data Flow Guarantees

| Data Type | Location | Cloud Exposure |
|-----------|----------|----------------|
| Raw documents | Local only | ❌ Never |
| Embeddings | Local ChromaDB | ❌ Never |
| Raw PII values | Local only | ❌ Never |
| Masked context | Sent to cloud | ✅ Tokenized only |
| Cloud answer | Received from cloud | ✅ Tokens only |
| Final answer | Local restored | ❌ Never sent |

## Configuration Reference

All configurable parameters are in `config.json` at the project root:

```json
{
  "paths": { "data_dir", "chroma_dir", "src_dir" },
  "ingestion": { "embedding_model", "chunk_size", "chunk_overlap", "text_splitter_separators" },
  "retrieval": { "top_k", "embedding_model" },
  "models": { "local_llm_model", "cloud_llm_model", "embedding_model" },
  "cloud_providers": { "gemini", "openai" },
  "pii_masking": { "entity_types", "token_format", "output_format" },
  "streamlit": { "page_title", "page_icon", "layout", "initial_sidebar_state", "ollama_base_url" }
}
```