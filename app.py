"""
Streamlit UI for Privacy-Preserving Agentic RAG.

Features:
- Query input
- Restored answer display
- Collapsible Privacy Audit panel showing PII masking details
"""

import re
import streamlit as st
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.graph import run_agentic_rag
from src.config_loader import config


# Page configuration
st.set_page_config(
    page_title=config.get("streamlit.page_title", "Privacy-Preserving Agentic RAG"),
    page_icon=config.get("streamlit.page_icon", "🔒"),
    layout=config.get("streamlit.layout", "wide"),
    initial_sidebar_state=config.get("streamlit.initial_sidebar_state", "expanded"),
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .audit-panel {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .pii-item {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.25rem;
        font-family: monospace;
        font-size: 0.875rem;
    }
    .token-item {
        display: inline-block;
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        margin: 0.25rem;
        font-family: monospace;
        font-size: 0.875rem;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)


def check_setup():
    """Check if the system is properly set up."""
    issues = []
    
    # Check data directory
    if not config.data_dir.exists() or not list(config.data_dir.glob("*.pdf")):
        issues.append("No PDF files found in `data/` directory. Please add PDF documents.")
    
    # Check chroma_db
    if not config.chroma_dir.exists() or not list(config.chroma_dir.iterdir()):
        issues.append("Vector database not found. Run `python src/ingest.py` to ingest documents.")
    
    # Check API keys
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("NVIDIA_API_KEY"):
        issues.append("No cloud API key found. Set `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `NVIDIA_API_KEY` in `.env` file.")
    
    return issues


def display_privacy_audit(audit: dict):
    """Display the privacy audit panel."""
    with st.expander("🔍 Privacy Audit Panel", expanded=False):
        st.markdown("### Privacy Audit Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PII Detected & Masked:**")
            pii_detected = audit.get("pii_detected", [])
            if pii_detected:
                for pii in pii_detected:
                    st.markdown(f'<span class="pii-item">{pii}</span>', unsafe_allow_html=True)
            else:
                st.markdown("*No PII detected in retrieved context*")
            
            st.markdown("---")
            st.markdown(f"**Masking Model:** {audit.get('masking_model', 'N/A')}")
        
        with col2:
            st.markdown("**Tokens Used:**")
            tokens_used = audit.get("tokens_used", [])
            if tokens_used:
                for token in tokens_used:
                    st.markdown(f'<span class="token-item">{token}</span>', unsafe_allow_html=True)
            else:
                st.markdown("*No tokens used*")
            
            st.markdown("---")
            st.markdown(f"**Cloud Model:** {audit.get('cloud_model', 'N/A')}")
        
        st.markdown("---")
        # Determine cloud provider name for display
        cloud_provider_name = "Gemini/GPT"
        if os.getenv("NVIDIA_API_KEY"):
            cloud_provider_name = "NVIDIA Nemotron-3 Ultra"
        elif os.getenv("GEMINI_API_KEY"):
            cloud_provider_name = "Google Gemini"
        elif os.getenv("OPENAI_API_KEY"):
            cloud_provider_name = "OpenAI GPT"
        
        st.markdown("**How it works:**")
        st.markdown(f"""
        1. **Local Retrieval**: Relevant document chunks retrieved from local ChromaDB
        2. **Local PII Masking**: Local LLM (qwen2.5:7b) identifies and masks PII with tokens
        3. **Cloud Synthesis**: Masked context sent to cloud LLM ({cloud_provider_name}) for answer generation
        4. **Local De-anonymization**: Local LLM restores original PII values in the final answer
        """)
        
        st.info("🛡️ **Privacy Guarantee**: Raw PII never leaves your machine. Only tokenized context is sent to the cloud.")


def main():
    # Header
    st.markdown('<div class="main-header">🔒 Privacy-Preserving Agentic RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Hybrid local-and-cloud RAG with guaranteed PII protection</div>', unsafe_allow_html=True)
    
    # Check setup
    issues = check_setup()
    if issues:
        with st.sidebar:
            st.error("⚠️ Setup Issues:")
            for issue in issues:
                st.markdown(f"- {issue}")
            st.markdown("---")
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### System Status")
        
        # Check Ollama
        try:
            import requests
            ollama_url = config.ollama_base_url
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                st.success("✅ Ollama running")
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if "qwen2.5:7b" in model_names:
                    st.success("✅ qwen2.5:7b available")
                else:
                    st.warning("⚠️ qwen2.5:7b not pulled")
                if any(re.match(r"^nomic-embed-text(:|$)", m) for m in model_names):
                    st.success("✅ nomic-embed-text available")
                else:
                    st.warning("⚠️ nomic-embed-text not pulled")
            else:
                st.error("❌ Ollama not responding")
        except:
            st.error("❌ Ollama not running")
        
        st.markdown("---")
        # Determine cloud provider
        cloud_provider = "Unknown"
        if os.getenv("GEMINI_API_KEY"):
            cloud_provider = "Google Gemini"
        elif os.getenv("NVIDIA_API_KEY"):
            cloud_provider = "NVIDIA Nemotron-3 Ultra"
        elif os.getenv("OPENAI_API_KEY"):
            cloud_provider = "OpenAI"
        
        st.markdown("### Architecture")
        st.markdown(f"""
        - **Local**: Ollama ({config.local_llm_model}, {config.embedding_model})
        - **Vector DB**: ChromaDB (local)
        - **Cloud**: {cloud_provider} ({config.cloud_llm_model})
        - **Orchestration**: LangGraph
        - **UI**: Streamlit
        """)
    
    # Main query interface
    st.markdown("### Ask a Question")
    
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the company's policy on employee data retention?",
        height=100,
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        submit = st.button("🔍 Query", type="primary", use_container_width=True)
    with col2:
        clear = st.button("🗑️ Clear", use_container_width=True)
    
    if clear:
        st.rerun()
    
    if submit and question.strip():
        with st.spinner("Processing with privacy protection..."):
            try:
                result = run_agentic_rag(question.strip())
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(result["answer"])
                
                # Display privacy audit
                display_privacy_audit(result["privacy_audit"])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
    
    elif submit:
        st.warning("Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #94a3b8;'>"
        "Privacy-Preserving Agentic RAG • Built with LangGraph, Ollama, ChromaDB, Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()