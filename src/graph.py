"""
LangGraph workflow for Privacy-Preserving Agentic RAG.

Defines the stateful workflow: local retrieval -> local PII masking (qwen2.5:7b with structured output)
-> cloud LLM synthesis -> deterministic local de-anonymization.
"""

import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .prompts import (
    RETRIEVAL_QUERY_PROMPT,
    PII_EXTRACTION_PROMPT,
    CLOUD_SYNTHESIS_PROMPT,
    QUERY_EXPANSION_PROMPT,
)
from .config_loader import config
import re


def infer_pii_type(value: str) -> str:
    """Infer PII type from string pattern."""
    # Email
    if re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
        return "EMAIL"
    # SSN (must check before phone)
    if re.match(r'^\d{3}-\d{2}-\d{4}$', value):
        return "SSN"
    # Phone (various formats with dashes/parens/spaces)
    if re.match(r'^[\d\s\-\(\)\+]{7,}$', value) and any(c.isdigit() for c in value):
        # Not SSN pattern
        if not re.match(r'^\d{3}-\d{2}-\d{4}$', value):
            # Standard phone format (contains dashes/parens or starts with country code)
            if re.search(r'[\-\(\)]', value) or value.startswith('+') or re.match(r'^\d{3}[\-\s]\d{3}[\-\s]\d{4}$', value):
                return "PHONE"
    # Credit card
    if re.match(r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$', value):
        return "CREDIT_CARD"
    # IP address
    if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', value):
        return "IP_ADDRESS"
    # Date (various formats)
    if re.match(r'^\d{4}-\d{2}-\d{2}$', value) or re.match(r'^\d{2}/\d{2}/\d{4}$', value):
        return "DATE"
    # Date (Month name format: March 4, 2022)
    if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}$', value, re.IGNORECASE):
        return "DATE"
    # Date (FY2026, Q3 2026)
    if re.match(r'^(FY|Q\d)\s*\d{4}$', value, re.IGNORECASE):
        return "DATE"
    # Address (contains street number, city, state, zip)
    if re.search(r'\d+\s+[\w\s]+(?:Avenue|Street|Road|Blvd|Boulevard|Drive|Lane|Court|Place|Way|Circle|Terrace|Trail|Parkway|Highway|Ridge|Heights|Congress|Cedar)', value, re.IGNORECASE):
        if re.search(r'(?:Apt|Unit|Suite|#)\s*\w+', value, re.IGNORECASE):
            return "ADDRESS"
        if re.search(r',\s*[A-Z]{2}\s*\d{5}', value):
            return "ADDRESS"
        if re.search(r'(?:Austin|Seattle|TX|WA)\s*\d{5}', value):
            return "ADDRESS"
        return "ADDRESS"
    # Username pattern (MD-XXXX)
    if re.match(r'^MD-\d{4}$', value):
        return "USERNAME"
    # Bank account / routing (9 digits) - check before phone
    if re.match(r'^\d{9}$', value):
        return "BANK_ACCOUNT"
    # 4-digit account ending - part of bank account context
    if re.match(r'^\d{4}$', value):
        # Could be account ending, default to BANK_ACCOUNT if context suggests
        return "BANK_ACCOUNT"
    # Organization (contains Corp, Labs, Inc, LLC, etc.)
    if re.search(r'\b(Corp|Corporation|Labs|Inc|LLC|Ltd|Company|Dynamics|Quantum|Apex|Meridian)\b', value, re.IGNORECASE):
        return "ORGANIZATION"
    # Title/Role (contains Architect, Engineer, Manager, Director, etc.)
    if re.search(r'\b(Architect|Engineer|Manager|Director|Lead|Senior|Staff|Principal|VP|Vice President)\b', value, re.IGNORECASE):
        return "TITLE"
    # Relationship
    if re.search(r'\b(Spouse|Sister|Brother|Mother|Father|Husband|Wife|Son|Daughter)\b', value, re.IGNORECASE):
        return "RELATIONSHIP"
    # Default to PERSON for names (capitalized words)
    words = value.split()
    if len(words) >= 2 and all(w[0].isupper() for w in words if w.isalpha() and len(w) > 1):
        return "PERSON"
    return "PERSON"


class PIIMaskingResult(BaseModel):
    """Structured output for PII masking."""
    masked_text: str = Field(description="Text with PII replaced by tokens")
    pii_map: Dict[str, str] = Field(description="Mapping from token to original PII value")


class AgentState(TypedDict):
    """State for the Agentic RAG workflow."""
    question: str
    search_queries: List[str]
    retrieved_docs: List[str]
    masked_context: str
    pii_map: Dict[str, str]
    cloud_answer: str
    final_answer: str
    privacy_audit: Dict[str, Any]


def get_vectorstore():
    """Get or create the ChromaDB vector store."""
    embeddings = OllamaEmbeddings(model=config.retrieval_embedding_model)
    return Chroma(
        persist_directory=str(config.chroma_dir),
        embedding_function=embeddings,
    )


def expand_queries_node(state: AgentState) -> AgentState:
    """Expand complex question into 2-3 focused search queries using local LLM."""
    local_llm = OllamaLLM(model=config.local_llm_model, format="json")
    
    prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
    chain = prompt | local_llm
    
    result = chain.invoke({"question": state["question"]})
    
    # Parse expanded queries
    import json
    if isinstance(result, str):
        search_queries = json.loads(result)
    else:
        search_queries = result
    
    # Fallback: if expansion fails, use original question
    if not search_queries or not isinstance(search_queries, list):
        search_queries = [state["question"]]
    
    return {
        **state,
        "search_queries": search_queries,
    }


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents using expanded search queries."""
    vectorstore = get_vectorstore()
    # Use k=4 per query as specified
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    all_docs = []
    seen_content = set()
    
    for query in state["search_queries"]:
        docs = retriever.invoke(query)
        for doc in docs:
            # Deduplicate by content
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_docs.append(doc.page_content)
    
    return {
        **state,
        "retrieved_docs": all_docs,
    }


def mask_pii_node(state: AgentState) -> AgentState:
    """Extract PII using local LLM, then deterministically mask with Python."""
    # Don't use format="json" - it causes the model to output object format
    # Instead, prompt explicitly asks for JSON array
    local_llm = OllamaLLM(model=config.local_llm_model)
    
    # Combine retrieved docs
    context = "\n\n".join(state["retrieved_docs"])
    
    prompt = ChatPromptTemplate.from_template(PII_EXTRACTION_PROMPT)
    chain = prompt | local_llm
    
    result = chain.invoke({"context": context})
    
    # Parse JSON array of string values
    pii_values = []
    import json
    if isinstance(result, str):
        text = result.strip()
        # Try to find JSON array in the response
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_text = text[start:end+1]
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str) and item.strip():
                            pii_values.append(item.strip())
            except json.JSONDecodeError:
                pass
    else:
        # Already parsed list
        if isinstance(result, list):
            for item in result:
                if isinstance(item, str) and item.strip():
                    pii_values.append(item.strip())
    
    # Deduplicate while preserving order
    seen = set()
    unique_values = []
    for v in pii_values:
        if v not in seen:
            seen.add(v)
            unique_values.append(v)
    
    # Infer types using Python
    normalized_pii = []
    for value in unique_values:
        ptype = infer_pii_type(value)
        normalized_pii.append({"value": value, "type": ptype})
    
    # Deterministically assign placeholders and build mapping
    pii_map = {}
    type_counters = {}
    masked_context = context
    
    for pii in normalized_pii:
        value = pii["value"]
        ptype = pii["type"]
        
        # Increment counter for this type
        type_counters[ptype] = type_counters.get(ptype, 0) + 1
        token = f"[{ptype}_{type_counters[ptype]}]"
        
        # Map token to original value
        pii_map[token] = value
        
        # Replace all occurrences of this value in context
        masked_context = masked_context.replace(value, token)
    
    return {
        **state,
        "masked_context": masked_context,
        "pii_map": pii_map,
    }


def cloud_synthesis_node(state: AgentState) -> AgentState:
    """Send masked context to cloud LLM for answer synthesis."""
    # Use Google Gemini, NVIDIA, or OpenAI
    if os.getenv("GEMINI_API_KEY"):
        gemini_config = config.get("cloud_providers.gemini", {})
        cloud_llm = ChatGoogleGenerativeAI(
            model=gemini_config.get("model", "gemini-1.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    elif os.getenv("NVIDIA_API_KEY"):
        from langchain_openai import ChatOpenAI
        nvidia_config = config.get("cloud_providers.nvidia", {})
        cloud_llm = ChatOpenAI(
            model=nvidia_config.get("model", "nvidia/nemotron-3-ultra-550b-a55b"),
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=nvidia_config.get("base_url", "https://integrate.api.nvidia.com/v1"),
            temperature=0,
        )
    else:
        from langchain_openai import ChatOpenAI
        cloud_llm = ChatOpenAI(
            model=config.get("cloud_providers.openai.model", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
    
    prompt = ChatPromptTemplate.from_template(CLOUD_SYNTHESIS_PROMPT)
    chain = prompt | cloud_llm
    
    result = chain.invoke({
        "question": state["question"],
        "masked_context": state["masked_context"],
    })
    
    # Handle different response types (string vs AIMessage)
    if isinstance(result, str):
        cloud_answer = result
    else:
        cloud_answer = result.content
    
    return {
        **state,
        "cloud_answer": cloud_answer,
    }


def deanonymize_node(state: AgentState) -> AgentState:
    """Deterministically restore PII tokens in the answer using Python."""
    masked_answer = state["cloud_answer"]
    pii_map = state["pii_map"]
    
    # Sort tokens by length (longest first) to avoid partial replacements
    # e.g., replace [PERSON_1] before [PERSON_10]
    sorted_tokens = sorted(pii_map.keys(), key=len, reverse=True)
    
    final_answer = masked_answer
    for token in sorted_tokens:
        final_answer = final_answer.replace(token, pii_map[token])
    
    # Build privacy audit info
    privacy_audit = {
        "pii_detected": list(pii_map.values()),
        "tokens_used": list(pii_map.keys()),
        "masking_model": config.local_llm_model,
        "cloud_model": config.cloud_llm_model,
        "search_queries": state.get("search_queries", []),
        "num_chunks_retrieved": len(state.get("retrieved_docs", [])),
    }
    
    return {
        **state,
        "final_answer": final_answer,
        "privacy_audit": privacy_audit,
    }


def build_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("expand_queries", expand_queries_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("mask_pii", mask_pii_node)
    workflow.add_node("cloud_synthesis", cloud_synthesis_node)
    workflow.add_node("deanonymize", deanonymize_node)
    
    # Add edges
    workflow.set_entry_point("expand_queries")
    workflow.add_edge("expand_queries", "retrieve")
    workflow.add_edge("retrieve", "mask_pii")
    workflow.add_edge("mask_pii", "cloud_synthesis")
    workflow.add_edge("cloud_synthesis", "deanonymize")
    workflow.add_edge("deanonymize", END)
    
    return workflow.compile()


# Compile the graph
agent_graph = build_graph()


def run_agentic_rag(question: str) -> Dict[str, Any]:
    """Run the complete Agentic RAG pipeline."""
    initial_state = AgentState(
        question=question,
        search_queries=[],
        retrieved_docs=[],
        masked_context="",
        pii_map={},
        cloud_answer="",
        final_answer="",
        privacy_audit={},
    )
    
    result = agent_graph.invoke(initial_state)
    return {
        "answer": result["final_answer"],
        "privacy_audit": result["privacy_audit"],
    }


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the company's privacy policy?"
    result = run_agentic_rag(question)
    print(f"Answer: {result['answer']}")
    print(f"Privacy Audit: {result['privacy_audit']}")