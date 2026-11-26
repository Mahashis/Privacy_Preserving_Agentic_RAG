"""
LangGraph workflow for Privacy-Preserving Agentic RAG.

Defines the stateful workflow: local retrieval -> local PII masking (local LLM with structured output)
-> cloud LLM synthesis -> deterministic local de-anonymization.
"""

import os
import logging
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from .prompts import (
    RETRIEVAL_QUERY_PROMPT,
    PII_EXTRACTION_PROMPT,
    CLOUD_SYNTHESIS_PROMPT,
    QUERY_EXPANSION_PROMPT,
)
from .config_loader import config


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
    logger.info(f"Expanding query: {state['question']}")
    local_llm = OllamaLLM(model=config.local_llm_model)
    
    prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
    chain = prompt | local_llm
    
    result = chain.invoke({"question": state["question"]})
    logger.debug(f"Query expansion raw result: {result}")
    
    # Parse expanded queries - LLM may return array directly or object with "queries" key
    import json
    if isinstance(result, str):
        # Try to find JSON in the response
        text = result.strip()
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end+1])
            except json.JSONDecodeError:
                parsed = {}
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {}
    else:
        parsed = result
    
    # Handle both formats: direct array or {"queries": [...]}
    if isinstance(parsed, dict) and "queries" in parsed:
        search_queries = parsed["queries"]
    elif isinstance(parsed, list):
        search_queries = parsed
    else:
        search_queries = []
    
    # Handle case where queries are objects with "query" key instead of strings
    cleaned_queries = []
    for q in search_queries:
        if isinstance(q, dict) and "query" in q:
            cleaned_queries.append(q["query"])
        elif isinstance(q, str):
            cleaned_queries.append(q)
    search_queries = cleaned_queries
    logger.info(f"Expanded queries: {search_queries}")
    
    # Fallback: if expansion fails, use original question
    if not search_queries or not isinstance(search_queries, list):
        search_queries = [state["question"]]
    
    return {
        **state,
        "search_queries": search_queries,
    }


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents using expanded search queries."""
    logger.info(f"Retrieving documents for queries: {state['search_queries']}")
    vectorstore = get_vectorstore()
    # Use k from config
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.expansion_top_k})
    
    all_docs = []
    seen_content = set()
    
    for query in state["search_queries"]:
        docs = retriever.invoke(query)
        logger.debug(f"Query '{query}' returned {len(docs)} docs")
        for doc in docs:
            # Deduplicate by content
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_docs.append(doc.page_content)
    
    logger.info(f"Total unique docs retrieved: {len(all_docs)}")
    return {
        **state,
        "retrieved_docs": all_docs,
    }


def mask_pii_node(state: AgentState) -> AgentState:
    """Extract PII using local LLM, then deterministically mask with Python."""
    # Don't use format="json" - it causes the model to output object format
    # Instead, prompt explicitly asks for JSON array
    logger.info(f"Masking PII for {len(state['retrieved_docs'])} docs")
    local_llm = OllamaLLM(model=config.local_llm_model)
    
    # Combine retrieved docs
    context = "\n\n".join(state["retrieved_docs"])
    
    # Build entity types list from config for prompt
    entity_types = config.pii_entity_types
    entity_types_str = ", ".join(entity_types)
    
    # Use config for token format
    token_format = config.pii_token_format  # e.g., "[{TYPE}_{N}]"
    
    prompt = ChatPromptTemplate.from_template(PII_EXTRACTION_PROMPT)
    chain = prompt | local_llm
    
    result = chain.invoke({
        "context": context,
        "entity_types": entity_types_str,
    })
    logger.debug(f"PII extraction raw result: {result[:500] if isinstance(result, str) else result}")
    
    # Parse JSON array of objects with "value" and "type"
    pii_list = []
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
                        if isinstance(item, dict) and "value" in item and "type" in item:
                            value = item["value"].strip() if isinstance(item["value"], str) else str(item["value"]).strip()
                            ptype = item["type"].strip().upper() if isinstance(item["type"], str) else str(item["type"]).strip().upper()
                            # Validate type is in configured entity types
                            if value and ptype and ptype in entity_types:
                                pii_list.append({"value": value, "type": ptype})
            except json.JSONDecodeError:
                pass
    else:
        # Already parsed list
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "value" in item and "type" in item:
                    value = item["value"].strip() if isinstance(item["value"], str) else str(item["value"]).strip()
                    ptype = item["type"].strip().upper() if isinstance(item["type"], str) else str(item["type"]).strip().upper()
                    if value and ptype and ptype in entity_types:
                        pii_list.append({"value": value, "type": ptype})
    
    # Deduplicate by value (keep first occurrence with its type)
    seen = set()
    normalized_pii = []
    for pii in pii_list:
        value = pii["value"]
        if value not in seen:
            seen.add(value)
            normalized_pii.append(pii)
    
    logger.info(f"PII detected: {len(normalized_pii)} entities")
    for pii in normalized_pii:
        logger.debug(f"  {pii['type']}: {pii['value']}")
    
    # Deterministically assign placeholders and build mapping
    pii_map = {}
    type_counters = {}
    masked_context = context
    
    for pii in normalized_pii:
        value = pii["value"]
        ptype = pii["type"]
        
        # Increment counter for this type
        type_counters[ptype] = type_counters.get(ptype, 0) + 1
        
        # Build token using config format
        token = token_format.format(TYPE=ptype, N=type_counters[ptype])
        
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
    logger.info("Synthesizing answer with cloud LLM")
    # Use cloud provider priority from config
    cloud_provider = None
    cloud_llm = None
    
    for provider in config.cloud_provider_priority:
        provider_config = config.get(f"cloud_providers.{provider}", {})
        api_key_env = provider_config.get("api_key_env")
        
        if api_key_env and os.getenv(api_key_env):
            cloud_provider = provider
            
            if provider == "gemini":
                cloud_llm = ChatGoogleGenerativeAI(
                    model=provider_config.get("model"),
                    google_api_key=os.getenv(api_key_env),
                    temperature=provider_config.get("temperature", 0),
                )
            elif provider in ["nvidia", "openai"]:
                from langchain_openai import ChatOpenAI
                cloud_llm = ChatOpenAI(
                    model=provider_config.get("model"),
                    api_key=os.getenv(api_key_env),
                    base_url=provider_config.get("base_url"),
                    temperature=provider_config.get("temperature", 0),
                )
            break
    
    if cloud_llm is None:
        raise RuntimeError("No cloud API key found. Configure one of: " + ", ".join(
            config.get(f"cloud_providers.{p}.api_key_env") for p in config.cloud_provider_priority
        ))
    
    logger.info(f"Using cloud provider: {cloud_provider}")
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
    
    logger.info(f"Cloud answer received ({len(cloud_answer)} chars)")
    return {
        **state,
        "cloud_answer": cloud_answer,
    }


def deanonymize_node(state: AgentState) -> AgentState:
    """Deterministically restore PII tokens in the answer using Python."""
    logger.info("De-anonymizing answer")
    masked_answer = state["cloud_answer"]
    pii_map = state["pii_map"]
    
    # Sort tokens by length (longest first) to avoid partial replacements
    # e.g., replace [PERSON_1] before [PERSON_10]
    sorted_tokens = sorted(pii_map.keys(), key=len, reverse=True)
    
    final_answer = masked_answer
    for token in sorted_tokens:
        final_answer = final_answer.replace(token, pii_map[token])
    
    logger.info(f"Restored {len(pii_map)} PII tokens")
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