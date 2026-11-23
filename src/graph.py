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
    PII_MASKING_PROMPT,
    CLOUD_SYNTHESIS_PROMPT,
    DEANONYMIZATION_PROMPT,
)
from .config_loader import config


class PIIMaskingResult(BaseModel):
    """Structured output for PII masking."""
    masked_text: str = Field(description="Text with PII replaced by tokens")
    pii_map: Dict[str, str] = Field(description="Mapping from token to original PII value")


class AgentState(TypedDict):
    """State for the Agentic RAG workflow."""
    question: str
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


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents from local vector store."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.top_k})
    
    docs = retriever.invoke(state["question"])
    retrieved_texts = [doc.page_content for doc in docs]
    
    return {
        **state,
        "retrieved_docs": retrieved_texts,
    }


def mask_pii_node(state: AgentState) -> AgentState:
    """Mask PII in retrieved context using local LLM with structured output."""
    local_llm = OllamaLLM(model=config.local_llm_model, format="json")
    
    # Combine retrieved docs
    context = "\n\n".join(state["retrieved_docs"])
    
    prompt = ChatPromptTemplate.from_template(PII_MASKING_PROMPT)
    chain = prompt | local_llm
    
    result = chain.invoke({"context": context})
    
    # Parse structured output
    import json
    parsed = json.loads(result)
    
    return {
        **state,
        "masked_context": parsed["masked_text"],
        "pii_map": parsed["pii_map"],
    }


def cloud_synthesis_node(state: AgentState) -> AgentState:
    """Send masked context to cloud LLM for answer synthesis."""
    # Use Google Gemini or OpenAI
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if os.getenv("GEMINI_API_KEY"):
        cloud_llm = ChatGoogleGenerativeAI(
            model=config.cloud_llm_model,
            google_api_key=api_key,
            temperature=0,
        )
    else:
        from langchain_openai import ChatOpenAI
        cloud_llm = ChatOpenAI(
            model=config.get("cloud_providers.openai.model", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0,
        )
    
    prompt = ChatPromptTemplate.from_template(CLOUD_SYNTHESIS_PROMPT)
    chain = prompt | cloud_llm
    
    result = chain.invoke({
        "question": state["question"],
        "masked_context": state["masked_context"],
    })
    
    return {
        **state,
        "cloud_answer": result.content,
    }


def deanonymize_node(state: AgentState) -> AgentState:
    """Deterministically restore PII tokens in the answer."""
    local_llm = OllamaLLM(model=config.local_llm_model)
    
    prompt = ChatPromptTemplate.from_template(DEANONYMIZATION_PROMPT)
    chain = prompt | local_llm
    
    result = chain.invoke({
        "masked_answer": state["cloud_answer"],
        "pii_map": str(state["pii_map"]),
    })
    
    # Build privacy audit info
    privacy_audit = {
        "pii_detected": list(state["pii_map"].values()),
        "tokens_used": list(state["pii_map"].keys()),
        "masking_model": config.local_llm_model,
        "cloud_model": config.cloud_llm_model,
    }
    
    return {
        **state,
        "final_answer": result.content,
        "privacy_audit": privacy_audit,
    }


def build_graph() -> StateGraph:
    """Build the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("mask_pii", mask_pii_node)
    workflow.add_node("cloud_synthesis", cloud_synthesis_node)
    workflow.add_node("deanonymize", deanonymize_node)
    
    # Add edges
    workflow.set_entry_point("retrieve")
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