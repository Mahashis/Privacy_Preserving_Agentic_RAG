"""
Centralized prompt templates for Privacy-Preserving Agentic RAG.

Contains system and user prompt templates for:
- Local PII redaction (masking)
- Cloud reasoning/synthesis
- Local de-anonymization
"""

# Prompt for query expansion - split complex question into focused sub-queries
QUERY_EXPANSION_PROMPT = """You are a query expansion system. Given a user's complex question, split it into 2-3 focused search queries that together cover all aspects of the original question.

Return a JSON array of strings (2-3 queries). Each query should be concise and target a specific aspect.

Return ONLY the JSON array. No explanations, no markdown, no examples. If unable to expand, return [].

User Question: {question}

Search Queries:"""

# Prompt for generating retrieval queries (optional enhancement)
RETRIEVAL_QUERY_PROMPT = """Given the user's question, generate an optimized search query for retrieving relevant documents from a vector database.

User Question: {question}

Optimized Search Query:"""


# Prompt for local PII extraction - LLM only extracts unique PII strings (Python infers types)
PII_EXTRACTION_PROMPT = """You are a privacy protection system. Your task is to identify ALL Personally Identifiable Information (PII) in the provided text.

Find EVERY person name, organization, date, email, phone, SSN, address, username, bank account, credit card, medical record, IP address, license plate, passport, driver license, etc. Return a JSON array of unique string values. Do NOT include types.

Return ONLY the JSON array. No explanations, no markdown, no extra text. No examples.

Context to analyze:
{context}"""


# Prompt for cloud LLM synthesis (receives masked context only)
CLOUD_SYNTHESIS_PROMPT = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. The context has been anonymized - all personally identifiable information has been replaced with tokens like [PERSON_1], [EMAIL_1], etc. 

Do NOT attempt to guess or infer the original values of these tokens. Use them as-is in your response. If the context doesn't contain enough information to answer, say so clearly.

User Question: {question}

Anonymized Context:
{masked_context}

Answer:"""


# Prompt for local de-anonymization (deterministic restoration)
DEANONYMIZATION_PROMPT = """You are a de-anonymization system. Your task is to restore the original PII values in the answer by replacing tokens with their corresponding values from the provided mapping.

Replace each token in the masked answer with its original value from the PII map. Do this deterministically - each token must be replaced with exactly its mapped value.

Masked Answer: {masked_answer}

PII Mapping: {pii_map}

Restored Answer:"""