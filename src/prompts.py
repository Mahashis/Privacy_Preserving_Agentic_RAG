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

Example:
Input: "What is Elena Rostova's base salary, target bonus, and retention grant RSU vesting schedule policy?"
Output: [
  "Elena Rostova base salary target bonus",
  "retention grant RSU vesting schedule policy"
]

User Question: {question}

Search Queries:"""

# Prompt for generating retrieval queries (optional enhancement)
RETRIEVAL_QUERY_PROMPT = """Given the user's question, generate an optimized search query for retrieving relevant documents from a vector database.

User Question: {question}

Optimized Search Query:"""


# Prompt for local PII extraction - LLM only extracts unique PII strings with types
PII_EXTRACTION_PROMPT = """You are a privacy protection system. Your task is to identify all Personally Identifiable Information (PII) in the provided text.

Extract each unique PII entity and its type. Return a JSON array of objects, each with:
- "value": The exact PII string as it appears in the text
- "type": The PII type (PERSON, EMAIL, PHONE, ADDRESS, ORGANIZATION, DATE, SSN, CREDIT_CARD, BANK_ACCOUNT, MEDICAL_RECORD, IP_ADDRESS, USERNAME, LICENSE_PLATE, PASSPORT, DRIVER_LICENSE)

Only return unique values (deduplicate identical strings). Order by first appearance in text.

PII types to detect:
- PERSON: Names of people
- EMAIL: Email addresses
- PHONE: Phone numbers
- ADDRESS: Physical addresses
- ORGANIZATION: Company/organization names
- DATE: Specific dates (not general time references)
- SSN: Social Security Numbers
- CREDIT_CARD: Credit card numbers
- BANK_ACCOUNT: Bank account numbers
- MEDICAL_RECORD: Medical record numbers
- IP_ADDRESS: IP addresses
- USERNAME: Usernames/handles
- LICENSE_PLATE: Vehicle license plates
- PASSPORT: Passport numbers
- DRIVER_LICENSE: Driver's license numbers

Return ONLY a JSON array. Example:
[
  {{"value": "John Smith", "type": "PERSON"}},
  {{"value": "Acme Corp", "type": "ORGANIZATION"}},
  {{"value": "john.smith@acme.com", "type": "EMAIL"}},
  {{"value": "555-123-4567", "type": "PHONE"}}
]

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