"""
Centralized prompt templates for Privacy-Preserving Agentic RAG.

Contains system and user prompt templates for:
- Local PII redaction (masking)
- Cloud reasoning/synthesis
- Local de-anonymization
"""

# Prompt for query expansion - split complex question into focused sub-queries
QUERY_EXPANSION_PROMPT = """Split this question into 2-3 focused search queries. 
Rules:
1. If the question asks about a specific person's compensation, leave, or transfer, create a query with their name and the specific facts needed.
2. ALWAYS include at least one "step-back" policy query that completely removes the person's name and searches for the general company-wide rule, policy, or schedule
Return JSON array only.

Question: {question}

Queries:"""

# Prompt for generating retrieval queries (optional enhancement)
RETRIEVAL_QUERY_PROMPT = """Given the user's question, generate an optimized search query for retrieving relevant documents from a vector database.

User Question: {question}

Optimized Search Query:"""


# Prompt for local PII extraction - LLM identifies PII and its type
# Uses {entity_types} placeholder which will be filled from config
PII_EXTRACTION_PROMPT = """You are a privacy protection system. Your task is to identify ALL Personally Identifiable Information (PII) in the provided text.

Find EVERY occurrence of the following PII types: {entity_types}

Return a JSON array of objects, each with "value" and "type" fields.

Type Definitions:
- PERSON: Full names of individuals (e.g., Elena Rostova, Marcus Vance)
- ORGANIZATION: Company names, institutions, corp, labs, inc, etc.
- DATE: Calendar dates, fiscal years (FY2026), quarters (Q3 2026), hire dates
- EMAIL: Email addresses
- PHONE: Phone numbers with dashes, parentheses, spaces
- ADDRESS: Physical addresses with street, city, state, zip
- SSN: Social Security Numbers (XXX-XX-XXXX format)
- CREDIT_CARD: Credit card numbers
- BANK_ACCOUNT: Bank routing numbers, account numbers
- MEDICAL_RECORD: Medical record numbers
- IP_ADDRESS: IP addresses
- USERNAME: Employee IDs, usernames (e.g., MD-XXXX format)
- LICENSE_PLATE: Vehicle license plates
- PASSPORT: Passport numbers
- DRIVER_LICENSE: Driver license numbers
- TITLE: Job titles, roles (Architect, Engineer, Manager, Director, etc.)
- RELATIONSHIP: Family relationships (Spouse, Sister, Brother, etc.)

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