"""
Centralized prompt templates for Privacy-Preserving Agentic RAG.

Contains system and user prompt templates for:
- Local PII redaction (masking)
- Cloud reasoning/synthesis
- Local de-anonymization
"""

# Prompt for generating retrieval queries (optional enhancement)
RETRIEVAL_QUERY_PROMPT = """Given the user's question, generate an optimized search query for retrieving relevant documents from a vector database.

User Question: {question}

Optimized Search Query:"""


# Prompt for local PII masking using structured output
PII_MASKING_PROMPT = """You are a privacy protection system. Your task is to identify and mask all Personally Identifiable Information (PII) in the provided text.

Replace each PII entity with a unique token in the format [ENTITY_TYPE_N] where N is an incrementing number (e.g., [PERSON_1], [EMAIL_2], [PHONE_1], [ADDRESS_1], [ORGANIZATION_1], [DATE_1], [SSN_1], [CREDIT_CARD_1], etc.).

PII types to detect and mask:
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

Return a JSON object with:
- "masked_text": The text with all PII replaced by tokens
- "pii_map": A dictionary mapping each token to its original value

Example:
Input: "John Smith from Acme Corp can be reached at john.smith@acme.com or 555-123-4567."
Output: {{
  "masked_text": "[PERSON_1] from [ORGANIZATION_1] can be reached at [EMAIL_1] or [PHONE_1].",
  "pii_map": {{
    "[PERSON_1]": "John Smith",
    "[ORGANIZATION_1]": "Acme Corp",
    "[EMAIL_1]": "john.smith@acme.com",
    "[PHONE_1]": "555-123-4567"
  }}
}}

Context to mask:
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