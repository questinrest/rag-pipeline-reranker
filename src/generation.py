from langchain_groq import ChatGroq
from src.config import (
    OPENAI_MODEL_GROQ,
    TEMPERATURE,
    MAX_TOKENS,
    MAX_RETRIES
)



llm = ChatGroq(
    model=OPENAI_MODEL_GROQ,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    max_retries=MAX_RETRIES
)