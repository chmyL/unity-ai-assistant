import threading

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

_embedding_instance = None
_embedding_lock = threading.Lock()


def create_llm(model_name: str = None, temperature: float = None, role: str = "default", **kwargs):
    if role == "coder":
        api_key = Config.OPENAI_API_KEY_2
        base_url = Config.OPENAI_API_BASE_2
        if model_name is None:
            model_name = Config.CODER_LLM_MODEL
        if temperature is None:
            temperature = 0.2
    else:
        api_key = Config.OPENAI_API_KEY_1
        base_url = Config.OPENAI_API_BASE_1
        if model_name is None:
            model_name = Config.DEFAULT_LLM_MODEL
        if temperature is None:
            temperature = Config.TEMPERATURE

    params = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": Config.MAX_TOKENS,
        "api_key": api_key,
        "base_url": base_url,
        **kwargs
    }

    if role == "coder":
        params["reasoning_effort"] = Config.CODER_REASONING_EFFORT

    return ChatOpenAI(**params)


def create_embedding():
    global _embedding_instance
    if _embedding_instance is None:
        with _embedding_lock:
            if _embedding_instance is None:
                _embedding_instance = HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cuda'},
                    encode_kwargs={'normalize_embeddings': True}
                )
    return _embedding_instance
