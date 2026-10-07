import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent
    DEFAULT_LLM_MODEL = "qwen3.8-flash"
    CODER_LLM_MODEL = "qwen3.8-flash"
    TEMPERATURE = 0.3
    MAX_TOKENS = 8192
    CODER_REASONING_EFFORT = "medium"
    EMBEDDING_MODEL = str(PROJECT_ROOT / "models" / "huggingface" / "models_sentence_transformers_all_MiniLM_L6_v2" /
                          "snapshots" / "all_MiniLM_L6_v2")
    CHROMA_GLOBAL_DOCS_DIR = str(PROJECT_ROOT / "chroma_db" /"docs_db")
    CHROMA_CODE_DIR = str(PROJECT_ROOT / "chroma_db" / "{user_id}" / "code_db")
    CHROMA_PROJECT_DOCS_DIR = str(PROJECT_ROOT / "chroma_db" / "{user_id}" / "project_docs_db")
    DOCUMENTS_DIR = str(PROJECT_ROOT / "global_docs" / "sample_docs")
    GLOBAL_DOCS_DIR = str(PROJECT_ROOT / "global_docs")
    OPENAI_API_KEY_1 = os.getenv("OPENAI_API_KEY_1")
    OPENAI_API_BASE_1 = os.getenv("OPENAI_BASE_URL_1")
    OPENAI_API_KEY_2 = os.getenv("OPENAI_API_KEY_2")
    OPENAI_API_BASE_2 = os.getenv("OPENAI_BASE_URL_2")
    CHROMA_GLOBAL_DOCS_COLLECTION_NAME = "global_docs"
    CHROMA_CODE_COLLECTION_NAME = "code_{project_id}"
    CHROMA_PROJECT_DOCS_COLLECTION_NAME = "project_docs_{project_id}"
    MEMORY_DIR = str(PROJECT_ROOT / "projects_memory")
    CHAT_HISTORY_DIR = str(PROJECT_ROOT / "chat_histories")

    @staticmethod
    def get_default_llm_params():
        return {
            "model": Config.DEFAULT_LLM_MODEL,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
            "api_key": Config.OPENAI_API_KEY_1,
            "base_url": Config.OPENAI_API_BASE_1
        }

    @staticmethod
    def get_coder_llm_params():
        return {
            "model": Config.CODER_LLM_MODEL,
            "temperature": 0.2,  # coder一般更低
            "max_tokens": Config.MAX_TOKENS,
            "api_key": Config.OPENAI_API_KEY_2,
            "base_url": Config.OPENAI_API_BASE_2
        }
