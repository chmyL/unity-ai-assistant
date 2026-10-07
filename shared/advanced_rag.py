import re
from typing import List, Optional, Dict, Any
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from shared.llm_factory import create_llm
from shared.vector_store import ensure_code_vector_store
from v1.memory import current_context
import threading

try:
    from langchain.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    try:
        from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    except ImportError:
        LLMChainExtractor = None
        print("警告：LLMChainExtractor不可用，压缩检索功能将禁用")


# 空检索器类
class _EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, **kwargs):
        return []

    async def _aget_relevant_documents(self, query: str, **kwargs):
        return []


# 全局缓存 BM25 索引（每个项目/用户独立，因为代码库会变化）
_bm25_index_cache: Dict[str, BM25Retriever] = {}
_cache_lock = threading.Lock()


# 某个项目代码发生变化，删除这个项目旧BM25，下次检索时重新从code vector store构建。
def invalidate_bm25_cache(project_id: str, user_id: str):
    cache_key = f"{user_id}_{project_id}"

    with _cache_lock:
        if cache_key in _bm25_index_cache:
            del _bm25_index_cache[cache_key]
            print(f"已清除 BM25 缓存：{user_id}/{project_id}")


# 从代码向量库中获取所有代码块，构建BM25检索器（基于词频）
def get_bm25_retriever(project_id: str, user_id: str, k: int = 4):
    cache_key = f"{user_id}_{project_id}"

    # 先检查缓存
    with _cache_lock:
        if cache_key in _bm25_index_cache:
            return _bm25_index_cache[cache_key]

    # 不要在 _cache_lock 内调用 ensure
    vector_store = ensure_code_vector_store(project_id, user_id)
    documents = []

    if vector_store is None:
        print("代码向量库不可用，BM25检索器将返回空结果")
        bm25 = _EmptyRetriever()

        with _cache_lock:
            _bm25_index_cache[cache_key] = bm25

        return bm25

    try:
        all_docs = vector_store.get()

        for text, meta in zip(
            all_docs["documents"],
            all_docs["metadatas"]
        ):
            if text:
                doc = Document(
                    page_content=text,
                    metadata=meta
                )
                documents.append(doc)

    except Exception as e:
        print(f"获取代码库文档失败：{e}")
        documents = []

    if not documents:
        print("没有代码文档，BM25检索器将返回空结果")
        bm25 = _EmptyRetriever()
    else:
        bm25 = BM25Retriever.from_documents(documents)

    with _cache_lock:
        _bm25_index_cache[cache_key] = bm25

    return bm25


# 返回EnsembleRetriever，结合BM25和向量检索（code_db）
def get_hybrid_code_retriever(project_id: str, user_id: str, k: int = 6, bm25_weight: float =0.5):
    vector_store = ensure_code_vector_store(project_id, user_id)
    if vector_store is None:
        return _EmptyRetriever()

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    bm25_retriever = get_bm25_retriever(project_id, user_id, k=k)
    ensemble = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[1 - bm25_weight, bm25_weight])
    return ensemble


# 使用LLM生成多个查询以提高召回率
def get_multi_query_retriever(base_retriever:BaseRetriever, num_queries: int = 3):
    llm = create_llm(temperature=0)
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        num_queries=num_queries,
        include_original=True
    )


# 使用LLM压缩检索结果，提取最相关片段
def get_compressed_retriever(base_retriever: BaseRetriever, llm=None):
    if llm is None:
        llm = create_llm(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor = compressor)


# 高级检索统一入口
def search_hybrid(qurey: str, retriever_type: str = "ensemble", top_k: int = 5, **kwargs):
    project_id = current_context.get("project_id")
    user_id = current_context.get("user_id")
    if not project_id or not user_id:
        return []

    if retriever_type == "ensemble":
        retriever = get_hybrid_code_retriever(project_id, user_id, k=top_k, bm25_weight=kwargs.get("bm25_weight", 0.5))
    elif retriever_type == "multi_query":
        base = get_hybrid_code_retriever(project_id, user_id, k=top_k)
        retriever = get_multi_query_retriever(base, num_queries=kwargs.get("num_queries", 3))
    elif retriever_type == "compressed":
        base = get_hybrid_code_retriever(project_id, user_id, k=top_k*2)
        retriever = get_compressed_retriever(base)
    else:
        raise ValueError(f"Unknown retriever_type：{retriever_type}")

    return retriever.invoke(qurey)
