from shared.vector_store import (
    get_global_retriever,
    get_project_docs_retriever,
    get_code_retriever
)
from v1.memory import current_context, get_chat_history, get_project_memory
from shared.advanced_rag import search_hybrid

TOP_K_VECTOR = 5
MAX_CHAI_HISTORY = 10
MAX_RELATED_SCRIPTS = 3

SCENE_MEMORY_RULES = {
    "code_gen": {
        "chat_history": True,
        "project_memory": True,
        "rag": {
            "code_db": {"enabled": True, "retriever_type": "ensemble", "top_k": 4, "weight": 0.8},
            "project_docs": {"enabled": False},
            "global_docs": {"enabled": False},
        }
    },
    "doc_summary": {
        "chat_history": True,
        "project_memory": False,
        "rag": {
            "code_db": {"enabled": False},
            "project_docs": {"enabled": True, "retriever_type": "compressed", "top_k": 6},
            "global_docs": {"enabled": True, "retriever_type": "ensemble", "top_k": 3},
        }
    },
    "qa": {
        "chat_history": True,
        "project_memory": False,
        "rag": {
            "code_db": {"enabled": False},
            "project_docs": {"enabled": True, "retriever_type": "multi_query", "top_k": 5},
            "global_docs": {"enabled": True, "retriever_type": "multi_query", "top_k": 5},
        }
    },
    "debug": {
        "chat_history": True,
        "project_memory": True,
        "rag": {
            "code_db": {"enabled": True, "retriever_type": "compressed", "top_k": 6, "weight": 0.7},
            "project_docs": {"enabled": True, "retriever_type": "ensemble", "top_k": 2},
            "global_docs": {"enabled": False},
        }
    },
    "hybrid": {
        "chat_history": True,
        "project_memory": True,
        "rag": {
            "code_db": {"enabled": True, "retriever_type": "ensemble", "top_k": 4, "weight": 0.6},
            "project_docs": {"enabled": True, "retriever_type": "ensemble", "top_k": 3},
            "global_docs": {"enabled": True, "retriever_type": "ensemble", "top_k": 2},
        }
    }
}


def build_context(query: str, scene_type: str = "code_gen"):
    project_id = current_context["project_id"]
    if project_id is None:
        raise ValueError("当前没有设置project_id，请先调用set_current_project")
    chat_id = current_context["chat_id"]
    if chat_id is None:
        raise ValueError("当前没有设置chat_id，请先调用set_current_chat")
    user_id = current_context["user_id"]
    if user_id is None:
        raise ValueError("当前没有设置user_id，请先调用set_current_user")

    rule = SCENE_MEMORY_RULES.get(scene_type, SCENE_MEMORY_RULES['hybrid'])
    sections = []

    if rule.get("chat_history"):
        chat_history = get_chat_history()[-MAX_CHAI_HISTORY:]
        chat_text = "\n".join([f"{m['role']}: {m['message']}" for m in chat_history])
        sections.append(f"对话历史:\n{chat_text}")

    if rule.get("project_memory"):
        project_memory = get_project_memory()
        overview_text = project_memory.get('overview', "")
        sections.append(f"项目描述:\n{overview_text}")

        code_retriever = get_code_retriever(project_id, user_id)
        if code_retriever:
            code_docs = code_retriever.invoke(query)[-MAX_RELATED_SCRIPTS:]
        else:
            code_docs = []
        related_scripts = []
        for doc in code_docs:
            filename = doc.metadata.get("filename")
            if not filename or filename not in project_memory.get("scripts", {}):
                continue
            data = project_memory['scripts'][filename]
            summary = data.get('summary', "（暂无总结）")
            structure = data.get("structure", "（暂无结构信息）")
            structure_text = ""
            if isinstance(structure, dict):
                for cls in structure.get("classes", []):
                    structure_text += f"class {cls['name']} with methods: {', '.join(cls.get('methods', []))}\n"
                call_sites = structure.get("call_sites", [])
                if call_sites:
                    structure_text += f"调用: {', '.join(call_sites)}\n"
            else:
                structure_text = "\n".join(structure)
            dependencies = data.get("dependencies", "（暂无依赖文件）")
            dependencies_text = ",".join(dependencies)
            block = f"""文件：{filename}\n[结构]\n{structure_text}\n[功能总结]\n{summary}\n[依赖]{dependencies_text}"""
            related_scripts.append(block)

        if related_scripts:
            sections.append("Related Scripts:\n" + "\n\n".join(related_scripts))

    for db_name, db_cfg in rule.get("rag", {}).items():
        if not db_cfg.get("enabled"):
            continue
        retriever_type = db_cfg.get("retriever_type", "ensemble")
        top_k = db_cfg.get("top_k", 4)
        if db_name == "code_db":
            docs = search_hybrid(query, retriever_type=retriever_type, top_k=top_k,
                                 bm25_weight=db_cfg.get("weight", 0.5))
            if docs:
                code_snippets = []
                for d in docs:
                    meta = d.metadata
                    header = f"文件: {meta.get('filename', 'unknown')}"
                    if meta.get('class_name'):
                        header += f" 类: {meta['class_name']}"
                    if meta.get('method_name'):
                        header += f" 方法: {meta['method_name']}"
                    if meta.get('is_public') is not None:
                        header += f" 公开: {meta['is_public']}"
                    code_snippets.append(f"{header}\n{d.page_content}")
                sections.append("相关代码片段（混合检索）：\n\n" + "\n\n".join(code_snippets))
        elif db_name == "project_docs":
            retriever = get_project_docs_retriever(project_id, user_id, k=top_k)
            if retriever:
                docs = retriever.invoke(query)[:TOP_K_VECTOR]
            else:
                docs = []
            text = "\n".join(d.page_content for d in docs)
            sections.append(f"项目文档:\n{text}")
        elif db_name == "global_docs":
            retriever = get_global_retriever(k=top_k)
            if retriever:
                docs = retriever.invoke(query)[:TOP_K_VECTOR]
            else:
                docs = []
            text = "\n".join(d.page_content for d in docs)
            sections.append(f"全局文档:\n{text}")

    return "\n\n".join(sections)

