import random
from pathlib import Path

from shared.vector_store import get_global_retriever, get_project_docs_retriever, \
    get_code_retriever, get_code_retriever_with_filter
from langchain_core.tools import StructuredTool
from pydantic import Field, BaseModel
from shared.llm_factory import create_llm
from v1.memory import get_project_context, update_scripts, get_project_memory, current_context, \
    extract_dependencies_llm, wait_for_dependencies, map_dep_names_to_files, add_chat_message, ROLE_ASSISTANT, \
    ROLE_SYSTEM, get_reverse_deps
from config import Config
from shared.prompts import UNITY_CODE_PROMPT, UNITY_FILENAME_PROMPT, UNITY_UPDATE_PROMPT
from langchain_core.output_parsers import StrOutputParser
from shared.utils import ensure_unique_filename, on_summary_done, parse_update_input, get_script_code,_soft_truncate, \
    extract_all_identifiers, get_all_project_symbols
from typing import Optional

MAX_DOCS = 20


class SearchSchema(BaseModel):
    query: str = Field(description="要检索的关键词")


class SummarySchema(BaseModel):
    query: Optional[str] = Field(default="", description="用于指定要总结的文档内容范围，例如：'总结全文'、'用户管理模块'、'系统架构设计'等；如果不提供则默认总结全文")


class CreateSchema(BaseModel):
    description: str = Field(description="用于描述新生成Unity C#脚本所需要的功能")


class UpdateSchema(BaseModel):
    input: str = Field(description="完整的修改指令，需包含目标脚本文件名和修改需求描述")


class OverviewSchema(BaseModel):
    overview: str = Field(description="项目的一句话描述，包含游戏类型、核心玩法、主要系统等")


class CodeSearchSchema(BaseModel):
    query: str = Field(description="搜索关键词（例如：'玩家移动逻辑'）")
    class_name: Optional[str] = Field(default=None, description="限定类名（如 PlayerController）")
    method_name: Optional[str] = Field(default=None, description="限定方法名（如 Start）")
    top_k: Optional[int] = Field(default=3, description="返回结果数量（默认3，最大10）")


# 检索文档
def search_document(query: str) -> str:
    try:
        project_id = current_context["project_id"]
        user_id = current_context["user_id"]
        if not project_id:
            return "当前没有设置project_id，请先调用set_current_project()"
        if not user_id:
            return "当前没有设置user_id，请先调用set_current_user()"
        retrievers = [
            ("全局文档", get_global_retriever(k=3)),
            ("项目文档", get_project_docs_retriever(project_id, user_id, k=3)),
            ("代码库", get_code_retriever(project_id, user_id, k=2))
        ]
        parts = []
        for label, retriever in retrievers:
            try:
                res = retriever.invoke(query)
                if not res:
                    continue
                for i, doc in enumerate(res, 1):
                    source = doc.metadata.get("source", label)
                    parts.append(f"[{label} {i}] {source}\n{doc.page_content}")
            except Exception:
                continue
        if not parts:
            return "未找到相关文档内容，请先上传文档并更新向量数据库。"

        final_result = "检索结果：\n\n" + "\n\n".join(parts)
        # 软截断后记录
        add_chat_message(ROLE_ASSISTANT, f"[search_document] {_soft_truncate(final_result, 300)}")
        return final_result
    except Exception as e:
        return f"检索失败：{e}"


# 文档总结
def summarize_document(query: str = ""):
    try:
        project_id = current_context["project_id"]
        user_id = current_context["user_id"]
        if not project_id:
            return "当前没有设置 project_id，请先调用 set_current_project()"
        if not user_id:
            return "当前没有设置user_id，请先调用set_current_user()"
        docs = []
        llm = create_llm(temperature=0.3)
        if not query or "全文" in query:
            retriever = get_project_docs_retriever(project_id,user_id, k=MAX_DOCS)
            project_docs = retriever.invoke("全部")
            if project_docs:
                docs.extend(random.sample(project_docs, min(len(project_docs), MAX_DOCS)))
        else:
            retrievers = [
                ("全局文档", get_global_retriever(k=4)),
                ("项目文档", get_project_docs_retriever(project_id, user_id, k=4)),
            ]
            for label, retriever in retrievers:
                try:
                    res = retriever.invoke(query)
                    if res:
                        docs.extend(res[:MAX_DOCS])
                except Exception:
                    continue
        if not docs:
            return "没有可总结的文档"
        partial_summaries = []
        for doc in docs:
            chunk = doc.page_content if hasattr(doc, "page_content") else str(doc)
            prompt = f"请对以下内容进行简要总结（不超过5点）：{chunk}"
            res = llm.invoke(prompt)
            text = res.content if hasattr(res, 'content') else str(res)
            partial_summaries.append(text)
        combined = "\n\n".join(partial_summaries)
        final_prompt = f"请用中文对一下内容进行清晰总结，分点列出核心要点：\n\n{combined}"
        response = llm.invoke(final_prompt)

        final_answer = response.content if hasattr(response, 'content') else str(response)
        add_chat_message(ROLE_ASSISTANT, f"[summarize_document] {_soft_truncate(final_answer, 300)}")
        return final_answer
    except Exception as e:
        return f"总结文档失败：{e}"


# C#脚本生成
def create_unity_script(description: str):
    try:
        project_context = get_project_context()
        project_memory = get_project_memory()

        similar_code = search_similar_code(description)
        if similar_code and "未找到相关代码" not in similar_code:
            project_context += f"\n\n【已有相似代码供参考】\n{similar_code}\n请参考上述代码的风格和接口设计，保持一致性。"

        if project_memory:
            all_funcs = []
            for fname, info in project_memory["scripts"].items():
                structure = info.get("structure", {})
                if isinstance(structure, dict):
                    # 从 classes 中收集类名和方法名
                    for cls in structure.get("classes", []):
                        all_funcs.append(cls["name"])
                        all_funcs.extend(cls.get("methods", []))

            dep_names = extract_dependencies_llm(description, all_funcs)
            dep_files = map_dep_names_to_files(dep_names)
            wait_for_dependencies(dep_files)
        else:
            print("这是项目的第一个脚本，没有依赖可以调用")

        llm = create_llm(model_name=Config.CODER_LLM_MODEL, temperature=0.2, role="coder")
        chain = UNITY_CODE_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"project_context": project_context, "description": description})
        if result is None:
            raise ValueError("代码生成失败：LLM 未返回有效内容")
        clean_code = str(result).replace("```csharp", "").replace("```", "").strip()
        if not clean_code:
            raise ValueError("代码生成失败：LLM 返回内容为空")

        # new_symbols = extract_all_identifiers(clean_code)
        # exisiting_symbols = get_all_project_symbols()
        # missing_symbols = new_symbols - exisiting_symbols
        #
        # if missing_symbols:
        #     warn_msg = f"新代码中引用了以下未定义的符号：{', '.join(missing_symbols)}。请确保这些类/方法已在项目中定义。"
        #     print(warn_msg)
        #     add_chat_message(ROLE_SYSTEM, warn_msg)
        # else:
        #     print("新代码中的符号均已在项目中定义，依赖检查通过。")

        default_llm = create_llm()
        filename_res = default_llm.invoke(UNITY_FILENAME_PROMPT.invoke({'description': description}))
        suggested_filename = (filename_res.content if hasattr(filename_res, 'content') else str(filename_res)).strip().replace("\n", "").replace(" ", "")
        if not suggested_filename.endswith(".cs"):
            suggested_filename += ".cs"
        suggested_filename = ensure_unique_filename(suggested_filename, project_memory["scripts"])

        update_scripts(suggested_filename, clean_code, default_llm, on_summary_done)
        return f"[{suggested_filename}]\n\n{clean_code}"
    except ValueError:
        raise
    except Exception as e:
        return f"代码生成失败：{e}"


# def create_unity_script(description: str):
#     try:
#         # 只获取项目上下文（如果出错就给空）
#         try:
#             project_context = get_project_context()
#         except:
#             project_context = "（暂无项目上下文）"
#
#         # 直接调用 LLM 生成代码
#         llm = create_llm(model_name=Config.CODER_LLM_MODEL, temperature=0.2, role="coder")
#         from langchain_core.prompts import ChatPromptTemplate
#         # 使用一个极简的 prompt 先测试
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "你是一个 Unity C# 代码生成助手。请直接输出代码，不要包含解释。"),
#             ("user", "根据以下需求生成 Unity C# 脚本：\n{description}\n\n项目上下文：\n{project_context}")
#         ])
#         chain = prompt | llm | StrOutputParser()
#         result = chain.invoke({"description": description, "project_context": project_context})
#         print(f"DEBUG: LLM 返回长度 = {len(result)}")
#         if not result or len(result.strip()) == 0:
#             return "代码生成失败：LLM 未返回有效内容"
#
#         clean_code = result.replace("```csharp", "").replace("```", "").strip()
#         if not clean_code:
#             return "代码生成失败：清洗后代码为空"
#
#         # 生成文件名
#         default_llm = create_llm()
#         from shared.prompts import UNITY_FILENAME_PROMPT
#         filename_res = default_llm.invoke(UNITY_FILENAME_PROMPT.invoke({'description': description}))
#         suggested_filename = (
#             filename_res.content if hasattr(filename_res, 'content') else str(filename_res)).strip().replace("\n",
#                                                                                                              "").replace(
#             " ", "")
#         if not suggested_filename.endswith(".cs"):
#             suggested_filename += ".cs"
#         from shared.utils import ensure_unique_filename
#         from v1.memory import get_project_memory
#         project_memory = get_project_memory()
#         suggested_filename = ensure_unique_filename(suggested_filename, project_memory["scripts"])
#
#         # 保存脚本（不进行依赖分析）
#         from v1.memory import update_scripts
#         from shared.utils import on_summary_done
#         update_scripts(suggested_filename, clean_code, default_llm, on_summary_done)
#         return f"[{suggested_filename}]\n\n{clean_code}"
#     except Exception as e:
#         return f"代码生成失败：{e}"


def update_unity_script_tool(input: str):
    # 先解析 LLM 输出
    params = parse_update_input(input, default_file="Unknown.cs")

    # 然后再调用原来的 update_unity_script
    return update_unity_script(
        filename=params['filename'],
        description=params['description']
    )


# 修改已有C#脚本
def update_unity_script(filename: str, description: str):
    try:
        project_context = get_project_context()
        project_memory = get_project_memory()

        similar_code = search_similar_code(description)  # 复用已有工具
        if "未找到" not in similar_code:
            project_context += f"\n\n【已有相似代码供参考】\n{similar_code}\n请参考上述代码的风格和接口设计，保持一致性。"

        project_id = current_context["project_id"]
        user_id = current_context["user_id"]
        old_code = get_script_code(project_id, user_id, filename)
        if not old_code:
            return f"未找到脚本：{filename}（文件不存在）"

        llm = create_llm(Config.CODER_LLM_MODEL, temperature=0.25)
        default_llm = create_llm()
        chain = UNITY_UPDATE_PROMPT | llm
        new_code = chain.invoke({'project_context': project_context, 'old_code': old_code, 'description': description})
        if new_code is None:
            raise ValueError("代码修改失败：LLM 未返回有效内容")
        text = new_code.content if hasattr(new_code, "content") else str(new_code)
        if text is None:
            raise ValueError("代码修改失败：LLM 未返回有效内容")
        clean_code = text.replace("```csharp", "").replace("```", "").strip()
        if not clean_code:
            raise ValueError(f"代码修改失败：LLM 返回内容为空，未修改 {filename}")

        # new_symbols = extract_all_identifiers(clean_code)
        # exisiting_symbols = get_all_project_symbols()
        # missing_symbols = new_symbols - exisiting_symbols
        #
        # if missing_symbols:
        #     warn_msg = f"新代码中引用了以下未定义的符号：{', '.join(missing_symbols)}。请确保这些类/方法已在项目中定义。"
        #     print(warn_msg)
        #     add_chat_message(ROLE_SYSTEM, warn_msg)
        # else:
        #     print("新代码中的符号均已在项目中定义，依赖检查通过。")

        update_scripts(filename, clean_code, default_llm, on_summary_done)

        # 获取依赖当前脚本的文件列表
        dependent_scripts = get_reverse_deps(filename)
        if dependent_scripts:
            dep_msg = f"脚本 {filename} 已更新。以下脚本依赖它，请检查是否需要同步调整：\n" + "\n".join(f" - {d}" for d in dependent_scripts)
            print(dep_msg)
            add_chat_message(ROLE_SYSTEM, dep_msg)
        else:
            print(f"脚本 {filename} 更新完成，无其他脚本依赖它。")
        return f"[{filename}]\n\n{clean_code}"
    except ValueError:
        raise
    except Exception as e:
        return f"代码修改失败：{e}"


# Agent 调用以记录项目整体描述
def set_project_overview_tool(overview: str) -> str:
    try:
        from v1.memory import set_project_overview
        set_project_overview(overview)
        add_chat_message(ROLE_ASSISTANT, f"[set_project_overview] {_soft_truncate(overview, 300)}")
        return "项目描述已成功记录。"
    except Exception as e:
        return f"设置项目描述失败：{e}"


def search_similar_code(query: str, class_name: str = None, method_name: str = None, top_k: int = 3) -> str:
    try:
        project_id = current_context["project_id"]
        user_id = current_context["user_id"]
        if not project_id or not user_id:
            return "未设置项目或用户"

        vector_dir = Path(
            f"{Config.CHROMA_CODE_DIR.format(user_id=user_id)}/{project_id}"
        )

        # 目录不存在 = 从未建立过代码库
        if not vector_dir.exists():
            print("代码向量库目录不存在，首次跳过代码检索")
            return ""

        filter_dict = {}
        if class_name:
            filter_dict["class_name"] = class_name
        if method_name:
            filter_dict["method_name"] = method_name

        # 限制 top_k 不超过 10
        k = min(top_k, 10)
        try:
            retriever = get_code_retriever_with_filter(
                project_id,
                user_id,
                filter_dict if filter_dict else None,
                k=k
            )

            # 先探测一次
            test_docs = retriever.invoke("test")

        except Exception as e:
            print(f"代码库首次初始化未完成: {e}")
            return ""

        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关代码"

        result = "相关代码片段：\n\n"
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            result += f"[{i}] {meta.get('filename', 'unknown')} "
            if meta.get('class_name'):
                result += f"类 {meta['class_name']} "
            if meta.get('method_name'):
                result += f"方法 {meta['method_name']} "
            result += f"(公开: {meta.get('is_public', False)})\n"
            result += f"{doc.page_content}\n\n"
        return result
    except Exception as e:
        print(f"代码检索失败（忽略）: {e}")
        return ""  # 返回空字符串，不影响后续生成


search = StructuredTool.from_function(
    func=search_document,
    name="search_document",
    description="检索用户上传的文档（研究论文、游戏设计文档等）",
    args_schema=SearchSchema, # type: ignore
    return_direct=False,
)

summary = StructuredTool.from_function(
    func=summarize_document,
    name="summarize_document",
    description="对上传的文档进行总结",
    args_schema=SummarySchema, # type: ignore
    return_direct=False
)

create_script = StructuredTool.from_function(
    func=create_unity_script,
    name="create_unity_script",
    description="生成Unity C#脚本，支持整个游戏项目中多个脚本之间的联系和调用",
    args_schema=CreateSchema, # type: ignore
    return_direct=False
)

update_script = StructuredTool.from_function(
    func=update_unity_script_tool,
    name="update_unity_script",
    description="修改已生成的Unity C#脚本，支持整个游戏项目中多个脚本之间的联系和调用。你必须提供明确的文件名（例如 PlayerController.cs）和修改需求描述。",
    args_schema=UpdateSchema, # type: ignore
    return_direct=False
)


set_overview = StructuredTool.from_function(
    func=set_project_overview_tool,
    name="set_project_overview",
    description="当用户描述项目整体目标、核心玩法或项目类型时，调用此工具记录项目描述。调用一次即可完成，不需要重复调用。后续生成代码时会自动参考该描述。",
    args_schema=OverviewSchema,
    return_direct=False
)


search_code = StructuredTool.from_function(
    func=search_similar_code,
    name="search_similar_code",
    description="根据自然语言描述搜索项目中已有的代码片段，可按类名或方法名过滤",
    args_schema=CodeSearchSchema,
    return_direct=False
)

TOOLS = [search, summary, create_script, update_script, set_overview, search_code]
