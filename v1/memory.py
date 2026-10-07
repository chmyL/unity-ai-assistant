import hashlib
import re
import time
from typing import Dict
from langchain_core.messages import HumanMessage
from shared.llm_factory import create_llm
from shared.prompts import UNITY_SUMMARY_PROMPT, CHUNK_SUMMARY_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from shared.utils import smart_code_split, save_scripts, load_project_memory, load_chat_history, \
    save_project_memory, append_chat_message, validate_path_component, list_user_projects
from shared.vector_store import index_code
import threading
import tree_sitter_c_sharp as tscs
from tree_sitter import Parser, Language
from contextvars import ContextVar

# 初始化 C# 解析器
c_sharp_language = Language(tscs.language())   # 包装成 Language 对象
c_sharp_parser = Parser(c_sharp_language)

project_memories: Dict[tuple[str, str], Dict] = {}
chat_histories: Dict[tuple[str, str, str], list] = {}

CHUNK_SIZE = 1500
MAX_SUMMARY_CHUNKS = 5
ROLE_USER = "user"
ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
_DEFAULT_CONTEXT = {
    "project_id": None,
    "chat_id": None,
    "user_id": None
}
# 上下文变量
_context_var = ContextVar(
    "current_context",
    default=None
)


class CurrentContextProxy:
    def _get_context(self):
        context = _context_var.get()
        if context is None:
            context = _DEFAULT_CONTEXT.copy()
            _context_var.set(context)
        return context

    def __getitem__(self, key):
        return self._get_context()[key]

    def __setitem__(self, key, value):
        self._get_context()[key] = value

    def get(self, key, default=None):
        return self._get_context().get(key, default)

    def copy(self):
        return self._get_context().copy()

    def set(self, context: dict):
        _context_var.set(context.copy())


# 中间转换一下
current_context = CurrentContextProxy()

# # 新项目开始时清空记忆
# def reset_project(project_id):
#     project_memories.pop(project_id, None)


def _save_project_overview_async(user_id: str, project_id: str, overview: str, overview_version: int):
    try:
        project_memory = project_memories.get((user_id, project_id))
        if project_memory is None:
            project_memory = load_project_memory(user_id, project_id)
        if project_memory is None:
            return

        current_version = project_memory.get("overview_version", 0)
        if current_version != overview_version:
            print(
                f"[Memory] 项目描述已过期，"
                f"当前版本={current_version}，"
                f"任务版本={overview_version}，放弃旧结果"
            )
            return

        # 确保后台线程保存的 overview 与它启动时一致
        project_memory["overview"] = overview
        project_memory["overview_version"] = overview_version

        save_project_memory(user_id, project_id, project_memory)
        print(f"[项目描述保存完成] {user_id}/{project_id}")

    except Exception as e:
        print(f"[项目描述保存失败] {user_id}/{project_id}：{e}")


def set_project_overview(overview: str):
    project_memory = get_project_memory()
    # 项目描述版本号递增
    overview_version = project_memory.get("overview_version", 0) + 1
    project_memory['overview'] = overview
    project_memory["overview_version"] = overview_version

    user_id = current_context.get("user_id")
    project_id = current_context["project_id"]
    if user_id:
        thread = threading.Thread(
            target=_save_project_overview_async,
            args=(user_id, project_id, overview, overview_version)
        )
        thread.start()

    print(f"[项目描述已更新] {overview}")


def set_current_project(project_id: str):
    user_id = current_context["user_id"]
    if user_id is None:
        raise ValueError("当前没有设置user_id，请先调用 set_current_user")

    project_id = validate_path_component(project_id, "project_id")
    if project_id not in list_user_projects(user_id):
        raise ValueError(f"项目不存在或不属于当前用户：{project_id}")

    # 切换项目时必须清空旧项目的聊天上下文
    current_context["project_id"] = project_id
    current_context["chat_id"] = None
    project_key = (user_id, project_id)
    if project_key not in project_memories:
        project_memories[project_key] = {"scripts": {}, "overview": "", "overview_version": 0, "reverse_deps": {}}
        loaded = load_project_memory(user_id, project_id)
        if loaded:
            project_memories[project_key] = loaded


def set_current_chat(chat_id: str):
    project_id = current_context["project_id"]
    user_id = current_context["user_id"]
    if user_id is None:
        raise ValueError("当前没有设置user_id，请先调用 set_current_user")
    if project_id is None:
        raise ValueError("当前没有设置project_id，请先调用set_current_project")

    # None 表示清空当前聊天，不创建 None 聊天
    if chat_id is None:
        current_context["chat_id"] = None
        return

    current_context['chat_id'] = chat_id
    chat_key = (user_id, project_id, chat_id)
    if chat_key not in chat_histories:
        chat_histories[chat_key] = []
        loaded = load_chat_history(user_id, project_id, chat_id)
        if loaded:
            chat_histories[chat_key] = loaded


def set_current_user(user_id: str):
    current_context['user_id'] = user_id
    current_context["project_id"] = None
    current_context["chat_id"] = None


# 向当前chat_id的对话窗口添加消息
def add_chat_message(role: str, message: str):
    project_id = current_context["project_id"]
    if project_id is None:
        raise ValueError("当前没有设置project_id，请先调用set_current_project")

    chat_id = current_context["chat_id"]
    if chat_id is None:
        raise ValueError("当前没有设置chat_id，请先调用set_current_chat")

    user_id = current_context["user_id"]
    if user_id is None:
        raise ValueError("当前没有设置user_id，请先调用set_current_user")

    chat_key = (user_id, project_id, chat_id)
    chat_histories[chat_key].append({
        'role': role,
        'message': message
    })
    append_chat_message(user_id,project_id,chat_id, {'role': role, 'message': message})


# 获取项目记忆
def get_project_memory():
    user_id = current_context["user_id"]
    project_id = current_context["project_id"]
    if user_id is None:
        raise ValueError("当前没有设置user_id，请先调用 set_current_user")
    if project_id is None:
        raise ValueError("当前没有设置project_id，请先调用set_current_project")

    project_key = (user_id, project_id)
    if project_key not in project_memories:
        project_memories[project_key] = {
            "scripts": {},
            "overview": "",
            "overview_version": 0,
            "reverse_deps": {}
        }

    return project_memories[project_key]


# 获取当前chat_id的对话历史
def get_chat_history():
    user_id = current_context["user_id"]
    project_id = current_context["project_id"]
    chat_id = current_context["chat_id"]

    if user_id is None:
        raise ValueError("当前没有设置user_id，请先调用set_current_user")
    if project_id is None:
        raise ValueError("当前没有设置project_id，请先调用set_current_project")
    if chat_id is None:
        raise ValueError("当前没有设置chat_id，请先调用set_current_chat")

    chat_key = (user_id, project_id, chat_id)

    return chat_histories[chat_key]


# 从内存缓存里移除某个会话（删除会话时调用）
def remove_chat_from_memory(user_id: str, project_id: str, chat_id: str):
    key = (user_id, project_id, chat_id)
    chat_histories.pop(key, None)


# # 提取代码结构
# def smart_structure(code: str, max_len=700):
#     lines = code.splitlines()
#     important = []
#     method_pattern = re.compile(
#         r'^\s*(public|private|protected|internal|static|virtual|override|async|sealed|partial)+\s+[\w<>\[\]]+\s+\w+\s*\(.*\)\s*{?'
#     )
#
#     for line in lines:
#         line_strip = line.strip()
#         # 类、结构体、枚举
#         if line_strip.startswith(("class ", "struct ", "enum ")):
#             important.append(line_strip)
#         # 方法签名匹配
#         elif method_pattern.match(line_strip):
#             important.append(line_strip)
#
#     return important

# def smart_structure(code: str):
#     tree = c_sharp_parser.parse(code.encode('utf-8'))
#     root_node = tree.root_node
#
#     classes = []
#     public_methods = []
#     call_sites = []
#
#     def traverse(node):
#         nonlocal classes, public_methods, call_sites
#         if node.type == "class_declaration":
#             class_name = None
#             class_methods = []
#             # 获取类名
#             for child in node.children:
#                 if child.type == "identifier":
#                     class_name = code[child.start_byte:child.end_byte]
#                     break
#             # 遍历类内所以方法声明
#             for child in node.children:
#                 if child.type == "method_declaration":
#                     method_name = None
#                     is_public = False
#                     for sub in child.children:
#                         if sub.type == "identifier":
#                             method_name = code[sub.start_byte:sub.end_byte]
#                         elif sub.type == "modifier" and code[sub.start_byte:sub.end_byte] == "public":
#                             is_public = True
#                     if method_name:
#                         class_methods.append(method_name)
#                         if is_public:
#                             public_methods.append(method_name)
#             if class_name:
#                 classes.append({"name":class_name, "methods": class_methods})
#         elif node.type == "invocation_expression":
#             call_text = code[node.start_byte:node.end_byte].strip()
#             if call_text and len(call_text) < 200:
#                 call_sites.append(call_text)
#         for child in node.children:
#             traverse(child)
#     traverse(root_node)
#
#     # 去重
#     public_methods = list(set(public_methods))
#     call_sites = list(dict.fromkeys(call_sites))[:100]
#
#     return {
#         "classes": classes,
#         "public_methods": public_methods,
#         "call_sites": call_sites
#     }


def smart_structure(code: str):
    import re

    # 1. 提取所有类及其方法
    class_pattern = re.compile(r'^\s*(?:public|private|protected|internal)?\s+(?:partial\s+)?class\s+(\w+)', re.MULTILINE)
    classes = []
    for match in class_pattern.finditer(code):
        class_name = match.group(1)
        # 定位类的大括号范围（简单括号计数，快速定位）
        start_pos = match.start()
        brace_count = 0
        in_class = False
        class_end = start_pos
        for i, ch in enumerate(code[start_pos:], start_pos):
            if ch == '{':
                brace_count += 1
                in_class = True
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    class_end = i
                    break
        if not in_class:
            class_end = len(code)  # 防御：没找到闭合括号就取到末尾
        class_body = code[start_pos:class_end+1]

        # 在类体中提取方法名（包括 Unity 生命周期方法）
        method_pattern = re.compile(
            r'^\s*(?:public|private|protected|internal|static|virtual|override|async)?\s+[\w<>\[\]]+\s+(\w+)\s*\(',
            re.MULTILINE
        )
        methods = list(set(method_pattern.findall(class_body)))
        # 过滤掉 C# 属性访问器（get/set 等）
        methods = [m for m in methods if m not in ('get', 'set', 'add', 'remove')]

        # 提取公开属性（表达式体 + get/set 属性）
        # public Dimension CurrentDimension => _currentDimension;
        # public int Health { get; set; }
        property_pattern = re.compile(
            r'^\s*public\s+'
            r'(?!class\b|struct\b|enum\b|interface\b)'
            r'(?:static\s+)?(?:override\s+)?(?:virtual\s+)?'
            r'[\w<>\[\],\s]+?\s+(\w+)\s*(?:=>|\{)',
            re.MULTILINE
        )
        property_names = property_pattern.findall(class_body)

        # 提取公开字段（以 = 或 ; 结尾）
        field_pattern = re.compile(
            r'^\s*public\s+'
            r'(?!class\b|struct\b|enum\b|interface\b)'
            r'(?:static\s+)?(?:readonly\s+)?'
            r'[\w<>\[\],\.\s]+?'
            r'(\w+)\s*(?==|;)',
            re.MULTILINE
        )
        field_names = field_pattern.findall(class_body)

        # 合并去重 + 只过滤一次
        properties = list(set(property_names) | set(field_names))
        properties = [p for p in properties if p not in ('get', 'set', 'add', 'remove')]

        classes.append({
            "name": class_name,
            "methods": methods,
            "properties": properties,
        })

    # 2. 提取顶层 public 方法（一般 C# 脚本中很少见，但保留）
    top_method_pattern = re.compile(
        r'^\s*public\s+[\w<>\[\]]+\s+(\w+)\s*\(',
        re.MULTILINE
    )
    public_methods = list(set(top_method_pattern.findall(code)))

    # 3. 提取调用点（所有方法调用，保留方法名，过滤关键字）
    call_pattern = re.compile(r'\b(\w+)\s*\(')
    keywords = {'if', 'while', 'for', 'foreach', 'switch', 'using', 'catch', 'lock', 'fixed', 'sizeof', 'typeof', 'nameof', 'delegate'}
    call_sites = []
    for match in call_pattern.finditer(code):
        name = match.group(1)
        if name not in keywords:
            call_sites.append(name)
    # 去重并限制数量
    call_sites = list(dict.fromkeys(call_sites))[:100]

    return {
        "classes": classes,
        "public_methods": public_methods,
        "call_sites": call_sites
    }


# # 脚本总结
# def llm_summary(code: str, llm):
#     chunk_prompt = ChatPromptTemplate.from_template(CHUNK_SUMMARY_PROMPT)
#     chain = chunk_prompt | llm
#     chunks = smart_code_split(code)
#
#     if len(chunks)==1:
#         res = chain.invoke({"code": code})
#         return res.content if hasattr(res, "content") else str(res)
#     partials = []
#     for chunk in chunks:
#         res = chain.invoke({'code': chunk})
#         partials.append(res.content if hasattr(res, "content") else str(res))
#
#     final_prompt = ChatPromptTemplate.from_template(UNITY_SUMMARY_PROMPT)
#     final_chain = final_prompt | llm
#     final_result = final_chain.invoke({
#         'text': "\n".join(partials)
#     })
#     return final_result.content if hasattr(final_result, "content") else str(final_result)


def extract_dependencies_llm(description: str, project_functions:list):
    prompt = f"""
    以下是代码描述:
    {description}
    
    项目已有函数/类名列表: {project_functions}

    请只返回描述中可能会调用到的函数或类名，按列表形式，不要输出其它内容。
    """
    try:
        llm = create_llm(temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, 'content') else str(response)
        deps = [name.strip() for name in re.findall(r"\b\w+\b", content) if name.strip() in project_functions]
        return list(set(deps))
    except Exception as e:
        print(f"[extract_dependencies_llm] 出错: {e}")
        return []


def map_dep_names_to_files(dep_names: list) -> list:
    project_memory = get_project_memory()
    name_to_files = {}
    for fname, data in project_memory['scripts'].items():
        structure = data.get("structure", {})
        for cls in structure.get('classes', []):
            name_to_files.setdefault(cls['name'], []).append(fname)
            for m in cls.get('methods', []):
                name_to_files.setdefault(m, []).append(fname)
        for call in structure.get('call_sites'):
            func_name = call.split("(")[0]
            name_to_files.setdefault(func_name, []).append(fname)

    files = []
    for name in dep_names:
        if name in name_to_files:
            files.extend(name_to_files[name])  # 如果是重名，保守地等待所有包含该名的文件
    return list(set(files))  # 去重


# 阻塞等待依赖脚本summary完成
def wait_for_dependencies(filenames: list, timeout=30, check_interval=0.5):
    project_memory = get_project_memory()
    start = time.time()
    while True:
        all_done = True
        for fname in filenames:
            dep_info = project_memory['scripts'].get(fname)
            if not dep_info or not dep_info.get('summary_done'):
                all_done = False
                break
        if all_done:
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"等待文件超时: {filenames}")
        time.sleep(check_interval)


# 脚本总结（异步，后台生成并更新memory，不阻塞主进程)(可以通过callback通知summary完成)
def _generate_summary_in_background(filename: str, code: str, llm, new_hash, dependencies, user_id, project_id,
                                    chat_id, summary_version, callback=None):
    try:
        # 缓存
        project_key = (user_id, project_id)
        project_memory = project_memories.get(project_key)
        # 磁盘
        if project_memory is None:
            project_memory = load_project_memory(user_id, project_id)
        if project_memory is None:
            project_memory = {"scripts": {}, "overview": "", "reverse_deps": {}}

        chunk_prompt = ChatPromptTemplate.from_template(CHUNK_SUMMARY_PROMPT)
        chain = chunk_prompt | llm
        chunks = smart_code_split(code)
        partial_summaries = []
        for i in range(0, len(chunks), MAX_SUMMARY_CHUNKS):
            batch = chunks[i:i+MAX_SUMMARY_CHUNKS]
            batch_text = "\n\n".join(batch)
            res = chain.invoke({'code': batch_text})
            summary = res.content if hasattr(res, 'content') else str(res)
            partial_summaries.append(summary)
        final_prompt = ChatPromptTemplate.from_template(UNITY_SUMMARY_PROMPT)
        final_chain = final_prompt | llm
        full_summary = final_chain.invoke({
            'text': "\n\n".join(partial_summaries)
        })
        summary_text = full_summary.content if hasattr(full_summary, 'content') else str(full_summary)
        structure = smart_structure(code)

        # Summary 生成完成后，再次检查当前脚本是否还是这个版本
        current_memory = project_memories.get(project_key)

        if current_memory is None:
            current_memory = load_project_memory(user_id, project_id)

        if current_memory is None:
            print(f"[Memory] {filename} summary 放弃保存：项目记忆不存在")
            return

        current_script = current_memory.get("scripts", {}).get(filename)

        if current_script is None:
            print(f"[Memory] {filename} summary 放弃保存：脚本记录不存在")
            return

        current_version = current_script.get("summary_version", 0)

        if current_version != summary_version:
            print(
                f"[Memory] {filename} summary 已过期，"
                f"当前版本={current_version}，"
                f"任务版本={summary_version}，放弃旧结果"
            )
            return

        # 版本仍然一致，说明这个 summary 仍然属于当前代码版本
        current_script.update({
            "summary": summary_text or "（生成失败）",
            "structure": structure or "（解析失败）",
            "hash": new_hash,
            "dependencies": dependencies,
            "summary_done": True,
            "summary_stale": False,
            "summary_version": summary_version
        })

        # 写入当前对话的 chat_history
        if chat_id:
            append_chat_message(user_id, project_id, chat_id, {"role": ROLE_SYSTEM,
                                                               "message": f"脚本 {filename} 的 summary 已生成完成"})
        save_project_memory(user_id, project_id, project_memory)
        print(f"[Memory 更新完成] {filename}")
        if callback:
            callback(filename, user_id, project_id)
    except Exception as e:
        print(f"[Memory 更新失败] {filename}: {e}")


def get_hash(code):
    return hashlib.md5(code.encode()).hexdigest()


# 实际依赖 —— 通过 tree-sitter 提取"跨类引用"，匹配其他脚本的公开符号
def actual_dependencies(filename: str, code: str):
    """
    判断当前脚本依赖了哪些其他脚本。
    匹配目标：其他脚本的【类名】、【公开方法名】、【公开属性名】。
    匹配来源：当前代码里的 member_access（A.B）、new X()、类型引用。
    """
    project_memory = get_project_memory()

    # 1. 用 tree-sitter 提取当前代码里的"跨类引用"
    language = Language(tscs.language())
    parser = Parser(language)
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node

    referenced = set()

    def _text(node):
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def walk(node):
        # A.B → 取 B 部分
        if node.type == "member_access_expression":
            for c in reversed(node.children):
                if c.type == "identifier":
                    referenced.add(_text(c))
                    break

        # new X(...)
        elif node.type == "object_creation_expression":
            for c in node.children:
                if c.type in ("identifier", "qualified_name"):
                    name = _text(c).split(".")[-1]
                    referenced.add(name)
                    break

        # 类型引用（字段、参数、局部变量）
        elif node.type in ("variable_declaration", "parameter", "field_declaration"):
            for c in node.children:
                if c.type in ("identifier", "qualified_name", "generic_name"):
                    name = _text(c)
                    if "." in name:
                        name = name.split(".")[-1]
                    if "<" in name:
                        name = name.split("<")[0]
                    referenced.add(name)

        for c in node.children:
            walk(c)

    walk(root)

    # 2. 收集其他脚本的所有公开符号 → 文件名
    name_to_files = {}
    for other_file, data in project_memory["scripts"].items():
        if other_file == filename:
            continue

        structure = data.get("structure", {})
        if not isinstance(structure, dict):
            continue

        for cls in structure.get("classes", []):
            cname = cls.get("name")
            if cname:
                name_to_files.setdefault(cname, set()).add(other_file)

            for m in cls.get("methods", []):
                if m:
                    name_to_files.setdefault(m, set()).add(other_file)

            # ★ 新增：属性（老数据没有，用 .get 兜底）
            for p in cls.get("properties", []):
                if p:
                    name_to_files.setdefault(p, set()).add(other_file)

        for pm in structure.get("public_methods", []):
            if pm:
                name_to_files.setdefault(pm, set()).add(other_file)

    # 3. 求交集，按文件合并符号
    merged = {}
    for sym in referenced:
        if sym in name_to_files:
            for dep_file in name_to_files[sym]:
                merged.setdefault(dep_file, set()).add(sym)

    return [
        {"file": f, "symbols": sorted(syms)}
        for f, syms in merged.items()
    ]


# # 添加/更新脚本并生成总结，更新
# def actual_dependencies(filename: str, code: str):
#     project_memory = get_project_memory()
#     language = Language(tscs.language())
#     parser = Parser(language)
#     tree = parser.parse(code.encode('utf-8'))
#     root = tree.root_node
#
#     # 提取using命名空间
#     using_pattern = re.compile(r'using\s+([\w.]+);')
#     imported_nameplaces = using_pattern.findall(code)
#
#     # 从语法树中提取所有标识符
#     symbols_in_code = set()
#
#     def collect_symbols(node):
#         if node.type == "identifier":
#             sym = code[node.start_byte:node.end_byte]
#             symbols_in_code.add(sym)
#         elif node.type == "member_access_expression":
#             # 递归收集成员访问中的所有标识符（包括每一级）
#             for child in node.children:
#                 if child.type == "identifier":
#                     sym = code[child.start_byte:child.end_byte]
#                     symbols_in_code.add(sym)
#                 # 处理嵌套的 member_access_expression，如 a.b.c
#                 elif child.type == "member_access_expression":
#                     collect_symbols(child)
#         for child in node.children:
#             if node.type != "member_access_expression" or child.type != "member_access_expression":
#                 collect_symbols(child)
#     collect_symbols(root)
#
#     dependencies = []
#     for other_file, global_docs in project_memory['scripts'].items():
#         if other_file == filename:
#             continue
#         script_base = other_file.replace(".cs", "")
#         if not any(ns.endswith(script_base) for ns in imported_nameplaces):
#             pass
#         other_struct = global_docs.get('structure')
#         if not other_struct:
#             continue
#         if isinstance(other_struct, dict):
#             other_symbols = set()
#             # 收集其他脚本公开的类名和方法名
#             for cls in other_struct.get('classes', []):
#                 other_symbols.add(cls['name'])
#                 # 类内所有方法（包含非公开）也加入，实际依赖只需公开方法，但为了宽松匹配，全部加入
#                 other_symbols.update(cls.get('methods', []))
#         else:
#             other_symbols = set(re.findall(r'\b[a-zA-z][a-zA-Z-0-9_]*\b', str(other_struct)))
#
#         matched = symbols_in_code.intersection(other_symbols)
#         if matched:
#             dependencies.append({
#                 "file": other_file,
#                 "symbols": list(matched)
#             })
#     return dependencies


# 反向依赖，格式为[{"file": "xxx.cs", "symbols": [...]}]或旧格式的字符串列表
def _update_reverse_deps(user_id, project_id: str, filename: str, old_deps: list, new_deps: list):
    proj_mem = project_memories.get((user_id, project_id))
    if not proj_mem:
        return
    rev = proj_mem.setdefault("reverse_deps", {})

    # 辅助：从依赖项中提取文件名列表
    def extract_files(deps):
        files = []
        for d in deps:
            if isinstance(d, dict):
                files.append(d["file"])
            elif isinstance(d, str):
                files.append(d)
        return files

    old_files = extract_files(old_deps)
    new_files = extract_files(new_deps)

    # 删除旧的反向记录
    for of in old_files:
        if of in rev and filename in rev[of]:
            rev[of].remove(filename)
            if not rev[of]:
                del rev[of]

    # 添加新的反向记录
    for nf in new_files:
        rev.setdefault(nf, []).append(filename)
        rev[nf] = list(set(rev[nf]))   # 去重


# 返回依赖给定文件的所有脚本文件名列表
def get_reverse_deps(filename: str) -> list:
    project_memory = get_project_memory()
    # reverse_deps 存储在 project_memory 顶层，与 "scripts"、"overview" 同级
    reverse_map = project_memory.get("reverse_deps", {})
    return reverse_map.get(filename, [])


# 从项目记忆里移除某个脚本，同时清理反向依赖和引用
def remove_script_from_memory(filename: str) -> bool:
    project_memory = get_project_memory()
    scripts = project_memory.get("scripts", {})

    if filename not in scripts:
        return False

    # 1. 删除脚本记录
    del scripts[filename]

    # 2. 清理"谁依赖我"：直接删掉这个 key
    reverse_deps = project_memory.get("reverse_deps", {})
    reverse_deps.pop(filename, None)

    # 3. 清理其他脚本里指向它的依赖
    for other_file, data in scripts.items():
        deps = data.get("dependencies", [])
        if not deps:
            continue
        new_deps = []
        for d in deps:
            dep_file = d.get("file") if isinstance(d, dict) else d
            if dep_file != filename:
                new_deps.append(d)
        if len(new_deps) != len(deps):
            data["dependencies"] = new_deps

    return True


def update_scripts(filename: str, code: str, llm, callback=None):
    new_hash = get_hash(code)
    project_memory = get_project_memory()
    old = project_memory["scripts"].get(filename)
    old_deps = old.get("dependencies", []) if old else []
    old_summary = old.get("summary", "") if old else ""

    if old is None or old.get("hash") != new_hash:
        user_id = current_context["user_id"]
        project_id = current_context["project_id"]
        chat_id = current_context["chat_id"]

        if user_id is None:
            raise ValueError("当前没有设置user_id，请先调用set_current_user")
        if project_id is None:
            raise ValueError("当前没有设置project_id，请先调用set_current_project")

        save_scripts(project_id, filename, code, user_id)
        index_code(project_id, filename, code, current_context['user_id'])
        dependencies = actual_dependencies(filename, code)
        # 更新反向索引
        _update_reverse_deps(user_id, project_id, filename, old_deps, dependencies)

        # 当前脚本的 summary 版本号递增
        summary_version = old.get("summary_version", 0) + 1 if old else 1
        project_memory["scripts"][filename] = {
            'hash': new_hash,
            'structure': smart_structure(code),
            'summary': old_summary if old_summary else "（该脚本刚创建，summary正在后台生成）",
            'dependencies': dependencies,
            'summary_done': False,
            'summary_stale': bool(old_summary),
            "summary_version": summary_version
        }
        print(f"[Memory] 已创建脚本记录: {filename}")
        add_chat_message(ROLE_SYSTEM, f"脚本 {filename} 已保存，summary 异步生成中")
        # 先同步保存“summary生成中”的项目记忆
        # 确保 update_scripts() 返回前，普通 memory 已经持久化
        save_project_memory(
            user_id,
            project_id,
            project_memory
        )
        if llm:
            thread = threading.Thread(target=_generate_summary_in_background,
                                      args=(filename, code, llm, new_hash, dependencies, user_id, project_id, chat_id,
                                            summary_version, callback))
            thread.start()


# 给promote使用的项目上下文，让新生成的脚本能引用旧脚本函数
def get_project_context():
    project_memory = get_project_memory()
    if not project_memory["scripts"]:
        return "目前项目中没有任何脚本，这是第一个脚本，请从零开始生成"

    lines = []

    for filename, data in project_memory["scripts"].items():
        summary = data.get("summary", "（暂无总结）")
        if data.get("summary_stale"):
            summary = f"（注意：以下总结对应的是该脚本的旧版本，最新代码可能已变化）\n{summary}"
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

        dependencies = data.get("dependencies", [])
        if isinstance(dependencies, list) and dependencies:
            dependencies_text_list = []
            for dep in dependencies:
                dep_file = dep.get("file", "未知文件")
                symbols = dep.get("symbols", [])
                symbols_text = ", ".join(symbols) if symbols else "（无符号信息）"
                dependencies_text_list.append(f"{dep_file} -> {symbols_text}")
            dependencies_text = "; ".join(dependencies_text_list)
        else:
            dependencies_text = "（暂无依赖文件）"

        block = f"""文件：{filename}\n[结构]\n{structure_text}\n[功能总结]\n{summary}\n[依赖]{dependencies_text}"""
        lines.append(block)
    
    context = "\n\n".join(lines)
    
    if project_memory["overview"]:
        context = f"项目描述：{project_memory['overview']}\n\n{context}"
    return context
