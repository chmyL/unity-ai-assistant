import json
import os
import re
import threading
import time
from pathlib import Path, PureWindowsPath
from datetime import datetime
import tree_sitter_c_sharp as tscs
from tree_sitter import Parser, Language
from config import Config


HISTORY_LIMIT = 5  # 每个脚本最多保留历史版本数量
HISTORY_LIMIT_DOCS = 3  # 每个文档最多保留历史版本数量


# 生成不会因同一时间内连续保存而冲突的历史版本文件名。
def generate_history_filename(path, name, ext, user_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    counter = 1

    while True:
        history_filename = (f"{name}_{timestamp}_{user_id}_{counter:06d}{ext}")
        history_path = path / history_filename

        if not history_path.exists():
            return history_filename

        counter += 1


# 校验只能作为单个路径组件使用的字符串。
def validate_path_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}不能为空")

    value = value.strip()

    if "\x00" in value:
        raise ValueError(f"{field_name}包含非法字符")

    if value in {".", ".."}:
        raise ValueError(f"{field_name}不能是 . 或 ..")

    if "/" in value or "\\" in value:
        raise ValueError(f"{field_name}不能包含路径分隔符")

    windows_path = PureWindowsPath(value)

    if windows_path.is_absolute() or windows_path.drive or windows_path.root:
        raise ValueError(f"{field_name}不能是绝对路径")

    if windows_path.name != value:
        raise ValueError(f"{field_name}必须是单个路径组件")

    return value


def ensure_unique_filename(filename: str, scripts: dict):
    if filename not in scripts:
        return filename
    name, ext = os.path.splitext(filename)
    i = 2
    while f"{name}{i}{ext}" in scripts:
        i += 1
    return f"{name}{i}{ext}"


# 代码切割（为summary提供）
def smart_code_split(code: str, min_length=200):
    chunks = re.split(r'(?=\n\s*(?:public|private|protected)?\s*(?:class|void|IEnumerator)\s)', code)
    chunks = [c.strip() for c in chunks if c and len(c.strip()) >= min_length]
    # 脚本太短时整段返回
    if not chunks and code and code.strip():
        chunks = [code.strip()]
    return chunks


#  与 smart_code_split 逻辑相同，但返回 (start_byte, end_byte, chunk_text) 三元组列表。
def smart_code_split_with_positions(code: str, min_length=200, max_length=800):
    # 正则匹配分隔符位置（类/方法声明之前）
    pattern = re.compile(r'(?=\n\s*(?:public|private|protected)?\s*(?:class|void|IEnumerator)\s)')
    splits = list(pattern.finditer(code))
    chunks = []
    prev = 0
    for m in splits:
        start = m.start()
        if start > prev:
            chunk_text = code[prev:start].strip()
            if len(chunk_text) >= min_length:
                chunks.append((prev, start, chunk_text))
        prev = start
    # 最后一块
    if prev < len(code):
        chunk_text = code[prev:].strip()
        if len(chunk_text) >= min_length:
            chunks.append((prev, len(code), chunk_text))
    # 如果没有切分点（例如代码太短或没有匹配到关键字），则整个代码作为一个块
    if not chunks:
        chunk_text = code.strip()
        if chunk_text:
            chunks.append((0, len(code), chunk_text))

    # 2. 对过长块按行切分
    final_chunks = []
    for start_byte, end_byte, text in chunks:
        if len(text) <= max_length:
            final_chunks.append((start_byte, end_byte, text))
        else:
            # 获取该块对应的原始子串
            block = code[start_byte:end_byte]
            lines = block.splitlines(keepends=True)
            sub_chunks = []
            sub_start = start_byte
            sub_text = ""
            for line in lines:
                if len(sub_text) + len(line) > max_length and sub_text:
                    # 保存当前子块
                    sub_end = sub_start + len(sub_text)
                    final_chunks.append((sub_start, sub_end, sub_text.strip()))
                    sub_start = sub_end
                    sub_text = line
                else:
                    sub_text += line
            if sub_text:
                sub_end = sub_start + len(sub_text)
                final_chunks.append((sub_start, sub_end, sub_text.strip()))
    return final_chunks


# 向量库代码切割（返回格式: [{"text": str, "metadata": dict}, ...]
# metadata包含：source_file, class_name, method_name, is_public, call_sites (list), referenced_symbols (set)）
def split_code_with_metadata(code: str, filename: str, chunk_size: int = 1500):
    # 正则法切分块
    raw_chunks = smart_code_split_with_positions(code, min_length=200)
    if not raw_chunks:
        return []

    # 元数据获取
    language = Language(tscs.language())
    parser = Parser(language)
    tree = parser.parse(code.encode('utf-8'))
    root = tree.root_node

    class_ranges = []
    method_ranges = []
    call_sites_by_node = {}

    def collect_nodes(node, current_class=None):
        if node.type == "class_declaration":
            class_name = None
            for child in node.children:
                if child.type == "identifier":
                    class_name = code[child.start_byte:child.end_byte]
                    break
            if class_name:
                class_ranges.append((node.start_byte, node.end_byte, class_name))
                for child in node.children:
                    collect_nodes(child, class_name)
        elif node.type == "method_declaration":
            method_name = None
            is_public = False
            for child in node.children:
                if child.type == "identifier":
                    method_name = code[child.start_byte:child.end_byte]
                elif child.type == "modifier" and code[child.start_byte:child.end_byte] == "public":
                    is_public = True
            if method_name:
                method_ranges.append((node.start_byte, node.end_byte, current_class, method_name, is_public))
                for child in node.children:
                    collect_nodes(child, current_class)
        elif node.type == "invocation_expression":
            call_text = code[node.start_byte:node.end_byte].strip()
            if call_text and len(call_text) < 200:
                call_sites_by_node.setdefault((node.start_byte,node.end_byte), []).append(call_text)
        else:
            for child in node.children:
                collect_nodes(child, current_class)
    collect_nodes(root)

    # 为每个切分块添加元数据
    chunks_metadata = []
    for start_byte, end_byte, chunk_text in raw_chunks:
        # 确定类名
        class_name = None
        for cs, ce, cn in class_ranges:
            if cs <= start_byte and ce >= end_byte:
                class_name = cn
                break
        # 确定方法名及公开性
        method_name = None
        is_public = False
        for ms, me, mc, mn, mp in method_ranges:
            if ms <= start_byte and me >= end_byte and (class_name is None or mc == class_name):
                method_name = mn
                is_public = mp
                break

        # 收集 chunk 内的调用点（简单通过文本匹配，也可以用语法树重新解析该 chunk）
        call_sites_in_chunk = []
        for (pos_start, pos_end), texts in call_sites_by_node.items():
            if start_byte <= pos_start <= end_byte:
                call_sites_in_chunk.extend(texts)
        call_sites_in_chunk = list(dict.fromkeys(call_sites_in_chunk))  # 去重

        # 提取被引用的符号（从调用点中提取方法名，简单取左括号前的单词）
        referenced_symbols = set()
        for call in call_sites_in_chunk:
            # 例如 "Debug.Log(...)" -> "Log", "Move()" -> "Move"
            m = re.match(r'(\w+)(?:\.(\w+))?\(', call)
            if m:
                if m.group(2):
                    referenced_symbols.add(m.group(2))
                else:
                    referenced_symbols.add(m.group(1))

        metadata = {
            "source_file": filename,
            "class_name": class_name,
            "method_name": method_name,
            "is_public": is_public,
            "call_sites": call_sites_in_chunk,
            "referenced_symbols": list(referenced_symbols)
        }
        chunks_metadata.append({
            "text": chunk_text,
            "metadata": metadata
        })

    return chunks_metadata


# 获取项目根目录
def get_projects_root_path():
    path = Path(Config.PROJECT_ROOT) / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


# 获取用户目录
def get_user_projects_path(user_id: str):
    user_id = validate_path_component(user_id, "user_id")
    path = get_projects_root_path() / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# 创建新项目目录
def create_project_directory(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    project_path = get_user_projects_path(user_id) / project_id
    if project_path.exists():
        return project_path

    project_path.mkdir(parents=True, exist_ok=True)
    get_project_scripts_path(project_id, user_id)
    get_project_docs_path(project_id, user_id)

    return project_path


# 用户项目列表
def list_user_projects(user_id: str):
    projects_path = get_user_projects_path(user_id)

    return [
        path.name
        for path in projects_path.iterdir()
        if path.is_dir()
    ]


# 用户列表
def list_users():
    projects_root = get_projects_root_path()

    return [
        path.name
        for path in projects_root.iterdir()
        if path.is_dir()
    ]


# 获取脚本路径
def get_project_scripts_path(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    path = get_user_projects_path(user_id) / project_id / "scripts"
    path.mkdir(parents=True, exist_ok=True)
    return path


# 保存脚本
def save_scripts(project_id: str, filename: str, code: str, user_id: str):
    if code is None or not str(code).strip():
        raise ValueError("拒绝保存空脚本内容")
    code = str(code)
    filename = validate_path_component(filename, "filename")
    path = get_project_scripts_path(project_id, user_id)
    file_path = path / filename
    # 缩水保护：新代码不能比旧代码短太多
    if file_path.exists():
        old_code = file_path.read_text(encoding="utf-8")
        old_len = len(old_code.strip())
        new_len = len(code.strip())

        if old_len > 300 and new_len < old_len * 0.4:
            raise ValueError(
                f"拒绝保存：新代码长度({new_len}) 远小于旧代码({old_len})，"
                f"疑似只传入了片段。请提供完整文件内容。"
            )

        if re.search(r'\bclass\s+\w+', old_code) and not re.search(r'\bclass\s+\w+', code):
            raise ValueError(
                "拒绝保存：旧代码含 class 定义，新代码没有，疑似片段。"
            )
    if file_path.exists():
        name, ext = file_path.stem, file_path.suffix
        history_filename = generate_history_filename(path, name, ext, user_id)
        history_path = path / history_filename
        file_path.rename(history_path)
        print(f"[历史版本] 已保存: {history_filename}")

        # 清理过多历史版本，只保留最近 HISTORY_LIMIT 个
        pattern = re.compile(rf"^{re.escape(name)}_\d{{8}}_\d{{6}}_{re.escape(user_id)}_\d{{6}}{re.escape(ext)}$")
        histories = [f for f in path.iterdir() if pattern.match(f.name)]
        # 按时间戳排序（旧到新）
        histories.sort(key=lambda f: f.name)
        while len(histories) > HISTORY_LIMIT:
            to_delete = histories.pop(0)
            to_delete.unlink()
            print(f"[历史版本] 删除过期文件: {to_delete.name}")

    file_path.write_text(code, encoding="utf-8")
    print(f"[最新版本] 已保存: {filename}")
    return file_path


# 删除脚本文件（含所有历史版本）
def delete_script_file(project_id: str, filename: str, user_id: str) -> list:
    filename = validate_path_component(filename, "filename")
    path = get_project_scripts_path(project_id, user_id)
    deleted = []

    # 删当前文件
    target = path / filename
    if target.exists() and target.is_file():
        target.unlink()
        deleted.append(filename)

    # 删所有历史版本
    name, ext = os.path.splitext(filename)
    pattern = re.compile(
        rf"^{re.escape(name)}_\d{{8}}_\d{{6}}_{re.escape(user_id)}_\d{{6}}{re.escape(ext)}$"
    )
    for f in path.iterdir():
        if f.is_file() and pattern.match(f.name):
            f.unlink()
            deleted.append(f.name)

    return deleted


# 从文件系统读取脚本的原始代码
def get_script_code(project_id: str, user_id: str, filename: str):
    filename = validate_path_component(filename, "filename")
    path = get_project_scripts_path(project_id, user_id) / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


# 查看历史版本脚本代码
def list_script_history(project_id: str, filename: str, user_id: str):
    filename = validate_path_component(filename, "filename")
    path = get_project_scripts_path(project_id, user_id)
    name, ext = os.path.splitext(filename)
    pattern = re.compile(rf"^{re.escape(name)}(_\d{{8}}_\d{{6}}_{re.escape(user_id)}_\d{{6}})?{re.escape(ext)}$")
    files = [f.name for f in path.iterdir() if pattern.match(f.name)]
    files.sort()  # 从旧到新
    return files


# 获取指定历史版本代码
def get_script_version_code(project_id: str, user_id: str, version_name: str):
    version_name = validate_path_component(version_name, "version_name")
    path = get_project_scripts_path(project_id, user_id)
    version_path = path / version_name
    if version_path.exists() and version_path.is_file():
        return version_path.read_text(encoding="utf-8")
    return None


# 获取全局文档目录
def get_global_docs_path():
    path = Path(Config.GLOBAL_DOCS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


# 保存全局文档
def save_global_doc(filename, file_data):
    validate_path_component(filename, "filename")
    docs_path = get_global_docs_path()
    file_path = os.path.join(docs_path, filename)
    with open(file_path, "wb") as f:
        f.write(file_data)
    return file_path


# 获取全局文档文件路径
def get_global_document(filename):
    validate_path_component(filename, "filename")
    docs_path = get_global_docs_path()
    file_path = os.path.join(docs_path, filename)
    if not os.path.isfile(file_path):
        return None
    return file_path


# 获取所有全局文档
def list_global_documents():
    docs_path = get_global_docs_path()
    documents = []
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if os.path.isfile(file_path):
            documents.append(filename)
    return sorted(documents)


# 获取文档路径
def get_project_docs_path(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    path = get_user_projects_path(user_id) / project_id / "docs"
    path.mkdir(parents=True, exist_ok=True)
    return path


# 保存文档
def save_docs(project_id: str, filename: str, file_data, user_id: str):
    filename = validate_path_component(filename, "filename")
    path = get_project_docs_path(project_id, user_id)
    file_path = path / filename
    if file_path.exists():
        name, ext = file_path.stem, file_path.suffix
        history_filename = generate_history_filename(path, name, ext, user_id)
        history_path = path / history_filename
        file_path.rename(history_path)
        print(f"[历史版本] 已保存: {history_filename}")

        # 清理过多历史版本，只保留最近 HISTORY_LIMIT 个
        pattern = re.compile(rf"^{re.escape(name)}_\d{{8}}_\d{{6}}_{re.escape(user_id)}_\d{{6}}{re.escape(ext)}$")
        histories = [f for f in path.iterdir() if pattern.match(f.name)]
        # 按时间戳排序（旧到新）
        histories.sort(key=lambda f: f.name)
        while len(histories) > HISTORY_LIMIT_DOCS:
            to_delete = histories.pop(0)
            to_delete.unlink()
            print(f"[历史版本] 删除过期文件: {to_delete.name}")

    if isinstance(file_data, bytes):
        file_path.write_bytes(file_data)
    else:
        file_path.write_text(file_data, encoding="utf-8")
    print(f"[最新版本] 已保存: {filename}")
    return file_path


# 从文件系统读取项目文档
def get_project_document(project_id: str, user_id: str, filename: str):
    filename = validate_path_component(filename, "filename")
    path = get_project_docs_path(project_id, user_id) / filename
    if path.exists():
        return path
    return None


# 查看历史版本文档
def list_docs_history(project_id: str, filename: str, user_id: str):
    filename = validate_path_component(filename, "filename")
    path = get_project_docs_path(project_id, user_id)
    name, ext = os.path.splitext(filename)
    pattern = re.compile(rf"^{re.escape(name)}(_\d{{8}}_\d{{6}}_{re.escape(user_id)}_\d{{6}})?{re.escape(ext)}$")
    files = [f.name for f in path.iterdir() if pattern.match(f.name)]
    files.sort()  # 从旧到新
    return files


# 获取指定历史版本文档
def get_document_version(project_id: str, user_id: str, version_name: str):
    version_name = validate_path_component(version_name, "version_name")
    path = get_project_docs_path(project_id, user_id)
    version_path = path / version_name
    if version_path.exists() and version_path.is_file():
        return version_path
    return None


def on_summary_done(filename, user_id=None, project_id=None):
    print(f"[回调] {filename} summary 已生成完成！")


# 将LLM输出解析成filename+description
def parse_update_input(text: str, default_file: str = None):
    # 尝试直接匹配 文件名.cs
    match = re.search(r'([A-Za-z0-9_]+\.cs)', text)
    if match:
        filename = match.group(1)
        # 去掉文件名部分，剩下的作为描述
        description = text.replace(filename, '').strip()
        # 如果描述以冒号或句号开头，清理一下
        description = re.sub(r'^[：:。，\s]+修改需求[：:]?\s*', '', description).strip()
    else:
        # 尝试从“文件名：xxx.cs”中提取
        match = re.search(r'文件名\s*[：:]\s*([A-Za-z0-9_]+\.cs)', text)
        if match:
            filename = match.group(1)
            # 去掉“文件名：xxx.cs。修改需求：”这部分，保留剩余描述
            description = re.sub(r'文件名\s*[：:]\s*[A-Za-z0-9_]+\.cs[。，\s]*修改需求[：:]?\s*', '', text).strip()
        else:
            filename = default_file or "Unknown.cs"
            description = text.strip()

    return {"filename": filename, "description": description}


# 保存项目记忆
def save_project_memory(user_id: str, project_id: str, project_memory: dict):
    try:
        user_id = validate_path_component(user_id, "user_id")
        project_id = validate_path_component(project_id, "project_id")
        memory_dir = Path(Config.MEMORY_DIR)
        memory_dir.mkdir(parents=True, exist_ok=True)
        path = memory_dir / f"{user_id}_{project_id}_memory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(project_memory, f, ensure_ascii=False, indent=2)
        print(f"[Memory 保存完成] {user_id}/{project_id}")
    except Exception as e:
        print(f"[Memory 保存失败] {user_id}/{project_id}：{e}")


# 加载项目记忆
def load_project_memory(user_id: str, project_id: str):
    user_id = validate_path_component(user_id, "user_id")
    project_id = validate_path_component(project_id, "project_id")
    memory_dir = Path(Config.MEMORY_DIR)
    path = memory_dir / f"{user_id}_{project_id}_memory.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"scripts": {}, "overview": "", "reverse_deps": {}}


# # 保存对话历史
# def save_chat_history(user_id: str, project_id: str, chat_id: str, history: list):
#     user_id = validate_path_component(user_id, "user_id")
#     project_id = validate_path_component(project_id, "project_id")
#     chat_history_dir = Path(Config.CHAT_HISTORY_DIR)
#     chat_history_dir.mkdir(parents=True, exist_ok=True)
#     path = chat_history_dir / f"{user_id}_{project_id}_{chat_id}.json"
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(history, f, ensure_ascii=False, indent=2)


def generate_chat_id():
    return str(int(time.time() * 1000))


# 创建聊天文件(创建新对话)
def create_chat_history_file(user_id: str, project_id: str, chat_id: str):
    user_id = validate_path_component(user_id, "user_id")
    project_id = validate_path_component(project_id, "project_id")
    chat_id = validate_path_component(chat_id, "chat_id")
    chat_history_dir = Path(Config.CHAT_HISTORY_DIR)
    chat_history_dir.mkdir(parents=True, exist_ok=True)
    history_file = (chat_history_dir / f"{user_id}_{project_id}_{chat_id}.json")

    if not history_file.exists():
        history_file.touch()

    return history_file


# 删除聊天历史文件
def delete_chat_history_file(user_id: str, project_id: str, chat_id: str) -> bool:
    user_id = validate_path_component(user_id, "user_id")
    project_id = validate_path_component(project_id, "project_id")
    chat_id = validate_path_component(chat_id, "chat_id")
    chat_history_dir = Path(Config.CHAT_HISTORY_DIR)
    history_file = chat_history_dir / f"{user_id}_{project_id}_{chat_id}.json"

    if history_file.exists():
        history_file.unlink()
        return True
    return False


# 聊天列表历史
def list_chat_histories(user_id: str, project_id: str):
    user_id = validate_path_component(user_id, "user_id")
    project_id = validate_path_component(project_id, "project_id")
    chat_history_dir = Path(Config.CHAT_HISTORY_DIR)
    if not chat_history_dir.exists():
        return []

    pattern = f"{user_id}_{project_id}_*.json"
    chats = []
    for path in chat_history_dir.glob(pattern):
        chat_id = path.stem.replace(f"{user_id}_{project_id}_", "", 1)
        messages = load_chat_history(user_id, project_id, chat_id)
        preview = ""
        if messages:
            first_message = messages[0]
            preview = first_message.get("message", "")[:50]
        chats.append({
            "id": chat_id,
            "preview": preview
        })

    return chats


# 加载对话历史
def load_chat_history(user_id: str, project_id: str, chat_id: str):
    user_id = validate_path_component(user_id, "user_id")
    project_id = validate_path_component(project_id, "project_id")
    chat_id = validate_path_component(chat_id, "chat_id")
    chat_history_dir = Path(Config.CHAT_HISTORY_DIR)
    path = chat_history_dir / f"{user_id}_{project_id}_{chat_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        decoder = json.JSONDecoder()
        messages = []
        pos = 0
        while pos < len(content):
            # 跳过空白字符（空格、换行等）
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            try:
                obj, idx = decoder.raw_decode(content, pos)
                messages.append(obj)
                pos = idx
            except json.JSONDecodeError:
                # 遇到损坏的数据则停止解析
                break
        return messages
    return []


# 异步写入
def async_write(func, *args, **kwargs):
    thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    thread.start()


# 逐条保存对话历史
def append_chat_message(user_id: str, project_id: str, chat_id: str, message: dict):
    user_id = validate_path_component(user_id, "user_id")
    project_id = validate_path_component(project_id, "project_id")
    chat_id = validate_path_component(chat_id, "chat_id")
    chat_history_dir = Path(Config.CHAT_HISTORY_DIR)
    chat_history_dir.mkdir(parents=True, exist_ok=True)
    path = chat_history_dir / f"{user_id}_{project_id}_{chat_id}.json"

    def write():
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False, indent=2) + "\n")
    async_write(write)


# 在max_len字符内尽量保留完整的句子（以 。！？\n 为分隔）
def _soft_truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    # 在截断范围内找最后一个句子结束符
    match = re.search(r'(?<=[。！？\n])[^。！？\n]*$', truncated)
    if match and match.start() > 0:
        return truncated[:match.start()] + " ..."
    # 如果找不到，就硬截断
    return truncated + " ..."


# 提取代码中的所有标识符
def extract_all_identifiers(code: str) -> set:
    language = Language(tscs.language())
    parser = Parser(language)
    tree = parser.parse(code.encode('utf-8'))
    root = tree.root_node
    identifiers = set()

    def traverse(node):
        if node.type == "identifier":
            identifiers.add(code[node.start_byte:node.end_byte])
        for child in node.children:
            traverse(child)

    traverse(root)
    return identifiers


# 获取项目中所有已定义的符号（类名 + 方法名）
def get_all_project_symbols() -> set:
    from v1.memory import get_project_memory
    project_memory = get_project_memory()
    symbols = set()
    for data in project_memory['scripts'].values():
        structure = data.get('structure', {})
        if isinstance(structure, dict):
            for cls in structure.get('classes', []):
                symbols.add(cls['name'])
                symbols.update(cls.get('methods', []))
    return symbols