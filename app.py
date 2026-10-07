import gc
import re
import json
import hashlib
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
from fastapi import UploadFile, File

from config import Config
from shared.utils import get_script_code, list_script_history, list_user_projects, list_chat_histories, \
    generate_chat_id, create_chat_history_file, get_script_version_code, save_docs, create_project_directory, \
    list_users, get_user_projects_path, validate_path_component, get_project_docs_path, get_project_document, \
    list_docs_history, get_document_version, save_global_doc, list_global_documents, get_global_document
from shared.vector_store import index_project_docs, index_global_doc
from v1.agent import create_unified_agent
from v1.memory import (
    set_current_user, set_current_project, set_current_chat,
    get_project_memory, add_chat_message, ROLE_ASSISTANT, current_context,
)
from shared.intent_analyzer import detect_scene_type
from shared.context_builder import build_context
from shared.document_loader import load_single_document
from shared.db import (create_user, get_user_password_hash, is_user_admin, )
from shared.vector_store import get_code_vector_store
from shared.utils import get_project_scripts_path

# ---------- 初始化 Agent ----------
DEFAULT_USER = "test_user"
DEFAULT_PROJECT = "test_project_1"
DEFAULT_CHAT = "test_chat_1"

agent_executor = create_unified_agent()

# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(title="Unity AI Assistant")


# 检查是否管理员
def ensure_admin(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="未设置用户")
    if not is_user_admin(user_id):
        raise HTTPException(status_code=403, detail="需要管理员权限")


# 检查项目是否属于当前用户
def ensure_project_owned(user_id: str, project_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="未设置用户")

    if not project_id:
        raise HTTPException(status_code=400, detail="未设置项目")

    projects = list_user_projects(user_id)

    if project_id not in projects:
        raise HTTPException(
            status_code=403,
            detail="项目不存在或不属于当前用户"
        )


# 删除目录（Windows 上处理文件锁定，带重试）
def _rmtree_retry(path: Path, max_retries: int = 3):
    for i in range(max_retries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return True
        except PermissionError:
            print(f"[_rmtree_retry] 第 {i+1} 次删除失败（可能被占用），重试...")
            gc.collect()
            time.sleep(0.5)
    # 最后一次，失败就抛
    shutil.rmtree(path)
    return True


# 为每一个HTTP请求创建独立的current_context。
# 不同请求之间不会共享同一个dict。
@app.middleware("http")
async def context_middleware(request: Request, call_next):
    user_id = request.headers.get("X-User-ID")
    project_id = request.headers.get("X-Project-ID")
    chat_id = request.headers.get("X-Chat-ID")
    current_context.set({
        "user_id": user_id,
        "project_id": project_id,
        "chat_id": chat_id
    })
    response = await call_next(request)
    return response


# 静态文件目录（如果有额外资源，如图片、字体等）
static_dir = os.path.join(os.path.dirname(__file__), "frontend", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------- 页面路径 ----------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
LOGIN_PATH = os.path.join(FRONTEND_DIR, "login.html")
APP_PATH = os.path.join(FRONTEND_DIR, "index.html")


def _serve_html(path: str) -> HTMLResponse:
    if not os.path.exists(path):
        return HTMLResponse(
            content=f"<h1>Error</h1><p>HTML file not found at {path}</p>",
            status_code=404
        )
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# 根路径 → 登录页
@app.get("/")
async def serve_login():
    return _serve_html(LOGIN_PATH)


# 主界面
@app.get("/app")
async def serve_index():
    return _serve_html(APP_PATH)


# ---------- API 1: 聊天 ----------
@app.post("/api/chat")
async def chat_api(request: dict):
    user_message = request.get("message", "")
    if not user_message:
        return {"reply": "消息不能为空"}

    # 先保存用户消息。
    # 即使后面的 context / Agent 出错，也不能丢失用户本次输入。
    add_chat_message("user", user_message)

    try:
        # 1. 意图识别
        scene_type = detect_scene_type(user_message)

        # 2. 构建上下文
        context = build_context(user_message, scene_type=scene_type)

        # 3. 组装 Agent 输入
        augmented_input = (
            f"注意：请全程使用中文进行推理、输出和选择工具参数。\n\n"
            f"【系统提供的背景信息】\n{context}\n\n"
            f"【用户当前问题】\n{user_message}"
        )

        # 4. 调用 Agent
        response = agent_executor.invoke({"input": augmented_input})

        final_answer = response.get("output", "Agent 未返回有效回答")

    except Exception as e:
        print(f"[Chat] Agent 流程执行失败：{e}")
        final_answer = f"Agent 执行失败：{e}"

    # 无论 Agent/context 是否成功，都保存 assistant 消息
    add_chat_message(ROLE_ASSISTANT, final_answer)

    return {"reply": final_answer}


# ---------- API 2: 获取项目概览 ----------
@app.get("/api/project_info")
def project_info():
    user_id = current_context["user_id"]
    project_id = current_context["project_id"]

    if not user_id or not project_id:
        return {
            "error": "未设置用户或项目"
        }

    try:
        ensure_project_owned(user_id, project_id)
        memory = get_project_memory()
        overview = memory.get("overview", "未设置项目描述")
        scripts = memory.get("scripts", {})
        script_count = len(scripts)

        # 从memory里计算真实统计数据
        dep_count = 0
        method_count = 0
        call_site_count = 0
        latest_updated = "—"
        latest_ts = 0

        # 脚本目录（用于fallback取mtime）
        try:
            scripts_dir = get_project_scripts_path(project_id, user_id)
        except Exception:
            scripts_dir = None

        for fname, info in scripts.items():
            # 1. 依赖关系数
            deps = info.get("dependencies", [])
            if isinstance(deps, list):
                dep_count += len(deps)

            # 2. 总方法数 & 调用点数
            structure = info.get("structure", {})
            if isinstance(structure, dict):
                for cls in structure.get("classes", []):
                    methods = cls.get("methods", [])
                    if isinstance(methods, list):
                        method_count += len(methods)

                call_sites = structure.get("call_sites", [])
                if isinstance(call_sites, list):
                    call_site_count += len(call_sites)

            # 3. 最近更新时间：优先用memory里的last_updated；
            #    没有就用文件系统的mtime（兼容旧数据）
            ts = info.get("last_updated", 0)
            if not ts and scripts_dir is not None:
                try:
                    fpath = scripts_dir / fname
                    if fpath.exists():
                        ts = fpath.stat().st_mtime
                except Exception:
                    ts = 0

            if ts > latest_ts:
                latest_ts = ts
                latest_updated = fname

        # 4. 向量块数量（从ChromaDB直接读）
        vector_chunks = 0
        try:
            vs = get_code_vector_store(project_id, user_id)
            vector_chunks = vs._collection.count()
        except Exception as e:
            print(f"[project_info] 读取向量块数量失败: {e}")
            vector_chunks = 0

        return {
            "overview": overview,
            "script_count": script_count,
            "dep_count": dep_count,
            "vector_chunks": vector_chunks,
            "latest_updated": latest_updated,
            "method_count": method_count,
            "call_site_count": call_site_count,
        }

    except HTTPException as e:
        return {
            "error": e.detail
        }


# ---------- API 3: 获取项目列表 ----------
@app.get("/api/projects")
def list_projects():
    user_id = current_context["user_id"]
    if user_id is None:
        raise ValueError("当前没有设置user_id，请先调用 set_current_user")

    projects = list_user_projects(user_id)
    return {"projects": projects}


# ---------- API 4: 切换项目 ----------
@app.post("/api/switch_project")
def switch_project(request: dict):
    project_id = request.get("project_id")
    if not project_id:
        return {
            "success": False,
            "error": "缺少 project_id"
        }
    try:
        user_id = current_context["user_id"]
        if not user_id:
            return {
                "success": False,
                "error": "未设置用户"
            }

        # 1. 检查项目是否属于当前用户
        try:
            ensure_project_owned(user_id, project_id)
        except HTTPException as e:
            return {
                "success": False,
                "error": e.detail
            }

        # 2. 切换当前项目
        set_current_project(project_id)

        # 3. 获取项目记忆
        memory = get_project_memory()
        overview = memory.get("overview", "")
        scripts = list(memory.get("scripts", {}).keys())

        # 4. 获取当前项目的聊天列表
        chats = list_chat_histories(user_id, project_id)

        # 5. 确定当前聊天
        chat_id = None
        messages = []
        if chats:
            # 有历史聊天，使用第一个聊天
            chat_id = chats[0]["id"]
            # 设置当前聊天
            set_current_chat(chat_id)
            from v1.memory import get_chat_history
            chat_history = get_chat_history()
            # 前端聊天窗口只显示 user / assistant
            messages = [message for message in chat_history if message.get("role") in ("user", "assistant")]
        else:
            # 没有聊天时，创建一个新的空聊天
            chat_id = generate_chat_id()
            create_chat_history_file(user_id, project_id, chat_id)
            set_current_chat(chat_id)
            # 新创建的聊天加入当前返回结果
            chats = [{"id": chat_id, "title": "新会话"}]

        # 6. 返回项目切换后的完整业务状态
        return {
            "success": True,
            "user_id": user_id,
            "project_id": project_id,
            "overview": overview,
            "scripts": scripts,
            "chats": chats,
            "chat_id": chat_id,
            "messages": messages
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)}


# ---------- API 5: 获取会话列表 ----------
@app.get("/api/chats")
def list_chats():
    user_id = current_context["user_id"]
    project_id = current_context["project_id"]
    if not user_id or not project_id:
        return {"chats": []}
    chats = list_chat_histories(user_id, project_id)
    return {"chats": chats}


# ---------- API 6: 新建会话 ----------
@app.post("/api/new_chat")
def new_chat():
    try:
        user_id = current_context["user_id"]
        project_id = current_context["project_id"]
        if not user_id:
            return {
                "success": False,
                "error": "未设置用户"
            }
        if not project_id:
            return {
                "success": False,
                "error": "未设置项目"
            }

        # 生成新的会话 ID
        chat_id = generate_chat_id()
        # 同步创建一个空的聊天历史文件，确保立即能被 /api/chats 查询到
        create_chat_history_file(user_id, project_id, chat_id)
        # 设置当前会话
        set_current_chat(chat_id)
        # 重新获取当前项目的完整会话列表
        chats = list_chat_histories(user_id, project_id)
        return {
            "success": True,
            "user_id": user_id,
            "project_id": project_id,
            "chat_id": chat_id,
            "chats": chats,
            "messages": []
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ---------- API 7: 获取脚本内容及元数据 ----------
@app.get("/api/script/{filename}")
def get_script_info(filename: str):
    try:
        filename = validate_path_component(filename, "filename")
    except ValueError as e:
        return {"error": str(e)}

    project_id = current_context["project_id"]
    user_id = current_context["user_id"]
    if not project_id or not user_id:
        return {"error": "未设置项目或用户"}

    try:
        ensure_project_owned(user_id, project_id)
    except HTTPException as e:
        return {"error": e.detail}

    project_memory = get_project_memory()
    scripts = project_memory.get("scripts", {})
    if filename not in scripts:
        return {"error": "脚本不存在"}
    code = get_script_code(project_id, user_id, filename)
    if code is None:
        return {"error": "无法读取脚本代码"}
    info = scripts[filename]
    # 获取历史版本列表
    versions = list_script_history(project_id, filename, user_id)
    return {
        "filename": filename,
        "code": code,
        "summary": info.get("summary", ""),
        "summary_done": info.get("summary_done", True),
        "summary_stale": info.get("summary_stale", False),
        "structure": info.get("structure", {}),
        "dependencies": info.get("dependencies", []),
        "versions": versions
    }


# ---------- API 8: 加载历史版本  ----------
@app.get("/api/script/{filename}/version/{version_name}")
def get_script_version(filename: str, version_name: str):
    try:
        filename = validate_path_component(filename, "filename")
        version_name = validate_path_component(version_name, "version_name")
    except ValueError as e:
        return {"error": str(e)}

    project_id = current_context["project_id"]
    user_id = current_context["user_id"]
    if not project_id or not user_id:
        return {"error": "未设置项目或用户"}

    try:
        ensure_project_owned(user_id, project_id)
    except HTTPException as e:
        return {"error": e.detail}

    # 历史版本归属检查
    history = list_script_history(project_id, filename, user_id)
    if version_name not in history:
        raise HTTPException(status_code=404, detail="版本不存在")
    # 版本文件保存在scripts目录下
    code = get_script_version_code(project_id,user_id,version_name)
    if code is None:
        return {"error": "版本文件不存在"}

    return {
        "filename":filename,
        "version":version_name,
        "code": code}


# ---------- API 9: 删除脚本 ----------
@app.delete("/api/script/{filename}")
def delete_script(filename: str):
    try:
        filename = validate_path_component(filename, "filename")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    user_id = current_context["user_id"]
    project_id = current_context["project_id"]
    if not user_id or not project_id:
        return {"success": False, "error": "未设置用户或项目"}

    try:
        ensure_project_owned(user_id, project_id)

        project_memory = get_project_memory()
        scripts = project_memory.get("scripts", {})
        if filename not in scripts:
            return {"success": False, "error": "脚本不存在"}

        # 1. 删文件 + 历史版本
        from shared.utils import delete_script_file
        deleted_files = delete_script_file(project_id, filename, user_id)

        # 2. 从 memory 移除（含反向依赖清理）
        from v1.memory import remove_script_from_memory
        remove_script_from_memory(filename)

        # 3. 删向量索引
        try:
            from shared.vector_store import get_code_vector_store
            vs = get_code_vector_store(project_id, user_id)
            vs.delete(where={"$and": [
                {"filename": filename},
                {"project_id": project_id}
            ]})
        except Exception as e:
            print(f"[delete_script] 删向量失败：{e}")

        # 4. 同步保存 memory
        from shared.utils import save_project_memory
        save_project_memory(user_id, project_id, project_memory)

        # 5. 返回剩余脚本
        remaining = list(scripts.keys())

        return {
            "success": True,
            "deleted": filename,
            "deleted_files": deleted_files,
            "scripts": remaining,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- API 10:  上传项目文档  ----------
@app.post("/api/upload_docs")
async def upload_docs(files: list[UploadFile] = File(...)):
    user_id = current_context["user_id"]
    project_id = current_context["project_id"]

    if not user_id or not project_id:
        return {"error": "未设置用户或项目"}

    try:
        ensure_project_owned(user_id, project_id)
    except HTTPException as e:
        return {"error": e.detail}

    results = []

    for file in files:
        try:
            filename = validate_path_component(file.filename, "filename")
            file_data = await file.read()
            # 保存当前文档，同时自动处理历史版本
            file_path = save_docs(project_id, filename, file_data, user_id)
            # 使用统一文档加载函数
            docs = load_single_document(file_path)
            # 写入当前用户 + 当前项目的文档向量库
            index_project_docs(project_id, user_id, filename, docs)

            results.append({
                "filename": filename,
                "success": True,
                "message": f"已索引 {filename}"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "success": True,
        "files": results
    }


# ---------- API 11:  获取脚本列表  ----------
@app.get("/api/scripts")
def list_scripts():
    project_id = current_context["project_id"]
    user_id = current_context["user_id"]
    if not project_id or not user_id:
        return {"scripts": []}

    try:
        ensure_project_owned(user_id, project_id)
    except HTTPException as e:
        return {"scripts": [], "error": e.detail}

    memory = get_project_memory()
    scripts = list(memory.get("scripts", {}).keys())
    print(f"[list_scripts] 项目: {project_id}, 用户: {user_id}, 脚本数量: {len(scripts)}, 脚本列表: {scripts}")
    return {"scripts": scripts}


# ---------- API 12: 切换会话并加载历史 ----------
@app.post("/api/switch_chat")
def switch_chat(request: dict):
    chat_id = request.get("chat_id")
    if not chat_id:
        return {
            "success": False,
            "error": "缺少 chat_id"
        }

    try:
        user_id = current_context["user_id"]
        project_id = current_context["project_id"]
        if not user_id:
            return {
                "success": False,
                "error": "未设置用户"
            }
        if not project_id:
            return {
                "success": False,
                "error": "未设置项目"
            }

        # 校验 chat_id 必须是合法的单一路径组件
        chat_id = validate_path_component(chat_id, "chat_id")

        # 当前用户 + 当前项目下的聊天列表
        chats = list_chat_histories(user_id, project_id)
        # 只允许切换到当前用户当前项目拥有的会话
        valid_chat_ids = {chat["id"] for chat in chats}

        # 防止切换到其他用户 / 其他项目的聊天
        if chat_id not in valid_chat_ids:
            return {
                "success": False,
                "error": "聊天不存在或不属于当前项目"
            }

        # 更新当前会话
        set_current_chat(chat_id)
        # 读取该会话的完整历史消息
        from v1.memory import get_chat_history
        messages = get_chat_history()   # 返回 list of dict: [{"role":..., "message":...}, ...]
        # 过滤掉 system 消息，只返回 user 和 assistant
        filtered_messages = [message for message in messages if message.get("role") in ("user", "assistant")]

        return {
            "success": True,
            "user_id": user_id,
            "project_id": project_id,
            "chat_id": chat_id,
            "chats": chats,
            "messages": filtered_messages
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ---------- API 13: 删除会话 ----------
@app.delete("/api/chat/{chat_id}")
def delete_chat(chat_id: str):
    try:
        chat_id = validate_path_component(chat_id, "chat_id")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    user_id = current_context["user_id"]
    project_id = current_context["project_id"]
    if not user_id or not project_id:
        return {"success": False, "error": "未设置用户或项目"}

    try:
        # 1. 校验会话属于当前用户 + 当前项目
        chats = list_chat_histories(user_id, project_id)
        valid_ids = {c["id"] for c in chats}
        if chat_id not in valid_ids:
            return {"success": False, "error": "会话不存在或不属于当前项目"}

        # 2. 删除文件
        from shared.utils import delete_chat_history_file
        delete_chat_history_file(user_id, project_id, chat_id)

        # 3. 清内存缓存
        from v1.memory import remove_chat_from_memory
        remove_chat_from_memory(user_id, project_id, chat_id)

        # 4. 重新取列表
        remaining = list_chat_histories(user_id, project_id)

        # 5. 如果删的是当前会话 → 切换到第一个会话 / 或新建一个
        new_chat_id = current_context["chat_id"]
        messages = []

        if chat_id == current_context["chat_id"]:
            if remaining:
                new_chat_id = remaining[0]["id"]
                set_current_chat(new_chat_id)
                from v1.memory import get_chat_history
                history = get_chat_history()
                messages = [m for m in history if m.get("role") in ("user", "assistant")]
            else:
                # 没有会话了 → 新建一个空会话
                new_chat_id = generate_chat_id()
                create_chat_history_file(user_id, project_id, new_chat_id)
                set_current_chat(new_chat_id)
                remaining = [{"id": new_chat_id, "title": "新会话"}]

        return {
            "success": True,
            "deleted_chat_id": chat_id,
            "chats": remaining,
            "current_chat_id": new_chat_id,
            "messages": messages,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- API 14: 获取当前会话 ----------
@app.get("/api/current_chat")
def current_chat():
    chat_id = current_context.get("chat_id")
    if not chat_id:
        return {"chat_id": None}
    return {"chat_id": chat_id}


# ---------- API 15: 新建项目 ----------
@app.post("/api/new_project")
def new_project(request: dict):
    project_name = request.get("project_name")
    if not project_name:
        return {
            "success": False,
            "error": "缺少项目名称"
        }
    # 清理项目名，不允许路径分隔符等特殊字符
    import re
    project_name = re.sub(r'[\\/*?:"<>|]', "_", project_name.strip())
    if not project_name:
        return {
            "success": False,
            "error": "项目名称无效"
        }
    user_id = current_context["user_id"]
    if not user_id:
        return {
            "success": False,
            "error": "未设置用户"
        }

    try:
        # 1. 检查项目是否已经存在
        projects = list_user_projects(user_id)
        if project_name in projects:
            return {
                "success": False,
                "error": f"项目 {project_name} 已存在"
            }

        # 2. 创建项目目录结构
        create_project_directory(project_name, user_id)

        # 3. 初始化项目记忆
        from v1.memory import save_project_memory, set_current_chat
        initial_memory = {
            "overview": "",
            "overview_version": 0,
            "scripts": {},
            "reverse_deps": {}
        }
        save_project_memory(user_id, project_name, initial_memory)

        # 4. 切换到新项目
        set_current_project(project_name)

        # 5. 创建新项目的默认聊天
        chat_id = generate_chat_id()
        create_chat_history_file(user_id, project_name, chat_id)

        # 6. 设置当前聊天
        set_current_chat(chat_id)

        # 7. 获取新项目当前状态
        memory = get_project_memory()
        overview = memory.get("overview", "")
        scripts = list(memory.get("scripts", {}).keys())

        # 8. 获取当前项目聊天列表
        chats = list_chat_histories(user_id, project_name)

        # 9. 获取当前用户最新项目列表
        projects = list_user_projects(user_id)

        # 10. 返回完整的新项目初始化状态
        return {
            "success": True,
            "user_id": user_id,
            "projects": projects,
            "project_id": project_name,
            "chat_id": chat_id,
            "overview": overview,
            "scripts": scripts,
            "chats": chats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ---------- API 16: 删除项目 ----------
@app.delete("/api/project/{project_id}")
def delete_project(project_id: str):
    try:
        project_id = validate_path_component(project_id, "project_id")
    except ValueError as e:
        return {"success": False, "error": str(e)}

    user_id = current_context["user_id"]
    if not user_id:
        return {"success": False, "error": "未设置用户"}

    try:
        ensure_project_owned(user_id, project_id)

        # 1. 清内存缓存
        from v1.memory import project_memories, chat_histories
        project_memories.pop((user_id, project_id), None)

        keys_to_delete = [
            k for k in chat_histories.keys()
            if k[0] == user_id and k[1] == project_id
        ]
        for k in keys_to_delete:
            del chat_histories[k]

        # 2. 清向量库 collection（先清 collection，再删目录）
        from shared.vector_store import clear_vector_collection

        code_vs_path = Path(Config.CHROMA_CODE_DIR.format(user_id=user_id)) / project_id
        docs_vs_path = Path(Config.CHROMA_PROJECT_DOCS_DIR.format(user_id=user_id)) / project_id

        if code_vs_path.exists():
            try:
                clear_vector_collection(
                    code_vs_path,
                    Config.CHROMA_CODE_COLLECTION_NAME.format(project_id=project_id)
                )
            except Exception as e:
                print(f"[delete_project] 清代码向量 collection 失败：{e}")

        if docs_vs_path.exists():
            try:
                clear_vector_collection(
                    docs_vs_path,
                    Config.CHROMA_PROJECT_DOCS_COLLECTION_NAME.format(project_id=project_id)
                )
            except Exception as e:
                print(f"[delete_project] 清文档向量 collection 失败：{e}")

        # 3. 删项目目录
        project_path = get_user_projects_path(user_id) / project_id
        if project_path.exists():
            _rmtree_retry(project_path)

        # 4. 删 memory 文件
        memory_file = Path(Config.MEMORY_DIR) / f"{user_id}_{project_id}_memory.json"
        if memory_file.exists():
            memory_file.unlink()

        # 5. 删 chat history 文件
        chat_dir = Path(Config.CHAT_HISTORY_DIR)
        if chat_dir.exists():
            for f in chat_dir.glob(f"{user_id}_{project_id}_*.json"):
                try:
                    f.unlink()
                except Exception as e:
                    print(f"[delete_project] 删会话文件失败：{f.name}，{e}")

        # 6. 删向量库目录（重试）
        for p in [code_vs_path, docs_vs_path]:
            if p.exists():
                try:
                    _rmtree_retry(p)
                except Exception as e:
                    print(f"[delete_project] 删向量库目录失败：{p}，{e}")

        # 7. 剩余项目列表
        projects = list_user_projects(user_id)

        # 8. 没有剩余项目 → 空状态
        if not projects:
            return {
                "success": True,
                "deleted": project_id,
                "projects": [],
                "project_id": None,
                "chat_id": None,
                "overview": "",
                "scripts": [],
                "chats": [],
                "messages": [],
            }

        # 9. 有剩余项目 → 切到第一个
        new_project_id = projects[0]
        set_current_project(new_project_id)

        memory = get_project_memory()
        overview = memory.get("overview", "")
        scripts = list(memory.get("scripts", {}).keys())

        chats = list_chat_histories(user_id, new_project_id)
        chat_id = None
        messages = []

        if chats:
            chat_id = chats[0]["id"]
            set_current_chat(chat_id)
            from v1.memory import get_chat_history
            history = get_chat_history()
            messages = [
                m for m in history
                if m.get("role") in ("user", "assistant")
            ]
        else:
            chat_id = generate_chat_id()
            create_chat_history_file(user_id, new_project_id, chat_id)
            set_current_chat(chat_id)
            chats = [{"id": chat_id, "title": "新会话"}]

        return {
            "success": True,
            "deleted": project_id,
            "projects": projects,
            "project_id": new_project_id,
            "chat_id": chat_id,
            "overview": overview,
            "scripts": scripts,
            "chats": chats,
            "messages": messages,
        }

    except HTTPException as e:
        return {"success": False, "error": e.detail}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ---------- API 17: 获取当前项目 ----------
@app.get("/api/current_project")
def current_project():
    project_id = current_context.get("project_id")
    return {"project_id": project_id}


# ---------- API 18: 获取用户列表 ----------
@app.get("/api/users")
def list_users_api():
    return {"users": list_users()}


# ---------- API 19: 切换用户 ----------
@app.post("/api/switch_user")
def switch_user(request: dict):
    user_id = request.get("user_id")
    if not user_id:
        return {
            "success": False,
            "error": "缺少 user_id"
        }
    try:
        # 1. 切换当前用户
        # 确保用户目录存在
        get_user_projects_path(user_id)
        # 更新当前用户
        set_current_user(user_id)

        # 2. 获取该用户的项目
        projects = list_user_projects(user_id)
        # 没有项目时，用户切换流程到这里结束
        if not projects:
            return {
                "success": True,
                "user_id": user_id,
                "is_admin": is_user_admin(user_id),
                "projects": [],
                "project_id": None,
                "chat_id": None,
                "overview": "",
                "scripts": [],
                "chats": [],
                "messages": []
            }

        # 3. 自动选择该用户的第一个项目
        project_id = projects[0]
        set_current_project(project_id)

        # 4. 获取项目记忆
        memory = get_project_memory()
        overview = memory.get("overview", "")
        scripts = list(memory.get("scripts", {}).keys())

        # 5. 获取该项目的聊天列表
        chats = list_chat_histories(user_id, project_id)

        # 6. 自动选择第一个聊天
        # 如果没有聊天，则创建一个新的空聊天
        chat_id = None
        messages = []
        if chats:
            chat_id = chats[0]["id"]
            set_current_chat(chat_id)
            from v1.memory import get_chat_history
            chat_history = get_chat_history()
            # 前端聊天窗口只显示 user / assistant
            messages = [message for message in chat_history if message.get("role") in ("user", "assistant")]
        else:
            chat_id = generate_chat_id()
            create_chat_history_file(user_id, project_id, chat_id)
            set_current_chat(chat_id)
            # 新创建的聊天加入当前返回结果
            chats = [{"id": chat_id, "title": "新会话"}]

        # 7. 一次性把用户切换后的完整状态返回给前端
        return {
            "success": True,
            "user_id": user_id,
            "is_admin": is_user_admin(user_id),
            # 用户级数据
            "projects": projects,
            # 当前项目
            "project_id": project_id,
            "overview": overview,
            "scripts": scripts,
            # 当前项目的聊天
            "chats": chats,
            "chat_id": chat_id,
            "messages": messages
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ---------- API 20: 获取当前用户 ----------
@app.get("/api/current_user")
def current_user():
    user_id = current_context.get("user_id")
    return {"user_id": user_id}


# ---------- API 21:  获取文档列表  ----------
@app.get("/api/documents")
def get_documents():
    user_id = current_context["user_id"]
    project_id = current_context["project_id"]

    if not user_id or not project_id:
        return {"error": "未设置项目或用户"}

    try:
        ensure_project_owned(user_id, project_id)
        docs_path = get_project_docs_path(project_id, user_id)
        documents = []

        for file_path in docs_path.iterdir():
            if not file_path.is_file():
                continue

            filename = file_path.name

            # 当前文档列表只显示当前版本，不显示历史版本
            is_history = False

            for current_filename in [filename]:
                stem = file_path.stem
                name, ext = os.path.splitext(filename)

                # 判断是否符合：原文件名_YYYYMMDD_HHMMSS_user_id.ext
                pattern = rf"^(.+)_\d{{8}}_\d{{6}}_{re.escape(user_id)}{re.escape(ext)}$"

                if re.match(pattern, filename):
                    is_history = True
                    break

            if not is_history:
                documents.append(filename)

        documents.sort()

        return {
            "success": True,
            "documents": documents
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ---------- API 22: 获取文档内容及元数据 ----------
@app.get("/api/document/{filename}")
def get_document_info(filename: str):
    try:
        filename = validate_path_component(filename, "filename")
    except ValueError as e:
        return {"error": str(e)}

    project_id = current_context["project_id"]
    user_id = current_context["user_id"]

    if not project_id or not user_id:
        return {"error": "未设置项目或用户"}

    try:
        ensure_project_owned(user_id, project_id)
        file_path = get_project_document(project_id, user_id, filename)
        if file_path is None:
            return {"error": "文档不存在"}

        docs = load_single_document(file_path)
        content = "\n\n".join(doc.page_content for doc in docs)
        versions = list_docs_history(project_id, filename, user_id)

        return {
            "success": True,
            "filename": filename,
            "content": content,
            "versions": versions
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ---------- API 23: 获取文档内容及元数据 ----------
@app.get("/api/document/{filename}/version/{version_name}")
def get_document_version_info(filename: str, version_name: str):
    try:
        filename = validate_path_component(filename, "filename")
        version_name = validate_path_component(version_name, "version_name")
    except ValueError as e:
        return {"error": str(e)}

    project_id = current_context["project_id"]
    user_id = current_context["user_id"]

    if not project_id or not user_id:
        return {"error": "未设置项目或用户"}

    try:
        ensure_project_owned(user_id, project_id)
        versions = list_docs_history(project_id, filename, user_id)
        if version_name not in versions:
            return {"error": "该历史版本不属于当前文档"}

        version_path = get_document_version(project_id, user_id, version_name)
        if version_path is None:
            return {"error": "文档版本不存在"}

        docs = load_single_document(version_path)
        content = "\n\n".join(doc.page_content for doc in docs)

        return {
            "success": True,
            "filename": filename,
            "version": version_name,
            "content": content
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ---------- API 24: 删除项目文档（含历史版本 + 向量索引） ----------
@app.delete("/api/document/{filename}")
def delete_document(filename: str):
    try:
        filename = validate_path_component(filename, "filename")
    except ValueError as e:
        return {"error": str(e)}

    project_id = current_context["project_id"]
    user_id = current_context["user_id"]

    if not project_id or not user_id:
        return {"error": "未设置项目或用户"}

    try:
        ensure_project_owned(user_id, project_id)

        docs_path = get_project_docs_path(project_id, user_id)
        deleted = []

        # 1. 删除当前文件
        target = docs_path / filename
        if target.exists() and target.is_file():
            target.unlink()
            deleted.append(filename)

        # 2. 删除所有历史版本
        #    命名规则：name_YYYYMMDD_HHMMSS_userid.ext
        name, ext = os.path.splitext(filename)
        pattern = re.compile(
            rf"^{re.escape(name)}_\d{{8}}_\d{{6}}_{re.escape(user_id)}{re.escape(ext)}$"
        )
        for f in docs_path.iterdir():
            if f.is_file() and pattern.match(f.name):
                f.unlink()
                deleted.append(f.name)

        # 3. 删除向量库里的索引
        try:
            from shared.vector_store import get_project_docs_vector_store
            vs = get_project_docs_vector_store(project_id, user_id)
            vs.delete(
                where={"$and": [
                    {"project_id": project_id},
                    {"doc_id": filename}
                ]}
            )
        except Exception as e:
            print(f"[delete_document] 删除向量索引失败：{e}")

        if not deleted:
            return {"success": False, "error": "文档不存在"}

        return {
            "success": True,
            "filename": filename,
            "deleted": deleted
        }

    except HTTPException as e:
        return {"success": False, "error": e.detail}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- API 25: Global Docs 上传 ----------
@app.post("/api/upload_global_docs")
async def upload_global_docs(files: list[UploadFile] = File(...)):
    # 管理员校验
    user_id = current_context["user_id"]
    try:
        ensure_admin(user_id)
    except HTTPException as e:
        return {"success": False, "error": e.detail}

    results = []
    for file in files:
        try:
            filename = validate_path_component(file.filename, "filename")
            file_data = await file.read()
            file_path = save_global_doc(filename,file_data)
            docs = load_single_document(file_path)
            index_global_doc( filename, docs)
            results.append({"filename": filename, "success": True})

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "success": True,
        "results": results
    }


# ---------- API 26: Global Docs 列表 ----------
@app.get("/api/global_documents")
async def get_global_documents():
    documents = list_global_documents()
    return {
        "success": True,
        "documents": documents
    }


# ---------- API 27: Global Docs 列表 ----------
@app.get("/api/global_document/{filename}")
async def get_global_document_detail(filename: str):
    try:
        filename = validate_path_component(filename, "filename")
    except ValueError as e:
        return {"error": str(e)}
    file_path = get_global_document(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="全局文档不存在")
    docs = load_single_document(file_path)
    content = "\n\n".join(doc.page_content for doc in docs)
    return {
        "success": True,
        "filename": filename,
        "content": content
    }


# ---------- API 28: 删除全局文档 ----------
@app.delete("/api/global_document/{filename}")
async def delete_global_document(filename: str):
    try:
        filename = validate_path_component(filename, "filename")
    except ValueError as e:
        return {"error": str(e)}

    # 管理员校验
    user_id = current_context["user_id"]
    try:
        ensure_admin(user_id)
    except HTTPException as e:
        return {"success": False, "error": e.detail}

    try:
        from shared.utils import get_global_docs_path
        docs_path = get_global_docs_path()
        target = docs_path / filename

        if not target.exists() or not target.is_file():
            return {"success": False, "error": "文档不存在"}

        target.unlink()

        # 删除全局向量库里的索引
        try:
            from shared.vector_store import get_global_vector_store
            vs = get_global_vector_store()
            vs.delete(where={"filename": filename})
        except Exception as e:
            print(f"[delete_global_document] 删除向量索引失败：{e}")

        return {"success": True, "filename": filename}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------- 用户密码处理 ----------
# 暂时沿用原项目的 SHA-256 方案。
# 本阶段只将凭证存储从 JSON 改为 MySQL。
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ---------- API 29: 注册 ----------
@app.post("/api/register")
def register(request: dict):
    user_id = (request.get("user_id") or "").strip()
    password = request.get("password") or ""

    if not user_id or not password:
        return {
            "success": False,
            "error": "用户名和密码不能为空"
        }

    try:
        user_id = validate_path_component(user_id, "user_id")
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }

    if len(password) < 4:
        return {
            "success": False,
            "error": "密码至少 4 位"
        }

    password_hash = _hash_password(password)

    # 写入 MySQL users 表
    created = create_user(user_id, password_hash)

    if not created:
        return {
            "success": False,
            "error": f"用户 {user_id} 已存在"
        }

    # 保留原有文件系统逻辑：
    # 注册后创建该用户的项目根目录
    get_user_projects_path(user_id)

    return {
        "success": True,
        "user_id": user_id
    }


# ---------- API 30: 登录 ----------
@app.post("/api/login")
def login(request: dict):
    user_id = (request.get("user_id") or "").strip()
    password = request.get("password") or ""

    if not user_id or not password:
        return {
            "success": False,
            "error": "用户名和密码不能为空"
        }

    try:
        user_id = validate_path_component(user_id, "user_id")
    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }

    # 从 MySQL 查询用户密码哈希
    stored_password_hash = get_user_password_hash(user_id)

    if stored_password_hash is None:
        return {
            "success": False,
            "error": "用户不存在"
        }

    # 校验密码
    if stored_password_hash != _hash_password(password):
        return {
            "success": False,
            "error": "密码错误"
        }

    # 确保现有文件系统用户目录存在
    get_user_projects_path(user_id)

    return {
        "success": True,
        "user_id": user_id,
        "is_admin": is_user_admin(user_id),
    }


# ---------- 启动 ----------
if __name__ == "__main__":
    print("登录页路径:", LOGIN_PATH)
    print("主界面路径:", APP_PATH)
    uvicorn.run(app, host="0.0.0.0", port=7860)
