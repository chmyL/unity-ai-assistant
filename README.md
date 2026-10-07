# Unity AI Developer Agent

一个面向 Unity 项目开发场景的 AI Developer Agent。项目将 **LLM Agent、RAG、项目记忆、代码向量检索、意图分类、Unity C# 代码生成/修改、文档问答以及用户认证** 集成到一个可运行的 Web 应用中。

项目的核心目标不是单纯提供一个聊天机器人，而是让 Agent 能够理解当前 Unity 项目的上下文，并根据用户请求选择合适的知识、记忆和工具完成开发任务。

---

## 项目特点

- **Unity C# AI 编程助手**：支持 Unity C# 脚本生成、修改和问题分析。
- **LangChain Agent**：基于 Agent + Tool Calling 组织实际开发操作。
- **Hybrid RAG**：结合向量检索与关键词/BM25 检索，提高代码检索的针对性。
- **多级知识库**：全局文档、项目文档、项目代码分别维护向量库。
- **项目级 Memory**：保存项目概览、脚本信息、依赖关系等上下文。
- **会话历史**：不同用户、项目、会话分别保存聊天记录。
- **本地意图分类**：使用规则 + Sentence Transformers 小模型判断请求场景，减少每次请求都调用 LLM 进行意图判断。
- **MySQL 用户系统**：注册、登录、密码校验以及管理员标记由 MySQL `users` 表负责。
- **用户/项目隔离**：项目、脚本、文档、Memory 和代码向量库按照用户与项目进行隔离。
- **ContextVar 请求上下文**：使用 `ContextVar` 管理当前用户、项目和会话上下文，降低多请求场景下的全局状态污染风险。
- **脚本版本历史**：修改脚本时保留历史版本，并限制历史版本数量。
- **文档版本历史**：项目文档同样支持版本管理。
- **Web 前端**：提供登录页以及 Unity AI Developer Agent 主界面。

---

## 整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                         Web Frontend                        │
│            Login Page / Unity AI Assistant UI               │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP API
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI                             │
│                         app.py                              │
│                                                             │
│  用户认证 / 项目 / 会话 / 文档 / 脚本 / Agent 工作流        │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               │                      │
               ▼                      ▼
┌──────────────────────┐   ┌─────────────────────────────────┐
│       MySQL          │   │        LangChain Agent          │
│                      │   │                                 │
│ users                │   │ Intent → Context → Tools → LLM │
│ 用户认证与权限信息    │   └───────────────┬─────────────────┘
└──────────────────────┘                   │
                                           ▼
                         ┌──────────────────────────────────┐
                         │             RAG / Memory          │
                         │                                  │
                         │ Global Docs Vector Store         │
                         │ Project Docs Vector Store        │
                         │ Project Code Vector Store        │
                         │ Hybrid Retrieval / ProjectMemory │
                         └──────────────────────────────────┘

文件系统：
projects/{user_id}/{project_id}/
├── scripts/             # Unity C# 当前脚本 + 历史版本
└── docs/                # 项目文档 + 历史版本

chat_histories/          # 用户 / 项目 / 会话聊天记录
projects_memory/         # 用户 / 项目 Memory
```

---

## 核心工作流程

用户发送请求后，系统主要按照以下流程执行：

```text
用户请求
   │
   ▼
FastAPI /api/chat
   │
   ▼
意图识别
   │
   ├── 规则匹配
   │
   └── 本地语义分类模型
   │
   ▼
根据场景构建 Context
   │
   ├── 全局文档
   ├── 项目文档
   ├── 项目代码
   ├── 项目 Memory
   └── 当前会话上下文
   │
   ▼
LangChain Agent
   │
   ├── search_document
   ├── summarize_document
   ├── create_unity_script
   ├── update_unity_script
   ├── set_project_overview
   └── search_similar_code
   │
   ▼
LLM 推理 / 工具执行
   │
   ▼
返回结果
   │
   ├── 聊天记录保存
   ├── Memory 更新
   ├── 脚本保存 / 版本历史
   └── 代码重新索引
```

---

## 意图分类

项目没有简单地让 LLM 每次直接判断用户意图，而是实现了一个本地的两级分类流程。

### 1. 规则分类

首先使用关键词规则快速判断明显场景，例如：

| 场景 | 典型请求 |
|---|---|
| `code_gen` | 写一个新的 Unity C# 脚本、实现一个新功能 |
| `update` | 修改已有脚本、重构代码、增加功能 |
| `doc_summary` | 总结文档、提取文档重点 |
| `debug` | 分析报错、排查 Bug、定位异常 |
| `hybrid` | 同时涉及分析、修改、生成等多个任务 |

### 2. 本地语义分类

规则无法判断时，使用：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

将用户请求与预定义场景原型进行语义相似度比较，并使用置信度阈值进行兜底判断。

这样可以在保持较低成本的同时，使不同类型的请求进入不同的 Context 构建路径。

---

## RAG 与向量数据库

项目使用 Chroma 保存向量数据，并按照数据用途拆分为多个向量库。

### 1. 全局文档向量库

用于保存系统级 / 公共文档。

```text
chroma_db/docs_db/
```

对应 collection：

```text
global_docs
```

### 2. 项目文档向量库

项目文档按照用户和项目隔离。

```text
chroma_db/{user_id}/project_docs_db/
```

collection：

```text
project_docs_{project_id}
```

### 3. 项目代码向量库

项目代码按照用户和项目隔离，并对 C# 代码进行结构化切分和元数据提取。

```text
chroma_db/{user_id}/code_db/
```

collection：

```text
code_{project_id}
```

代码块 metadata 包含：

- `source_file`
- `class_name`
- `method_name`
- `is_public`
- `call_sites`
- `referenced_symbols`

项目通过 Tree-sitter C# Parser 对代码进行结构分析，为后续代码检索、依赖分析和上下文构建提供结构化信息。

---

## Hybrid RAG

代码检索不仅依赖向量相似度，还提供了 BM25 / 关键词检索路径。

```text
                    用户代码问题
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Vector Retrieval        BM25 Retrieval
              │                     │
              └──────────┬──────────┘
                         ▼
                    Ensemble / Hybrid
                         │
                         ▼
                    Agent Context
```

这种方式可以同时利用：

- 语义相似度
- 精确关键词匹配
- 类名 / 方法名等代码结构信息

从而改善代码库较大时的检索效果。

---

## Unity C# Agent Tools

当前 Agent 暴露的主要工具包括：

| Tool | 功能 |
|---|---|
| `search_document` | 检索项目 / 文档内容 |
| `summarize_document` | 总结文档内容 |
| `create_unity_script` | 根据需求生成 Unity C# 脚本 |
| `update_unity_script` | 修改已有 Unity C# 脚本 |
| `set_project_overview` | 设置 / 更新项目描述 |
| `search_similar_code` | 在当前项目代码库中查找相似代码 |

脚本生成和修改后，会同步更新项目 Memory 与代码向量库，使后续请求可以使用最新代码状态。

---

## 项目 Memory

项目 Memory 以 `(user_id, project_id)` 为核心进行隔离。

Memory 中包含的内容主要包括：

- 项目 overview
- 项目脚本信息
- 脚本 summary
- 脚本 structure
- dependencies
- public interfaces
- reverse dependencies / 引用关系等信息

脚本更新后，Memory 会同步更新相关信息，并重新建立代码索引。

项目 Memory 文件保存在：

```text
projects_memory/
```

---

## 文件系统数据隔离

项目数据并没有全部放进数据库，而是根据数据类型采用不同的存储方式。

### 项目数据

```text
projects/
└── {user_id}/
    └── {project_id}/
        ├── scripts/
        └── docs/
```

因此不同用户的数据天然处于不同目录层级。

### 聊天记录

```text
chat_histories/
```

聊天记录文件名包含用户、项目和会话信息。

### Project Memory

```text
projects_memory/
```

同样按照用户和项目进行区分。

### Vector DB

```text
chroma_db/
├── docs_db/
└── {user_id}/
    ├── code_db/
    └── project_docs_db/
```

这样形成了：

```text
用户
  └── 项目
      ├── 脚本
      ├── 文档
      ├── Chat
      ├── Memory
      └── Vector DB
```

---

# MySQL 用户系统

项目使用 **MySQL** 保存用户账户和认证相关信息。

数据库连接通过 `shared/db.py` 实现，使用 `PyMySQL` 连接 MySQL。

默认数据库配置来自 `.env`：

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=你的数据库密码
DB_NAME=unity_ai_agent
```

实际项目中的数据库名默认为：

```text
unity_ai_agent
```

## users 表

当前项目使用的核心用户表为：

```text
users
```

根据当前数据库实际表结构，字段如下：

| 字段 | 含义 |
|---|---|
| `id` | 用户记录 ID |
| `username` | 用户名 |
| `password_hash` | 密码哈希值 |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `is_admin` | 管理员标记 |

其中：

- `username` 用于用户唯一识别；注册时写入 `users` 表。
- `password_hash` 保存密码哈希，而不是明文密码。
- `is_admin` 用于区分普通用户和管理员用户，当前数据中管理员记录的 `is_admin` 为 `1`。
- `created_at` / `updated_at` 用于记录用户记录的创建与更新时间。

> 当前数据库截图展示了 `users` 表的实际数据结构。截图能够确认字段名称和实际记录，但无法从截图本身确认每个字段的 SQL 数据类型、长度及默认值，因此这里不虚构具体 `VARCHAR` / `INT` / `DATETIME` 等字段定义。

### 当前用户系统流程

```text
注册
 │
 ▼
用户名 / 密码校验
 │
 ▼
SHA-256 密码哈希
 │
 ▼
INSERT INTO users
 │
 ▼
创建用户项目目录
```

登录流程：

```text
用户名 / 密码
      │
      ▼
MySQL 查询 password_hash
      │
      ▼
SHA-256 校验
      │
      ├── 失败 → 登录失败
      │
      └── 成功
           │
           ├── 查询 is_admin
           └── 创建 / 确保用户项目目录
```

项目当前的密码哈希实现使用 SHA-256；如果后续作为生产系统继续使用，密码存储方案仍可以进一步升级为专门的密码哈希算法。

---

## 用户、项目与权限关系

当前项目采用：

```text
MySQL
└── users
    ├── username
    ├── password_hash
    └── is_admin

文件系统
└── projects
    └── user_id
        └── project_id
            ├── scripts
            └── docs
```

也就是说：

- **MySQL 负责用户身份与认证信息**。
- **文件系统负责项目、脚本、文档、聊天和 Memory 数据**。
- **Chroma 负责向量检索数据**。

这种设计避免把大量代码、文档和聊天内容全部塞进关系数据库，同时利用 MySQL 处理账户数据。

---

## ContextVar 请求上下文

项目早期使用全局 `current_context` 保存当前用户、项目和聊天状态。

后续改为基于 Python `ContextVar` 的请求上下文：

```python
_context_var = ContextVar(
    "current_context",
    default=None
)
```

当前上下文包含：

```python
{
    "user_id": ...,
    "project_id": ...,
    "chat_id": ...
}
```

切换用户时会清空项目和会话上下文；切换项目时会清空当前聊天上下文。

这样可以减少不同请求之间共享可变全局状态造成的数据串线问题。

---

## 脚本版本管理

当已有脚本再次保存时，系统不会直接覆盖旧版本，而是先将旧脚本改名保存为历史版本。

示例：

```text
scripts/
├── PlayerController.cs
├── PlayerController_20261008_103000_user123.cs
└── PlayerController_20261008_110000_user123.cs
```

项目当前通过历史版本数量限制控制文件数量，每个脚本最多保留最近的历史版本。

同时提供历史版本读取接口，可以查看指定版本的代码。

---

## 文档系统

项目包含两种文档：

### 全局文档

用于所有用户共享的系统级知识。

```text
upload_global_docs
        │
        ▼
Global Documents
        │
        ▼
Chroma global_docs
```

### 项目文档

属于当前用户当前项目的资料。

```text
User
 │
 ▼
Project
 │
 ▼
Project Documents
 │
 ▼
Project Docs Vector DB
```

项目文档同样支持历史版本管理，并通过用户 / 项目上下文进行隔离。

---

## 前端

前端位于：

```text
frontend/
├── login.html
└── index.html
```

### login.html

负责：

- 用户注册
- 用户登录
- 登录结果处理
- 管理员状态保存

### index.html

负责主系统 UI：

- 用户切换
- 项目切换
- 会话切换
- 项目创建 / 删除
- 文档管理
- 全局文档管理
- 脚本列表
- 脚本代码显示
- 脚本历史版本
- AI Chat
- Agent 工作状态与结果显示

前端主要负责 UI 与 API 调用，项目的核心业务流程集中在 FastAPI 后端。

---

## 项目结构

```text
pythonProject2/
│
├── app.py                         # FastAPI API 与业务流程入口
├── config.py                      # 模型、路径、API 配置
├── requirements.txt               # Python 依赖
│
├── frontend/
│   ├── login.html                  # 登录 / 注册页面
│   └── index.html                  # 主系统页面
│
├── shared/
│   ├── db.py                       # MySQL 用户数据库操作
│   ├── vector_store.py             # Chroma 向量库
│   ├── document_loader.py          # 文档加载
│   ├── advanced_rag.py             # Hybrid RAG / BM25
│   ├── intent_analyzer.py          # 本地意图分类
│   ├── context_builder.py          # Context 构建
│   ├── llm_factory.py              # LLM 创建
│   ├── prompts.py                  # Prompt
│   └── utils.py                    # 通用文件与数据操作
│
├── v1/
│   ├── agent.py                    # LangChain Agent
│   ├── tools.py                    # Agent Tools
│   └── memory.py                   # 项目 Memory / Chat / ContextVar
│
├── v2/
│   ├── state.py                    # 状态结构实验
│   └── graph.py                    # LangGraph 方向的实验代码
│
├── projects/                       # 用户项目数据
├── chat_histories/                 # Chat 历史
├── projects_memory/                # Project Memory
├── chroma_db/                      # Chroma 持久化数据
├── data/sample_docs/               # 示例文档
└── models/                          # 本地模型文件
```

---

## 技术栈

| 分类 | 技术 |
|---|---|
| Backend | Python / FastAPI / Uvicorn |
| AI Framework | LangChain |
| Agent | LangChain Agent / Tools |
| LLM | OpenAI-compatible API / Qwen 系列模型配置 |
| RAG | LangChain + Chroma + BM25 / Hybrid Retrieval |
| Vector DB | ChromaDB |
| Embedding | Sentence Transformers |
| Intent Classification | `paraphrase-multilingual-MiniLM-L12-v2` |
| Code Parsing | Tree-sitter / Tree-sitter C# |
| Database | MySQL |
| MySQL Driver | PyMySQL |
| Frontend | HTML / CSS / JavaScript |
| Project Data | Windows File System |

---

## 安装

建议使用 Python 3.10+ 环境。

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
fastapi
uvicorn
pymysql
langchain-core
langchain-classic
langchain-community
langchain-openai
langchain-huggingface
langchain-chroma
langchain-text-splitters
chromadb
sentence-transformers
tree-sitter
tree-sitter-c-sharp
numpy
gradio
python-dotenv
pydantic
```

---

## 环境变量

项目通过 `.env` 管理敏感配置。

至少需要根据本地环境配置：

```env
# LLM
OPENAI_API_KEY_1=...
OPENAI_BASE_URL_1=...
OPENAI_API_KEY_2=...
OPENAI_BASE_URL_2=...

# MySQL
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=...
DB_NAME=unity_ai_agent
```

不要将真实 API Key、数据库密码等敏感信息提交到 Git 仓库。

---

## 启动

项目入口为：

```bash
python app.py
```

默认启动：

```text
http://127.0.0.1:7860
```

启动后先进入登录页，登录成功后进入 Unity AI Assistant 主界面。

---

## 一个典型使用流程

```text
1. 注册 / 登录用户
        ↓
2. 创建或选择项目
        ↓
3. 上传项目文档
        ↓
4. Agent 建立项目 Context
        ↓
5. 用户提出 Unity 开发需求
        ↓
6. 意图分类
        ↓
7. 检索项目文档 / 项目代码 / Memory
        ↓
8. Agent 选择工具
        ↓
9. 生成或修改 C# 脚本
        ↓
10. 保存最新代码 + 历史版本
        ↓
11. 更新 Project Memory
        ↓
12. 更新代码 Vector DB
```

---

## 项目亮点

### 1. 从“聊天机器人”向“开发 Agent”转变

系统不是只让 LLM 输出代码，而是提供实际工具，让 Agent 可以结合项目现有代码、文档和 Memory 进行操作。

### 2. 多层 RAG

同时处理：

```text
Global Documents
      +
Project Documents
      +
Project Code
      +
Project Memory
      +
Chat History
```

让模型获得比单纯向量问答更完整的项目上下文。

### 3. 代码结构化检索

使用 Tree-sitter C# Parser 提取类、方法、调用点、引用符号等信息，并将这些信息作为 metadata 写入代码向量库。

### 4. 本地小模型意图分类

采用“规则优先 + 本地 embedding 模型兜底”的设计，降低简单请求的 LLM 调用成本，同时保留语义分类能力。

### 5. 用户级 / 项目级隔离

用户认证由 MySQL 负责，项目数据、Memory、Chat 和 Vector DB 按用户与项目隔离，形成较完整的多用户项目工作空间结构。

### 6. 版本化脚本工作流

脚本不是简单覆盖保存，而是保留历史版本，使 AI 修改代码后仍然可以追溯之前的代码状态。

---

## 当前项目定位

这是一个以 **AI Unity Developer Agent** 为核心的个人项目，重点展示：

- LLM Agent 设计
- LangChain Tool / Agent 集成
- RAG 系统设计
- Hybrid Retrieval
- 向量数据库
- 本地语义意图分类
- C# 代码结构分析
- Project Memory
- 多用户数据隔离
- MySQL 用户认证
- FastAPI 后端设计
- 前后端业务流程整合

项目重点不是单一模型效果，而是展示如何将 **LLM、Agent、RAG、Memory、数据库、文件系统和 Web API** 组合成一个实际可运行的 AI Developer Tool。

---

## 后续可扩展方向

当前版本已经具备完整的核心功能。如果继续产品化，可以进一步考虑：

- 更安全的密码存储方案
- 更完善的权限系统
- JWT / Session 等正式认证机制
- 更细粒度的项目成员权限
- LangGraph 状态机进一步接管 Agent 工作流
- 更完善的自动化测试
- Docker / 部署配置
- Unity Editor 插件化集成

这些属于后续产品化方向，并不是当前核心功能运行所必需的部分。
