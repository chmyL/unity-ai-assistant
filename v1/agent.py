import os
from shared.context_builder import build_context
from shared.document_loader import load_documents
from shared.intent_analyzer import detect_scene_type
from shared.llm_factory import create_llm
from config import Config
from langchain_classic.agents import AgentExecutor, create_react_agent
from shared.vector_store import index_project_docs, get_code_retriever
from v1.tools import TOOLS
from v1.memory import set_current_project, add_chat_message, set_current_chat, set_current_user, ROLE_ASSISTANT, \
    current_context, get_project_memory
from langchain_core.prompts import PromptTemplate
from shared.prompts import CHINESE_REACT_TEMPLATE


def create_unified_agent():
    llm = create_llm(
        model_name=Config.DEFAULT_LLM_MODEL,
        temperature=Config.TEMPERATURE
    )
    prompt = PromptTemplate.from_template(CHINESE_REACT_TEMPLATE)

    agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )
    return agent_executor


if __name__ == "__main__":
    print("正在初始化统一 Agent...")

    # 1. 必须提前设置 user_id，因为后续操作需要
    set_current_user("test_user")

    # 2. 设置项目和会话
    set_current_project("test_project_1")
    set_current_chat("test_chat_1")

    project_memory = get_project_memory()
    print(project_memory)

    print("正在索引项目文档...")
    try:
        all_docs = load_documents()  # 默认加载 Config.DOCUMENTS_DIR 下的所有文档
        if not all_docs:
            print("⚠️ 未找到任何文档，请检查 sample_docs 目录")
        else:
            # 按 metadata 中的 source 字段分组（每个文件一个 doc_id）
            source_docs = {}
            for doc in all_docs:
                source = doc.metadata.get("source", "unknown")
                # 仅保留文件名作为 doc_id
                doc_id = os.path.basename(source).replace(".docx", "")
                if doc_id not in source_docs:
                    source_docs[doc_id] = []
                source_docs[doc_id].append(doc)

            for doc_id, docs_list in source_docs.items():
                index_project_docs(
                    project_id="test_project_1",
                    user_id="test_user",
                    doc_id=doc_id,
                    documents=docs_list
                )
                print(f"  已索引：{doc_id} ({len(docs_list)} 个片段)")
    except Exception as e:
        print(f"文档索引失败：{e}")
        print("请确保已安装 unstructured 库：pip install unstructured")

    print("✅ 文档索引完成\n")

    print("正在初始化代码向量库...")

    try:
        retriever = get_code_retriever(
            project_id="test_project_1",
            user_id="test_user",
            k=1
        )

        # 强制触发一次 retrieval
        retriever.invoke("初始化")

        print("✅ 代码向量库初始化完成")

    except Exception as e:
        print(f"⚠️ 代码向量库初始化失败: {e}")

    # 3. 初始化 Agent
    agent_executor = create_unified_agent()

    test_cases = [
        # "我们的项目是一个2D平台跳跃游戏，玩家可以切换维度来解谜和战斗，主要包含角色控制、敌人AI和道具系统。",
        "生成一个能控制角色移动和跳跃的 PlayerController 脚本",
        # "总结一下我上传的所有文档，重点说明核心玩法",
        # "写一个管理道具 Inventory 的脚本，可以添加和移除物品",
        # "修改 Inventory.cs，添加一个物品数量限制的功能",
        # "检查 PlayerController.cs 中跳跃逻辑是否有问题，比如二段跳判定",
    ]
    for query in test_cases:
        print("\n" + "=" * 80)
        print(f"📥 用户输入：{query}")

        # 记录用户消息
        add_chat_message("user", query)

        # 4. 意图分析：自动判定场景类型
        scene_type = detect_scene_type(query)
        print(f"🧠 检测到场景类型：{scene_type}")

        # 5. 动态构建上下文（按场景加载不同的知识库/记忆）
        context = build_context(query, scene_type=scene_type)
        print(f"📚 上下文片段长度：{len(context)} 字符")

        # 6. 将上下文注入到 Agent 的输入中（拼接到用户问题前）
        augmented_input = (
            f"注意：请全程使用中文进行推理、输出和选择工具参数。\n\n"
            f"【系统提供的背景信息】\n"
            f"{context}\n\n"
            f"【用户当前问题】\n"
            f"{query}"
        )

        # 7. 流式执行 Agent
        final_output = ""
        for chunk in agent_executor.stream({"input": augmented_input}):
            if "intermediate_steps" in chunk:
                action, observation = chunk["intermediate_steps"][-1]
                if action and hasattr(action, 'tool'):
                    # 只显示工具名和简短输入
                    desc = str(action.tool_input)[:80]
                    print(f"\n🔧 调用工具：{action.tool} | 输入：{desc}...")
                if observation:
                    # 截断过长的返回内容（比如完整代码）
                    obs_text = str(observation)
                    if len(obs_text) > 100:
                        obs_text = obs_text[:100] + "..."
                    print(f"📄 结果：{obs_text}")
            elif "output" in chunk:
                final_output += chunk["output"]
                print(chunk["output"], end="", flush=True)
            else:
                for key, value in chunk.items():
                    if value and key not in ["intermediate_steps", "output"]:
                        print(f"\n{key}: {value}")
        if final_output:
            add_chat_message(ROLE_ASSISTANT, final_output)

        print("\n" + "-" * 40)
        print("✅ 单轮执行完成")

    print("\n" + "=" * 80)
    print("🎉 所有测试用例执行完毕！")
