import json
from pathlib import Path
from shared.llm_factory import create_embedding
from langchain_chroma import Chroma
from config import Config
import shutil
from shared.document_loader import load_documents, split_document, load_single_document
from shared.utils import smart_code_split, split_code_with_metadata, validate_path_component, get_project_docs_path, \
    get_project_scripts_path


def get_global_vector_store():
    persist_directory = Path(Config.CHROMA_GLOBAL_DOCS_DIR)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=create_embedding(),
        collection_name=Config.CHROMA_GLOBAL_DOCS_COLLECTION_NAME
    )


def get_global_retriever(k=6):
    vector_store = ensure_global_vector_store()
    if vector_store is None:
        return None
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


def get_code_vector_store(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    user_id = validate_path_component(user_id, "user_id")
    persist_directory = (Path(Config.CHROMA_CODE_DIR.format(user_id=user_id)) / project_id)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=create_embedding(),
        collection_name=Config.CHROMA_CODE_COLLECTION_NAME.format(project_id=project_id)
    )


def get_code_retriever(project_id: str, user_id: str, k=6):
    vector_store = ensure_code_vector_store(project_id,user_id)
    if vector_store is None:
        return None
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


# 精确检索（如按类名、方法名）
def get_code_retriever_with_filter(project_id: str, user_id: str, filter_dict: dict = None, k: int = 6):
    vector_store = ensure_code_vector_store(project_id, user_id)
    if vector_store is None:
        return None
    if filter_dict:
        return vector_store.as_retriever(search_kwargs={"k": k, "filter": filter_dict})
    else:
        return vector_store.as_retriever(search_kwargs={"k": k})


def get_project_docs_vector_store(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    user_id = validate_path_component(user_id, "user_id")
    persist_directory = (Path(Config.CHROMA_PROJECT_DOCS_DIR.format(user_id=user_id)) / project_id)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=create_embedding(),
        collection_name=Config.CHROMA_PROJECT_DOCS_COLLECTION_NAME.format(project_id=project_id)
    )


def get_project_docs_retriever(project_id: str, user_id: str, k=6):
    vector_store = ensure_project_docs_vector_store(project_id, user_id)
    if vector_store is None:
        return None
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


# 清空指定Chroma collection，但不删除整个Chroma数据库目录。
# 避免Windows下删除仍被 sqlite 使用的目录导致WinError32。
def clear_vector_collection(persist_directory: Path, collection_name: str):
    vector_store = Chroma(persist_directory=str(persist_directory), embedding_function=create_embedding(),
                          collection_name=collection_name)

    try:
        vector_store._client.delete_collection(name=collection_name)
        print(f"已删除旧向量集合：{collection_name}")
    except Exception as e:
        print(f"删除旧向量集合失败：{collection_name}，原因：{e}")

    return vector_store


def build_global_vector_store(force_rebuild: bool = False):
    vector_store_path = Path(Config.CHROMA_GLOBAL_DOCS_DIR)
    if force_rebuild and vector_store_path.exists():
        shutil.rmtree(vector_store_path)
        print("已删除旧global_docs，正在重建")
    print("加载全局文档")
    documents = load_documents()
    if not documents:
        print("没有全局文档")
        return None

    splits = split_document(documents)

    if not splits:
        print("全局文档切分后没有有效内容")
        return None

    # 为每个 chunk 补充统一的文件名元数据，
    # 方便后续单文件增量更新时删除旧索引
    for doc in splits:
        source = doc.metadata.get("source", "")
        doc.metadata["filename"] = Path(source).name if source else ""

    embedding = create_embedding()

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(vector_store_path),
        collection_name=Config.CHROMA_GLOBAL_DOCS_COLLECTION_NAME
    )
    print(f"global docs写入完成：{len(splits)}chunks")
    return vector_store


# 对单个全局文档进行增量索引。
# 如果该文件已经存在索引，则先删除旧索引，再写入当前文件的新内容。
def index_global_doc(filename: str, documents):
    filename = validate_path_component(filename, "filename")
    vector_store = get_global_vector_store()

    # 删除该文件旧的向量索引
    try:
        vector_store.delete(
            where={"filename": filename}
        )
    except Exception as e:
        print(f"删除旧全局文档索引失败：{filename}，原因：{e}")

    # 文档切分
    splits = split_document(documents)

    if not splits:
        print(f"跳过全局文档索引 {filename}: 文档切分后无有效块")
        return

    # 添加统一元数据
    texts = []
    metadatas = []

    for doc in splits:
        texts.append(doc.page_content)

        metadata = dict(doc.metadata)
        metadata["filename"] = filename

        metadatas.append(metadata)

    vector_store.add_texts(texts=texts, metadatas=metadatas)

    print(
        f"global docs增量索引完成："
        f"{filename}，{len(splits)} chunks"
    )


# 根据当前全局文档目录中的文档，自动重新建立全局文档向量索引。
def rebuild_global_vector_store():
    vector_store_path = Path(Config.CHROMA_GLOBAL_DOCS_DIR)

    # 读取当前全部全局文档
    print("开始恢复全局文档向量索引")
    documents = load_documents()

    if not documents:
        print("没有可建立索引的全局文档")
        return None

    # 文档切分
    splits = split_document(documents)

    if not splits:
        print("全局文档切分后没有有效内容")
        return None

    print(
        f"开始重建全局文档索引："
        f"documents={len(documents)}，chunks={len(splits)}"
    )

    # 为每个 chunk 补充统一的文件名元数据
    # 方便后续单文件增量更新时删除旧索引
    for doc in splits:
        source = doc.metadata.get("source", "")
        doc.metadata["filename"] = Path(source).name if source else ""

    # 建立新的向量数据库
    embedding = create_embedding()

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=str(vector_store_path),
        collection_name=Config.CHROMA_GLOBAL_DOCS_COLLECTION_NAME
    )

    print(
        f"全局文档向量索引恢复完成："
        f"{len(documents)} 个文档，{len(splits)} 个 chunks"
    )

    return vector_store


# 确保全局文档向量数据库存在且有有效数据。
# 如果数据库不存在或为空，则自动恢复。
def ensure_global_vector_store():
    vector_store_path = Path(Config.CHROMA_GLOBAL_DOCS_DIR)

    if not vector_store_path.exists():
        print("全局文档向量索引不存在，开始自动恢复")
        return rebuild_global_vector_store()

    vector_store = get_global_vector_store()

    try:
        count = vector_store._collection.count()
    except Exception as e:
        print(f"检查全局文档向量索引失败：{e}")
        print("开始重新建立全局文档向量索引")
        return rebuild_global_vector_store()

    if count > 0:
        return vector_store

    print("全局文档向量索引为空，开始自动恢复")

    # 不删除Chroma数据库目录，只删除旧 collection
    clear_vector_collection(
        vector_store_path,
        Config.CHROMA_GLOBAL_DOCS_COLLECTION_NAME
    )

    return rebuild_global_vector_store()


def index_project_docs(project_id: str, user_id: str, doc_id: str, documents):
    project_id = validate_path_component(project_id, "project_id")
    user_id = validate_path_component(user_id, "user_id")
    doc_id = validate_path_component(doc_id, "doc_id")
    vector_store = get_project_docs_vector_store(project_id, user_id)
    vector_store.delete(
        where={"$and": [
            {"project_id": project_id},
            {"doc_id": doc_id}
        ]}
    )
    splits = split_document(documents)
    if not splits:
        print(f"跳过文档索引 {doc_id}: 文档切分后无有效块")
        return
    texts = [d.page_content for d in splits]
    metadatas = [{"user_id": user_id, "project_id": project_id, "doc_id": doc_id} for _ in splits]
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"project docs写入完成：{project_id}/{doc_id}，{len(splits)} chunks")


# 根据当前用户/项目docs目录中的最新文档，重新建立项目文档向量索引。
def rebuild_project_docs_vector_store(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    user_id = validate_path_component(user_id, "user_id")

    docs_path = get_project_docs_path(project_id, user_id)

    if not docs_path.exists():
        print(f"项目文档目录不存在：{docs_path}")
        return None

    # 当前项目文档向量库路径
    vector_store_path = (Path(Config.CHROMA_PROJECT_DOCS_DIR.format(user_id=user_id)) / project_id)

    if vector_store_path.exists():
        clear_vector_collection(vector_store_path, Config.CHROMA_PROJECT_DOCS_COLLECTION_NAME.format(project_id=project_id))

    current_docs = []

    # 只处理当前文档，排除历史版本
    for file_path in docs_path.iterdir():
        if not file_path.is_file():
            continue

        filename = file_path.name

        # 历史版本格式：
        # name_YYYYMMDD_HHMMSS_userid.ext
        name = file_path.stem
        parts = name.rsplit("_", 3)

        if len(parts) == 4:
            _, date_part, time_part, history_user_id = parts

            if (
                len(date_part) == 8
                and len(time_part) == 6
                and history_user_id == user_id
                and date_part.isdigit()
                and time_part.isdigit()
            ):
                continue

        current_docs.append(file_path)

    if not current_docs:
        print(f"项目 {project_id} 没有可建立索引的文档")
        return None

    print(
        f"开始重建项目文档索引："
        f"user={user_id}, project={project_id}, "
        f"documents={len(current_docs)}"
    )

    indexed_count = 0
    for file_path in current_docs:
        try:
            documents = load_single_document(file_path)
            if not documents:
                print(f"跳过空文档：{file_path.name}")
                continue
            index_project_docs(project_id, user_id, file_path.name, documents)
            indexed_count += 1

        except Exception as e:
            print(f"文档索引失败：{file_path.name}，原因：{e}")

    if indexed_count == 0:
        print("没有成功建立任何项目文档索引")
        return None

    print(
        f"项目文档索引重建完成："
        f"{project_id}，共 {indexed_count} 个文档"
    )

    return get_project_docs_vector_store(project_id, user_id)


# 确保当前用户/项目存在有效的项目文档向量索引。
# 如果索引不存在或为空，则根据本地docs自动重建。
def ensure_project_docs_vector_store(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    user_id = validate_path_component(user_id, "user_id")

    vector_store_path = (
        Path(Config.CHROMA_PROJECT_DOCS_DIR.format(user_id=user_id))
        / project_id
    )

    if not vector_store_path.exists():
        print(
            f"项目文档向量索引不存在，"
            f"开始自动重建：{user_id}/{project_id}"
        )
        return rebuild_project_docs_vector_store(project_id, user_id)

    vector_store = get_project_docs_vector_store(project_id, user_id)

    try:
        count = vector_store._collection.count()
    except Exception as e:
        print(f"检查项目文档向量索引失败：{e}")
        return rebuild_project_docs_vector_store(project_id, user_id)

    if count > 0:
        return vector_store

    print(
        f"项目文档向量索引为空，"
        f"开始自动重建：{user_id}/{project_id}"
    )

    clear_vector_collection(
        vector_store_path,
        Config.CHROMA_PROJECT_DOCS_COLLECTION_NAME.format(
            project_id=project_id
        )
    )

    return rebuild_project_docs_vector_store(project_id, user_id)


def index_code(project_id: str, filename: str, code: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    filename = validate_path_component(filename, "filename")
    user_id = validate_path_component(user_id, "user_id")
    vector_store = get_code_vector_store(project_id, user_id)
    vector_store.delete(
        where={"$and": [
            {"filename": filename},
            {"project_id": project_id}
        ]}
    )
    chunks_metadata = split_code_with_metadata(code, filename)
    if not chunks_metadata:
        print(f"跳过索引 {filename}: 代码切分后无有效块，不进行 embedding")
        return
    texts = []
    metadatas = []
    for chunk in chunks_metadata:
        texts.append(chunk["text"])
        meta = {
            "filename": filename,
            "project_id": project_id,
            **chunk["metadata"]
        }
        meta["call_sites"] = json.dumps(meta.get("call_sites", []))
        meta["referenced_symbols"] = json.dumps(meta.get("referenced_symbols", []))
        metadatas.append(meta)

    vector_store.add_texts(texts=texts, metadatas=metadatas)
    print(f"code写入完成：{filename}")

    try:
        from shared.advanced_rag import invalidate_bm25_cache
        invalidate_bm25_cache(project_id, user_id)
    except Exception as e:
        print(f"清除 BM25 缓存失败：{e}")


# 根据当前用户/项目 scripts 目录中的最新代码，重新建立整个项目的代码向量索引。
def rebuild_code_vector_store(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    user_id = validate_path_component(user_id, "user_id")
    scripts_path = get_project_scripts_path(project_id, user_id)

    if not scripts_path.exists():
        print(f"项目脚本目录不存在：{scripts_path}")
        return None

    # 当前项目代码向量库路径
    vector_store_path = (Path(Config.CHROMA_CODE_DIR.format(user_id=user_id)) / project_id)

    if vector_store_path.exists():
        clear_vector_collection(vector_store_path, Config.CHROMA_CODE_COLLECTION_NAME.format(project_id=project_id))

    current_scripts = []

    # 只处理最新脚本，排除历史版本
    for file_path in scripts_path.iterdir():
        if not file_path.is_file():
            continue

        filename = file_path.name
        name = file_path.stem

        # 历史版本格式：
        # name_YYYYMMDD_HHMMSS_userid.ext
        parts = name.rsplit("_", 3)

        if len(parts) == 4:
            _, date_part, time_part, history_user_id = parts

            if (
                len(date_part) == 8
                and len(time_part) == 6
                and history_user_id == user_id
                and date_part.isdigit()
                and time_part.isdigit()
            ):
                continue

        current_scripts.append(file_path)

    if not current_scripts:
        # print(f"项目 {project_id} 没有可建立索引的代码文件")
        return None

    print(
        f"开始重建代码向量索引："
        f"user={user_id}, project={project_id}, "
        f"scripts={len(current_scripts)}"
    )

    indexed_count = 0

    for file_path in current_scripts:
        try:
            code = file_path.read_text(encoding="utf-8")

            if not code.strip():
                print(f"跳过空代码文件：{file_path.name}")
                continue

            index_code(project_id, file_path.name, code, user_id)
            indexed_count += 1

        except Exception as e:
            print(f"代码索引失败：{file_path.name}，原因：{e}")

    if indexed_count == 0:
        print("没有成功建立任何代码向量索引")
        return None

    print(
        f"代码向量索引重建完成："
        f"{project_id}，共 {indexed_count} 个代码文件"
    )
    return get_code_vector_store(project_id, user_id)


# 确保当前用户/项目存在有效的代码向量索引。
# 如果索引不存在或为空，则根据本地scripts自动重建。
def ensure_code_vector_store(project_id: str, user_id: str):
    project_id = validate_path_component(project_id, "project_id")
    user_id = validate_path_component(user_id, "user_id")

    vector_store_path = (Path(Config.CHROMA_CODE_DIR.format(user_id=user_id)) / project_id)

    if not vector_store_path.exists():
        print(
            f"代码向量索引不存在，"
            f"开始自动重建：{user_id}/{project_id}"
        )
        return rebuild_code_vector_store(project_id, user_id)

    vector_store = get_code_vector_store(project_id, user_id)

    try:
        count = vector_store._collection.count()
    except Exception as e:
        print(f"检查代码向量索引失败：{e}")
        return rebuild_code_vector_store(project_id, user_id)

    if count > 0:
        return vector_store

    print(
        f"代码向量索引为空，"
        f"开始自动重建：{user_id}/{project_id}"
    )

    clear_vector_collection(vector_store_path, Config.CHROMA_CODE_COLLECTION_NAME.format(project_id=project_id))

    return rebuild_code_vector_store(project_id, user_id)

if __name__ == "__main__":
    print("测试")
    vector_store = build_global_vector_store(force_rebuild=True)
    if vector_store:
        retriever = get_global_retriever(k=4)
        test_docs = retriever.invoke("测试检索")
        print(f"测试检索返回{len(test_docs)}条结果")
        if test_docs:
            print("第一条结果预览：", test_docs[0].page_content[:150])

