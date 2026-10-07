import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from config import Config
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader, CSVLoader, \
    JSONLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(documents_dir: str = None) -> List[Document]:
    if documents_dir is None:
        documents_dir = Config.DOCUMENTS_DIR

    documents_path = Path(documents_dir)
    if not documents_path.exists():
        os.makedirs(documents_path, exist_ok=True)
        print(f"文档目录已创建：{documents_path}")
        return []

    all_docs: List[Document] = []

    # 加载PDF文件
    try:
        pdf_loader = PyPDFDirectoryLoader(str(documents_path))
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f"加载{len(pdf_docs)}个PDF文件")
    except Exception as e:
        print(f"PDF加载失败{e}")

    # 加载txt文件
    try:
        txt_loader = DirectoryLoader(str(documents_path), glob="**/*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        all_docs.extend(txt_docs)
        print(f"加载{len(txt_docs)}个txt文件")
    except Exception as e:
        print(f"txt加载失败{e}")

    # 加载md文件
    try:
        md_loader = DirectoryLoader(str(documents_path), glob="**/*.md", loader_cls=TextLoader)
        md_docs = md_loader.load()
        all_docs.extend(md_docs)
        print(f"加载{len(md_docs)}个md文件")
    except Exception as e:
        print(f"md加载失败{e}")

    # 加载CSV文件
    for file in documents_path.rglob("*.csv"):
        try:
            csv_loader = CSVLoader(str(file))
            csv_docs = csv_loader.load()
            all_docs.extend(csv_docs)
            print(f"加载{len(csv_docs)}个CSV文件")
        except Exception as e:
            print(f"CSV加载失败：{file}，错误{e}")

    # 加载Json文件
    json_docs = []
    for file in documents_path.rglob("*.json"):
        try:
            json_loader = JSONLoader(str(file), jq_schema=".")
            json_docs.extend(json_loader.load())
        except Exception as e:
            print(f"Json加载失败: {file}，错误{e}")
    all_docs.extend(json_docs)
    print(f"加载{len(json_docs)}个Json文件")

    # 加载docx
    docx_docs = []
    for file in documents_path.rglob("*.docx"):
        try:
            docx_loader = UnstructuredWordDocumentLoader(str(file))
            docx_docs.extend(docx_loader.load())
        except Exception as e:
            print(f"docx加载失败：{file}，错误{e}")
    all_docs.extend(docx_docs)
    print(f"加载{len(docx_docs)}个docx文件")
    return all_docs


# 加载单个文档
def load_single_document(file_path: str) -> List[Document]:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文档不存在：{file_path}")

    if not file_path.is_file():
        raise ValueError(f"不是有效文件：{file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFDirectoryLoader(str(file_path.parent))
        documents = [
            doc for doc in loader.load()
            if Path(doc.metadata.get("source", "")).resolve() == file_path.resolve()
        ]

    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

    elif suffix == ".csv":
        loader = CSVLoader(str(file_path))
        documents = loader.load()

    elif suffix == ".json":
        loader = JSONLoader(
            str(file_path),
            jq_schema="."
        )
        documents = loader.load()

    elif suffix == ".docx":
        loader = UnstructuredWordDocumentLoader(str(file_path))
        documents = loader.load()

    else:
        raise ValueError(f"不支持的文档格式：{suffix}")

    print(f"单个文档加载完成：{file_path.name}，共{len(documents)}个文档")
    return documents


def split_document(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "!", "?", ";", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"文档分割完成，共{len(splits)}个文本块")
    return splits


if __name__ == "__main__":
    docs = load_documents()
    print(f"加载到的文档数量：{len(docs)}")
    if docs:
        splits = split_document(docs)
        print(f"分割后的块数量：{len(splits)}")

