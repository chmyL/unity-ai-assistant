from __future__ import annotations
from shared.llm_factory import create_llm
from shared.prompts import SCENE_PROMPT
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# 定义可用的场景类型
SCENE_TYPES = ["code_gen", "update", "doc_summary", "debug", "hybrid"]
# 问题示例，对比相似度
SCENE_PROTOTYPES: Dict[str, List[str]] = {
    "code_gen": [
        "用户要求生成新的代码",
        "用户要求从零开始编写代码",
        "用户要求实现一个新的功能",
        "用户要求创建一个程序",
        "用户要求写一个新的函数",
        "用户要求开发一个新的接口",
        "用户要求生成代码示例",
        "用户希望根据需求编写代码",
    ],
    "update": [
        "用户要求修改已有代码",
        "用户要求更新现有代码",
        "用户要求调整现有代码",
        "用户要求重构已有代码",
        "用户要求在原有代码基础上增加功能",
        "用户要求修改一个已经存在的函数",
        "用户要求优化现有实现",
        "用户提供代码并要求进行修改",
    ],
    "doc_summary": [
        "用户要求总结文档",
        "用户要求概括文档内容",
        "用户要求总结这份资料",
        "用户要求提取文档重点",
        "用户要求归纳文档内容",
        "用户要求总结一个文件",
        "用户要求阅读并总结文档",
        "用户要求解释文档主要内容",
    ],
    "debug": [
        "用户遇到了代码报错问题",
        "用户要求分析程序错误",
        "用户要求解决代码中的 bug",
        "用户要求排查异常",
        "用户要求分析 traceback",
        "用户的程序无法正常运行",
        "用户要求定位代码中的问题",
        "用户提供错误信息并要求解决",
    ],
    "hybrid": [
        "用户同时要求分析和修改代码",
        "用户同时要求解释问题并实现修改",
        "用户的请求包含多个不同类型的任务",
        "用户要求先分析再修改代码",
        "用户要求阅读代码并进行修改",
        "用户要求总结内容然后生成代码",
        "用户同时涉及代码生成和问题排查",
        "用户的请求同时包含多个任务",
    ],
}
# 规则
RULES = {
    "debug": ["报错", "错误", "异常", "bug", "bugs", "traceback", "exception", "error", "报错信息", "错误信息", "无法运行",
              "运行失败", "运行不了", "程序崩溃", "崩溃了", "哪里有问题", "出了什么问题", "排查问题", "定位问题", "debug", "调试",
              "空引用", "空指针", "null", "NullReference",],
    "doc_summary": ["总结", "摘要", "概括", "归纳", "总结一下", "概括一下", "提炼重点", "提取重点", "总结文档", "总结文件", "文档摘要",],
    "update": ["修改", "改一下", "帮我改", "修改一下", "更新", "调整", "重构", "优化", "改成", "替换", "删除", "增加", "添加", "在现有代码",
               "在原代码", "基于现有代码", "基于这段代码", "基于这个代码",],
    "code_gen": ["写一个", "写一段", "实现一个", "实现一下", "生成代码", "帮我写代码", "帮我实现", "编写代码", "创建一个", "开发一个",
                 "生成一个",],
}


# 模型参数设置
@dataclass
class SceneClassifierConfig:
    # 本地模型
    model_name: str = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    # 相似度低于这个值时，认为分类不可靠
    confidence_threshold: float = 0.45
    # 是否优先使用 CPU
    device: Optional[str] = None
    # encode batch size
    batch_size: int = 32


# 规则筛选分类
# 规则筛选分类
def detect_by_rules(query: str):
    if not query:
        return None

    q = query.lower().strip()
    matched_scenes = []

    for scene, keywords in RULES.items():
        for keyword in keywords:
            if keyword in q:
                matched_scenes.append(scene)
                break

    if not matched_scenes:
        return None

    # 同时包含“分析/理解”和“修改/实现”等操作时，判定为 hybrid
    if "update" in matched_scenes and (
        "分析" in q
        or "检查" in q
        or "理解" in q
        or "研究" in q
    ):
        return "hybrid"

    if len(matched_scenes) == 1:
        return matched_scenes[0]

    return "hybrid"


class LocalSceneClassifier:
    def __init__(self, config: Optional[SceneClassifierConfig] = None):
        self.config = config or SceneClassifierConfig
        print(f"[SceneClassifier] Loading model:"
              f"{self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        self.prototype_embeddings = self._build_prototype_embeddings()
        print("[SceneClassifier] Model loaded")

    def _build_prototype_embeddings(self)->Dict[str, np.ndarray]:
        embeddings = {}
        for scene, prototypes in SCENE_PROTOTYPES.items():
            vectors = self.model.encode(
                prototypes,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            vectors = np.asarray(vectors)
            embeddings[scene] = vectors
        return embeddings

    def _encode_query(self,query:str):
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding)

    @staticmethod
    def _cosine_similarity(query_embedding:np.ndarray, prototype_embeddings:np.ndarray):
        return np.dot(prototype_embeddings, query_embedding)

    def classfiy_with_scores(self, query:str):
        if not query or not query.strip():
            return "hybrid", 0.0
        query_embedding = self._encode_query(query)
        scene_scores = {}
        # 取每个场景相似度最高值
        for scene, prototype_embedding in self.prototype_embeddings.items():
            similarities = self._cosine_similarity(query_embedding, prototype_embedding)
            score = float(np.max(similarities))
            scene_scores[scene] = score
        ranked = sorted(scene_scores.items(), key=lambda x:x[1], reverse=True)
        best_scene, best_score = ranked[0]

        if best_score < self.config.confidence_threshold:
            return "hybrid", best_score

        return best_scene, best_score


_classifier: Optional[LocalSceneClassifier] = None


def get_scene_classifier()->LocalSceneClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LocalSceneClassifier()
    return _classifier


# 判断场景的总流程函数
def detect_scene_type(query:str):
    scene = detect_by_rules(query)
    if scene is not None:
        return scene
    classifier = get_scene_classifier()
    scene, confidence = classifier.classfiy_with_scores(query)
    return scene


if __name__ == "__main__":

    test_queries = [
        "帮我写一个 FastAPI 登录接口",
        "把这个函数改成异步版本",
        "为什么这里会出现 KeyError？",
        "帮我总结一下这个 README",
        "分析这个项目，然后把数据库层改成 PostgreSQL",
        "帮我实现一个 Redis 缓存",
        "这段代码运行时报错了",
        "请阅读这个文档并提取核心内容",
    ]

    for query in test_queries:

        result = detect_scene_type(query)

        print("\n" + "=" * 70)
        print("Query:", query)
        print("Scene:", result)

# # 判断用户query对应的场景类型
# def detect_scene_type_llm(query: str):
#     llm = create_llm(temperature=0.0)  # 意图分析可以用确定性输出
#     prompt = SCENE_PROMPT.format(query=query)
#
#     res = llm.invoke(prompt)
#     raw = res.content.strip() if hasattr(res, "content") else str(res).strip()
#
#     # 提取第一个合法场景词
#     for word in raw.split():
#         if word in SCENE_TYPES:
#             return word
#     return "hybrid"
