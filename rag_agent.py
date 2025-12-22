from typing import List, Dict, Optional, Tuple, Any

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    MODEL_NAME,
    TOP_K,
)
from vector_store import VectorStore


class RAGAgent:
    def __init__(
        self,
        model: str = MODEL_NAME,
    ):
        self.model = model

        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

        self.vector_store = VectorStore()

        """
        TODO: 实现并调整系统提示词，使其符合课程助教的角色和回答策略
        """
        self.system_prompt = """你是一位专业、耐心的课程助教，负责回答学生关于课程内容的问题。

    你的职责：
    1. 基于提供的课程材料，准确、清晰地回答学生问题
    2. 如果问题涉及课程材料中的内容，请引用具体来源（文件名和页码）
    3. 如果课程材料中没有相关内容，请明确说明并尝试提供一般性指导
    4. 使用友好、专业的语气，鼓励学生深入思考
    5. 对于复杂概念，优先用简洁易懂的方式解释，并给出关键步骤

    回答规范：
    - 回答要有条理，必要时使用编号或分点
    - 引用来源时使用格式：[来源：文件名, 第X页]，若无页码则写为[来源：文件名]
    - 如果学生的问题不清楚，可以请求澄清
    - 严禁编造资料；当上下文不足时要说明"""

    def retrieve_context(
        self, query: str, top_k: int = TOP_K
    ) -> Tuple[str, List[Dict]]:
        """检索相关上下文
        TODO: 实现检索相关上下文
        要求：
        1. 使用向量数据库检索相关文档
        2. 格式化检索结果，构建上下文字符串
        3. 每个检索结果需要包含来源信息（文件名和页码）
        4. 返回格式化的上下文字符串和原始检索结果列表
        """
        # 使用向量数据库检索相关文档
        retrieved_docs = self.vector_store.search(query, top_k=top_k)

        if not retrieved_docs:
            placeholder = "（未检索到与该问题直接相关的课程材料）"
            return placeholder, []
        
        # 格式化检索结果，构建上下文字符串
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "").strip() # 使用strip()去除多余空白
            if not content:
                continue
            metadata = doc.get("metadata", {})
            filename = metadata.get("filename", "未知文件")
            page_number = metadata.get("page_number", 0)
            
            # 构建带来源信息的上下文片段
            if page_number > 0:
                source_info = f"[来源：{filename}, 第{page_number}页]"
            else:
                source_info = f"[来源：{filename}]"
            
            context_parts.append(f"--- 参考资料 {i} {source_info} ---\n{content}\n")
        
        context_str = "\n".join(context_parts)
        return context_str, retrieved_docs

    def generate_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        """生成回答

        参数:
            query: 用户问题
            context: 检索到的上下文
            chat_history: 对话历史
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if chat_history:
            messages.extend(chat_history)

        """
        TODO: 实现用户提示词
        要求：
        1. 包含相关的课程内容
        2. 包含学生问题
        3. 包含来源信息（文件名和页码）
        4. 返回用户提示词
        """
        context_block = context.strip() if context.strip() else "（当前没有可用的课程材料，请基于已有知识回答并标明缺失）"
        user_text = f"""以下是与问题相关的课程材料：

{context_block}

---

学生问题：{query}

请根据上述课程材料回答学生的问题。如果引用了材料内容，请注明来源，并在来源缺失时说明理由。
如果课程材料无法回答问题，请说明并根据你本身的知识进行一般性回答。
"""

        messages.append({"role": "user", "content": user_text})

        # 多模态接口示意（如需添加图片支持，可参考以下格式）：
        # content_parts = [{"type": "text", "text": user_text}]
        # content_parts.append({
        #     "type": "image_url",
        #     "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        # })
        # messages.append({"role": "user", "content": content_parts})

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7, max_tokens=1500
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成回答时出错: {str(e)}"


    def answer_question(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: int = TOP_K,
        return_details: bool = False,
    ) -> Any:
        """回答问题

        参数:
            query: 用户问题
            chat_history: 对话历史
            top_k: 检索文档数量

        返回:
            生成的回答
        """
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)

        if not context:
            context = "（未检索到特别相关的课程材料）"

        answer = self.generate_response(query, context, chat_history)

        if return_details:
            return {
                "answer": answer,
                "context": context,
                "retrieved_docs": retrieved_docs,
                "is_exercise": False,
            }

        return answer
