import re
from typing import Any, Dict, List, Optional

from rag_agent import RAGAgent
from config import MODEL_NAME, TOP_K


class ExerciseAgent(RAGAgent):
    """扩展型代理：在基础问答能力之上支持定制练习题"""

    def __init__(self, model: str = MODEL_NAME):
        super().__init__(model=model)

        self.exercise_prompt = """你是一位专业课程助教，擅长基于课程材料设计分层练习题。

    生成习题时需要遵循以下要求：
    1. 优先满足学生对题目数量、难度与题型的要求；若未指定，默认提供3道题，分别为基础/提高/拓展难度，且题型覆盖选择题、判断题、简答题。
    2. 每道题必须包含：题干、难度标签、题型说明、标准答案、解析。
    3. 解析中需要依据课程材料给出出处，沿用格式：[来源：文件名, 第X页]。
    4. 如果缺少必要资料，请说明该题无法生成并给出补充建议。
    5. 输出保持清晰编号，并在题目之间明确区分难度与题型。
    """

        # 识别生成习题请求的关键词
        self.exercise_triggers = (
            "生成习题",
            "生成练习",
            "生成练习题",
            "生成题目",
            "自动出题",
            "出几道题",
            "出一些题",
            "练习题",
            "练习题目",
            "设计习题",
            "设计练习",
            "练习任务",
            "practice problem",
            "practice problems",
            "practice question",
            "practice questions",
            "generate exercise",
            "generate exercises",
            "create exercise",
            "create exercises",
            "quiz",
        )

        # 题型关键词映射
        self.question_type_keywords = {
            "选择题": ("选择题", "选择", "单选", "多选", "multiple choice", "mcq"),
            "判断题": ("判断题", "判断", "true/false", "true or false", "对错"),
            "简答题": ("简答题", "简答", "short answer", "问答", "解释"),
        }

        # 难度关键词映射
        self.difficulty_keywords = {
            "基础": ("基础", "简单", "入门", "basic", "beginner", "easy"),
            "提高": ("提高", "进阶", "中等", "medium", "intermediate"),
            "拓展": ("拓展", "高级", "困难", "advanced", "hard", "challenge"),
        }

    # 识别是否为习题生成请求
    def _is_exercise_request(self, query: str) -> bool:
        if not query:
            return False

        query_stripped = query.strip()
        q_lower = query_stripped.lower()

        for phrase in self.exercise_triggers:
            if phrase in query_stripped or phrase in q_lower:
                return True

        chinese_generators = ("生成", "设计", "出", "给", "安排")
        chinese_targets = ("习题", "练习", "练习题", "题目")

        if any(t in query_stripped for t in chinese_targets) and any(
            g in query_stripped for g in chinese_generators
        ):
            return True

        return False

    # 解析习题生成偏好：类型/难度/数量
    def _parse_exercise_preferences(self, query: str) -> Dict[str, Any]:
        preferences = {
            "num_questions": None,
            "difficulties": [],
            "types": [],
        }

        if not query:
            return preferences

        text = query.strip()
        text_lower = text.lower()

        count_match = re.search(
            r"(\d+)\s*(?:道|题|份|套)?\s*(?:题|练习|试题|question|questions)?",
            text_lower,
        )
        if count_match:
            preferences["num_questions"] = max(1, min(10, int(count_match.group(1))))

        for label, keywords in self.difficulty_keywords.items():
            if any(kw in text or kw in text_lower for kw in keywords):
                preferences["difficulties"].append(label)

        for label, keywords in self.question_type_keywords.items():
            if any(kw in text or kw in text_lower for kw in keywords):
                preferences["types"].append(label)

        preferences["difficulties"] = list(dict.fromkeys(preferences["difficulties"]))
        preferences["types"] = list(dict.fromkeys(preferences["types"]))

        return preferences

    # 生成习题
    def generate_exercise_response(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> str:
        messages = [{"role": "system", "content": self.exercise_prompt}]

        if chat_history:
            messages.extend(chat_history)

        context_block = (
            context.strip()
            if context.strip()
            else "（当前没有可用的课程材料，请基于已有知识给出练习题并标明缺失）"
        )

        preferences = self._parse_exercise_preferences(query)
        target_count = preferences["num_questions"] or 3 # 默认3道题

        preference_lines = [
            f"- 题目数量：{target_count} 道（{'学生指定' if preferences['num_questions'] else '默认值'}）",
        ]

        if preferences["difficulties"]:
            difficulty_desc = "、".join(preferences["difficulties"])
            preference_lines.append(
                f"- 难度要求：{difficulty_desc}（如资料不足需说明）"
            )
        else:
            preference_lines.append(
                "- 难度要求：默认覆盖基础/提高/拓展三种层级，保持梯度递进。"
            )

        if preferences["types"]:
            type_desc = "、".join(preferences["types"])
            preference_lines.append(
                f"- 题型要求：{type_desc}（如无法满足需解释）"
            )
        else:
            preference_lines.append(
                "- 题型要求：默认至少包含选择题、判断题、简答题，并尽量做到题型不重复。"
            )

        preference_block = "\n".join(preference_lines)
        user_text = f"""以下是与题目设计相关的课程材料：

{context_block}

---

学生需求：{query}

请基于材料内容生成{target_count}道练习题，并遵守以下偏好：
{preference_block}

每道题必须包含：
1. 题干（清晰陈述，必要时给出上下文）
2. 难度标签（基础/提高/拓展，或学生指定的其他等级）
3. 题型说明（如：选择题/判断题/简答题）
4. 标准答案
5. 解析：说明解题思路，并引用材料来源；若缺少依据，说明原因

优先围绕学生的具体要求设置题目主题与题型搭配。"""

        messages.append({"role": "user", "content": user_text})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=1800,
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"生成习题时出错: {str(e)}"

    def answer_question(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: int = TOP_K,
        return_details: bool = False,
    ) -> Any:
        # 主接口
        context, retrieved_docs = self.retrieve_context(query, top_k=top_k)

        if not context:
            context = "（未检索到特别相关的课程材料）"

        is_exercise = self._is_exercise_request(query)

        if is_exercise:
            answer = self.generate_exercise_response(query, context, chat_history)
        else:
            answer = self.generate_response(query, context, chat_history)

        if return_details:
            return {
                "answer": answer,
                "context": context,
                "retrieved_docs": retrieved_docs,
                "is_exercise": is_exercise,
            }

        return answer
