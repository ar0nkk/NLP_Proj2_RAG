from typing import Dict, List


# 格式化参考资料为Markdown
def _format_reference_markdown(retrieved_docs: List[Dict], limit: int = 10) -> str:
    if not retrieved_docs:
        return ""

    blocks = []
    for idx, doc in enumerate(retrieved_docs[:limit], start=1):
        metadata = doc.get("metadata", {})
        filename = metadata.get("filename", "未知文件")
        page_number = metadata.get("page_number")
        if page_number:
            header = f"{idx}. {filename}（第{page_number}页）"
        else:
            header = f"{idx}. {filename}"

        snippet = doc.get("content", "").strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:180].rstrip() + "..."

        blocks.append(f"{header}\n> {snippet}" if snippet else header)

    return "\n\n".join(blocks)

# 将聊天历史从列表转换为结构化格式
def _convert_history(history: List) -> List[Dict[str, str]]:
    structured = []
    for turn in history:
        if isinstance(turn, (list, tuple)):
            user_msg = turn[0] if len(turn) > 0 else ""
            assistant_msg = turn[1] if len(turn) > 1 else ""
        elif isinstance(turn, dict):
            # Chatbot 可能以{'role': 'user', 'content': '...'}形式传入
            user_msg = turn.get("content") if turn.get("role") == "user" else ""
            assistant_msg = turn.get("content") if turn.get("role") == "assistant" else ""
        else:
            user_msg = assistant_msg = ""

        if user_msg:
            structured.append({"role": "user", "content": user_msg})
        if assistant_msg:
            structured.append({"role": "assistant", "content": assistant_msg})
    return structured

# 运行命令行对话会话
def run_cli_session(agent) -> None:
    print("=" * 60)
    print("欢迎使用RAG智能课程助教系统！")
    print("提示：输入诸如“出3道中等难度选择题巩固HMM”自动生成习题。")
    print("=" * 60)

    chat_history: List[Dict[str, str]] = []

    while True:
        try:
            query = input("\n学生: ").strip()

            if not query:
                continue

            answer_payload = agent.answer_question(query, chat_history=chat_history)

            answer_text = (
                answer_payload.get("answer")
                if isinstance(answer_payload, dict)
                else answer_payload
            )

            print(f"\n助教: {answer_text}")

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer_text})

        except KeyboardInterrupt:
            print("\n结束对话。")
            break
        except Exception as exc:
            print(f"\n错误: {exc}")

# 启动Gradio UI界面
def launch_gradio_ui(agent, share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "未检测到gradio，请先运行 `pip install gradio` 再开启UI界面"
        ) from exc

    def respond(message: str, history: List[List[str]]):
        chat_history = _convert_history(history)
        result = agent.answer_question(
            message,
            chat_history=chat_history,
            return_details=True,
        )

        if isinstance(result, str):
            return result

        references = _format_reference_markdown(result.get("retrieved_docs", []))
        answer_text = result.get("answer", "")

        if references:
            answer_text = f"{answer_text}\n\n---\n**参考资料**\n{references}"

        return answer_text

    description = "输入课程问题或练习需求，系统会检索资料并给出引用明确的回答。"

    CUSTOM_CSS = """
    html, body {height: 100%; margin: 0; overflow: hidden !important;}
    #root, .gradio-container, .gradio-container > div, .gradio-page {height: 100%; overflow: hidden !important;}
    .gradio-container {position: fixed; inset: 0;}
    .gradio-chatbot {max-height: calc(100vh - 160px);}
    """

    MATHJAX_JS = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

    with gr.Blocks(css=CUSTOM_CSS, js=MATHJAX_JS) as demo:
        gr.ChatInterface(
            fn=respond,
            title="课程助教 RAG 助手",
            description=description,
            examples=[
                "总结一下隐马尔可夫模型（HMM）的核心概念。",
                "出题可指定题目数量、难度、类型，例如：生成3道中等的HMM的选择题。",
            ],
            chatbot=gr.Chatbot(
                height=620,
                render_markdown=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "\\[", "right": "\\]", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\(", "right": "\\)", "display": False},
                ],
            ),
            submit_btn="发送",
            stop_btn="停止",
        )

    demo.queue()
    demo.launch(share=share, inbrowser=True)
