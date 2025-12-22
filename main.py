import argparse
import os

from config import VECTOR_DB_PATH, MODEL_NAME
from exercise_agent import ExerciseAgent
from rag_agent import RAGAgent
from rag_ui import launch_gradio_ui, run_cli_session


def main():
    parser = argparse.ArgumentParser(description="课程助教RAG系统")
    parser.add_argument(
        "--mode",
        choices=["ui", "cli"],
        default="ui",
        help="选择交互方式：ui(默认) 或 cli",
    )
    args = parser.parse_args()

    if not os.path.exists(VECTOR_DB_PATH):
        print("未检测到向量数据库，请先构建知识库后再运行。")
        return

    agent = ExerciseAgent(model=MODEL_NAME)

    if args.mode == "ui":
        launch_gradio_ui(agent) # 启动Gradio UI界面
    else:
        run_cli_session(agent) # 运行命令行对话会话


if __name__ == "__main__":
    main()
