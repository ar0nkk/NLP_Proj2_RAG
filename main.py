import os

from config import VECTOR_DB_PATH, MODEL_NAME, INTERACTION_MODE
from exercise_agent import ExerciseAgent
from rag_ui import launch_gradio_ui, run_cli_session


def main():
    if not os.path.exists(VECTOR_DB_PATH):
        print("未检测到向量数据库，请先构建知识库后再运行。")
        return

    agent = ExerciseAgent(model=MODEL_NAME)

    mode = INTERACTION_MODE.lower()
    if mode not in {"ui", "cli"}: # 防止配置错误
        print("配置的交互方式无效，已使用默认 ui。")
        mode = "ui"

    if mode == "ui":
        launch_gradio_ui(agent) # 启动Gradio UI界面
    else:
        run_cli_session(agent) # 运行命令行对话会话


if __name__ == "__main__":
    main()
