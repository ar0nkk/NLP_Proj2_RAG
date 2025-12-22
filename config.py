# modify by yourself
# API配置
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.2-exp"
OPENAI_EMBEDDING_MODEL = "text-embedding-v4"

# 数据目录配置
DATA_DIR = "data/"

# 向量数据库配置
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "try"

# 文本处理配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TOKENS = 1000

# RAG配置
TOP_K = 5
THRESHOLD = 1.2
INTERACTION_MODE = "ui" # 交互方式配置：ui 或 cli