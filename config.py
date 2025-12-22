import os
from dotenv import load_dotenv
load_dotenv()

# modify by yourself
# API配置
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
TOP_K = 5 # 调小 TOP_K 表示过滤更严格，可以节约计算资源，减少幻觉风险，但可能遗漏有用信息
THRESHOLD = 1.2 # Chroma 距离阈值，范围[0, 2]，越小表示相关性越高，距离大于 THRESHOLD 的内容会被过滤掉
INTERACTION_MODE = "ui" # 交互方式配置：ui 或 cli