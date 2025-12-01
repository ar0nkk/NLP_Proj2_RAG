# NLP Project 2 - RAG 课程助教系统

基于 RAG (Retrieval-Augmented Generation) 技术的智能课程助教系统，支持多种文档格式的知识检索与问答。

## 项目结构

```
├── config.py           # 配置文件（API密钥、模型参数等）
├── document_loader.py  # 文档加载器（支持PDF/PPTX/DOCX/TXT）
├── text_splitter.py    # 文本切分器
├── vector_store.py     # 向量数据库（基于ChromaDB）
├── rag_agent.py        # RAG Agent核心逻辑
├── process_data.py     # 数据处理入口
├── main.py             # 主程序入口
├── data/               # 课程材料存放目录
└── benchmark_pipline/  # 评测流程
```

## 环境配置

```bash
pip install -r requirements.txt
cd benchmark_pipline
pip install -r requirements.txt
```

## 配置说明

在 `config.py` 中配置以下参数：

```python
OPENAI_API_KEY = "your-api-key"
OPENAI_API_BASE = "https://your-api-base/v1"
MODEL_NAME = "your-model-name"
OPENAI_EMBEDDING_MODEL = "your-embedding-model"
```

## 已实现功能

### 1. 文档加载 (`document_loader.py`)

- `load_pdf`: 使用 PyPDF2 按页提取 PDF 文本
- `load_pptx`: 使用 python-pptx 按幻灯片提取 PPT 文本
- `load_docx`: 使用 docx2txt 提取 Word 文档文本
- `load_txt`: 读取纯文本文件

### 2. 向量数据库 (`vector_store.py`)

- `get_embedding`: 调用 OpenAI API 获取文本向量
- `add_documents`: 将文档块存入 ChromaDB
- `search`: 向量相似度检索 Top-K 文档

### 3. RAG Agent (`rag_agent.py`)

- `system_prompt`: 定义课程助教角色与回答规范
- `retrieve_context`: 检索相关上下文并格式化来源信息
- `generate_response`: 构建提示词并调用 LLM 生成回答

## 使用方法

1. 将课程材料放入 `data/` 目录
2. 运行数据处理脚本构建向量库
3. 启动问答系统

```bash
python main.py
```

## Benchmark 评测

```bash
cd benchmark_pipline
python test_pipeline.py --config config.yaml
```
