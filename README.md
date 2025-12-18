# NLP Project 2 - RAG 课程助教系统

基于 RAG (Retrieval-Augmented Generation) 技术的智能课程助教系统，支持多种文档格式的知识检索与问答。
- [作业文档](https://gpy5q03kes.feishu.cn/wiki/JgqTwaqG2ih6hdkWk6pcYBhAnPd)
- [项目地址（带 benchmark）](https://github.com/HEHUA2005/SJTU-NLP-project2-benchmark/tree/main)

## TODO
- 修改完善原有代码，包括：
    - 定义助教的角色人设与回答规范
    - ……
- 完成附加功能，具体参考[作业文档](https://gpy5q03kes.feishu.cn/wiki/JgqTwaqG2ih6hdkWk6pcYBhAnPd)
- Benchmark 评测
- 制作报告 ppt
- 重写 `README.md`，包含对提交文件的必要解释

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
conda create -n nlp python=3.12 -y
conda activate nlp
pip install -r requirements.txt
cd benchmark_pipline
pip install -r requirements.txt
```

## 数据下载
由于网络原因，建议从 Hugging Face **手动下载**：
- [QA 数据集](https://huggingface.co/datasets/HEHUA2005/rag-benchmark-qa-dataset)
- [PDF 数据](https://huggingface.co/datasets/HEHUA2005/rag-benchmark-pdf-data)

下载成功后的数据格式如下所示：
```
QA_data/
├── README.md
└── data/
    ├── An_Introduction_to_Xi_Jinping_Thought_on_Socialism_with_Chinese_Characteristics_for_a_New_Era-00000-of-00001.parquet
    ├── Ideological_Morality_and_Legal_System-00000-of-00001.parquet
    ├── Mao_Zedong_Thought-00000-of-00001.parquet
    ├── Outline_of_Modern_and_Contemporary_Chinese_History-00000-of-00001.parquet
    └── Principles_of_Marxism-00000-of-00001.parquet

data/
├── README.md
├── An-Introduction-to-Xi-Jinping-Thought-on-Socialism-with-Chinese-Characteristics-for-a-New-Era/
│   └── *.pdf
├── Ideological-Morality-and-Legal-System/
│   └── *.pdf
├── Mao-Zedong-Thought/
│   └── *.pdf
├── Outline-of-Modern-and-Contemporary-Chinese-History/
│   └── *.pdf
└── Principles-of-Marxism/
    └── *.pdf
```
为了顺利运行 `test_pipeline.py`，我已注释掉代码第49行开始的“下载数据集”这一部分，改成了“直接读取本地文件”，如仍需使用脚本下载，需要自行修改为原始代码。

使用脚本的下载方法：
```
# 下载所有数据（PDF + QA 数据集）
python download_data.py

# 或只下载 QA 数据集
python download_data.py --download qa

# 或只下载 PDF 文档
python download_data.py --download pdf
```
如果因为网络原因下载失败，尝试添加镜像：
```
export HF_ENDPOINT=https://hf-mirror.com
```
> 由于 QA 数据集是 parquet 格式，我修改了`benchmark_pipline/run_benchmark.py` 中读取 QA 数据集的部分

## 配置说明
`config.py` 参数配置为：

```python
OPENAI_API_KEY = "your-api-key"
OPENAI_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.2-exp"
OPENAI_EMBEDDING_MODEL = "text-embedding-v4"
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
2. 运行数据处理脚本 `process_data.py` 构建向量库
3. 启动问答系统

```bash
python main.py
```

## Benchmark 评测

```bash
cd benchmark_pipline
python test_pipeline.py --config config.yaml
```
