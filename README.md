# NLP Project 2 - RAG 课程助教系统

基于 RAG (Retrieval-Augmented Generation) 技术的智能课程助教系统，支持多种文档格式。
你可以调用你想要的模型，并添加你的课程资料，模型会根据文档完成知识检索与问答
- [作业文档](https://gpy5q03kes.feishu.cn/wiki/JgqTwaqG2ih6hdkWk6pcYBhAnPd)

## 主要项目结构
```
NLP_Proj2_RAG/
├── README.md              # 项目说明文档
├── config.py              # 配置文件（API密钥、模型参数等）
├── document_loader.py     # 文档加载器（支持PDF/PPTX/DOCX/TXT）
├── text_splitter.py       # 文本切分器
├── vector_store.py        # 向量数据库（ChromaDB）
├── rag_agent.py           # RAG Agent核心逻辑
├── rag_ui.py              # RAG Gradio界面
├── exercise_agent.py      # 习题生成扩展Agent
├── process_data.py        # 数据处理入口（文档加载、切分、入库）
├── main.py                # 主程序入口
├── requirements.txt       # 依赖包列表
├── data/                  # 课程材料存放目录（已被 .gitignore 忽略）
│   └── .gitkeep           # 占位文件，保证空文件夹被git跟踪
├── vector_db/             # 向量数据库目录（自动生成，已被 .gitignore 忽略）
├── benchmark_data/        # 评测用数据（可选）
├── converted_data/        # 数据转换缓存（可选）
└── ...  # 其他辅助脚本/文件
```

- `data/` 目录仅用于本地存放原始课程资料，**不会被上传到Git仓库**。如需保留空文件夹，请勿删除 `.gitkeep` 文件。
- `vector_db/` 为自动生成的向量数据库目录，**无需手动管理**。

## 使用方法
### 1. 配置环境
```bash
git clone https://github.com/ar0nkk/NLP_Proj2_RAG.git
cd NLP_Proj2_RAG
conda create -n nlp python=3.12 -y
conda activate nlp
pip install -r requirements.txt
```

### 2. 添加资料
将课程文件放入 `data/` 文件夹中（目前支持的格式： `".pdf", ".pptx", ".docx", ".txt"`）。你也可以从 Hugging Face 下载样例数据：[五大红课 pdf](https://huggingface.co/datasets/HEHUA2005/rag-benchmark-pdf-data)

### 3. 处理数据
运行 `process_data.py`，时间取决于数据量

### 4. 修改配置文件
按需修改，`config.py` 中附有详细说明

### 5. 开始使用
运行 `main.py` 启动问答系统，现在你可以愉快使用 RAG 课程助教系统了！

## 基础功能
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

## 优化功能
### 1. 习题生成扩展 Agent (`exercise_agent.py`)
- `generate_mcq`: 基于课程材料生成选择题
- `generate_short_answer`: 基于课程材料生成简答题

### 2. UI 界面 (`rag_ui.py`)
- 交互式问答界面
- 支持 LaTeX 数学公式渲染

### 3. 相关度过滤
模型并不总是输出 K 项参考资料，相关度过低的资料不会被写入上下文。这样做可以：
- **节约 Token 消耗**。
- **减少幻觉风险**。如果无关文档里恰好有一些似是而非的词，大模型可能会强行解释，产生误导性的回答
- **避免污染上下文**。无关信息占据了宝贵的上下文窗口，可能挤掉真正有用的信息（比如对话历史）

### 4. 其他优化
- 使用 `pdfplumber` 替代 `PyPDF2` 进行高质量的 PDF 文本提取，保留文本之间相对结构，使文本顺序更接近“人看到的”，提升模型对 PDF 内容的理解程度

## TODO
- 多模态支持
- 由于上下文窗口有限，模型除了预训练和阅读目录，无法对大量材料形成深入的整体认知。可能的解决方案：
    - 递归摘要，逐步压缩信息
    - 使用支持更大上下文窗口的模型