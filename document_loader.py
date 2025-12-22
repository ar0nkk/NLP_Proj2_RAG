import os
from typing import List, Dict, Optional

import docx2txt
import pdfplumber
from pptx import Presentation

from config import DATA_DIR


class DocumentLoader:
    def __init__(
        self,
        data_dir: str = DATA_DIR,
    ):
        self.data_dir = data_dir
        self.supported_formats = [".pdf", ".pptx", ".docx", ".txt"]

    def load_pdf(self, file_path: str) -> List[Dict]:
        """加载PDF文件，按页返回内容

        TODO: 实现PDF文件加载
        要求：
        1. 使用PdfReader读取PDF文件
        2. 遍历每一页，提取文本内容
        3. 格式化为"--- 第 X 页 ---\n文本内容\n"
        4. 返回pdf内容列表，每个元素包含 {"text": "..."}
        """
        # [AI] 查询 PdfReader 文本提取方法函数 extract_text
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, 1): # start from page 1
                text = page.extract_text() or "" # handle None case
                formatted_text = f"--- 第 {page_idx} 页 ---\n{text}\n" # for identification
                pages.append({"text": formatted_text})
        return pages

    def load_pptx(self, file_path: str) -> List[Dict]:
        """加载PPT文件，按幻灯片返回内容

        TODO: 实现PPT文件加载
        要求：
        1. 使用Presentation读取PPT文件
        2. 遍历每一页，提取文本内容
        3. 格式化为"--- 幻灯片 X ---\n文本内容\n"
        4. 返回幻灯片内容列表，每个元素包含 {"text": "..."}
        """
        # [AI] 查询 Presentation 文本提取方法
        slides = []
        prs = Presentation(file_path)
        for slide_idx, slide in enumerate(prs.slides, 1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs: # type: ignore
                        for run in paragraph.runs:
                            texts.append(run.text)
            slide_text = "\n".join(texts)
            formatted_text = f"--- 幻灯片 {slide_idx} ---\n{slide_text}\n"
            slides.append({"text": formatted_text})
        return slides

    def load_docx(self, file_path: str) -> str:
        """加载DOCX文件
        TODO: 实现DOCX文件加载
        要求：
        1. 使用docx2txt读取DOCX文件
        2. 返回文本内容
        """
        return docx2txt.process(file_path)

    def load_txt(self, file_path: str) -> str:
        """加载TXT文件
        TODO: 实现TXT文件加载
        要求：
        1. 使用open读取TXT文件（注意使用encoding="utf-8"）
        2. 返回文本内容
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def load_document(self, file_path: str) -> List[Dict[str, str]]:
        """加载单个文档，PDF和PPT按页/幻灯片分割，返回文档块列表"""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        documents = []

        if ext == ".pdf":
            pages = self.load_pdf(file_path)
            for page_idx, page_data in enumerate(pages, 1):
                documents.append(
                    {
                        "content": page_data["text"],
                        "filename": filename,
                        "filepath": file_path,
                        "filetype": ext,
                        "page_number": page_idx,
                    }
                )
        elif ext == ".pptx":
            slides = self.load_pptx(file_path)
            for slide_idx, slide_data in enumerate(slides, 1):
                documents.append(
                    {
                        "content": slide_data["text"],
                        "filename": filename,
                        "filepath": file_path,
                        "filetype": ext,
                        "page_number": slide_idx,
                    }
                )
        elif ext == ".docx":
            content = self.load_docx(file_path)
            if content:
                documents.append(
                    {
                        "content": content,
                        "filename": filename,
                        "filepath": file_path,
                        "filetype": ext,
                        "page_number": 0,
                    }
                )
        elif ext == ".txt":
            content = self.load_txt(file_path)
            if content:
                documents.append(
                    {
                        "content": content,
                        "filename": filename,
                        "filepath": file_path,
                        "filetype": ext,
                        "page_number": 0,
                    }
                )
        else:
            print(f"不支持的文件格式: {ext}")

        return documents

    def load_all_documents(self) -> List[Dict[str, str]]:
        """加载数据目录下的所有文档"""
        if not os.path.exists(self.data_dir):
            print(f"数据目录不存在: {self.data_dir}")
            return None # type: ignore

        documents = []

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.supported_formats:
                    file_path = os.path.join(root, file)
                    print(f"正在加载: {file_path}")
                    doc_chunks = self.load_document(file_path)
                    if doc_chunks:
                        documents.extend(doc_chunks) # 展平列表

        return documents

''' documents example:
[
    {
        "content": "--- 第 1 页 ---\n第一页的文本内容\n",
        "filename": "report.pdf",
        "filepath": "/data/report.pdf",
        "filetype": ".pdf",
        "page_number": 1
    },
    {
        "content": "--- 第 2 页 ---\n第二页的文本内容\n",
        "filename": "report.pdf",
        "filepath": "/data/report.pdf",
        "filetype": ".pdf",
        "page_number": 2
    },
    {
        "content": "--- 幻灯片 1 ---\n演讲内容\n",
        "filename": "presentation.pptx",
        "filepath": "/data/presentation.pptx",
        "filetype": ".pptx",
        "page_number": 1
    },
    {
        "content": "整个 Word 文档内容",
        "filename": "doc.docx",
        "filepath": "/data/doc.docx",
        "filetype": ".docx",
        "page_number": 0
    },
    {
        "content": "纯文本文件内容",
        "filename": "notes.txt",
        "filepath": "/data/notes.txt",
        "filetype": ".txt",
        "page_number": 0
    }
]
'''