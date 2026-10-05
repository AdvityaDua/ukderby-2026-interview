import os
from pypdf import PdfReader
from docx import Document

class ResumeParser:
    @staticmethod
    def extract_text(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return ResumeParser._extract_from_pdf(file_path)
        elif ext == '.docx':
            return ResumeParser._extract_from_docx(file_path)
        elif ext == '.txt':
            return ResumeParser._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            raise RuntimeError(f"Error reading PDF: {e}")
        return text

    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            raise RuntimeError(f"Error reading DOCX: {e}")
        return text

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Error reading TXT: {e}")
