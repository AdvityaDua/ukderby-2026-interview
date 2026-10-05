from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
import shutil, tempfile, os, uuid
from typing import Optional
import httpx
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

from app.core.config import settings
from app.core.security import verify_jwt_token
from app.core.key_manager import key_manager
from app.services.cv_eval.evaluation_engine import evaluation_engine
from app.services.cv_eval.improvement_engine import Improvement


router = APIRouter()

# -----------------------
# Analytics helper
# -----------------------

async def _report_cv_token_usage(request: Request, usage: dict, source: str = "cv") -> None:
    """Extract user from JWT in Authorization header and POST token usage to analytics."""
    input_tokens  = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if input_tokens == 0 and output_tokens == 0:
        print(f"[CV-Upload] No tokens to report — skipping analytics POST.")
        return

    # Try to extract user_id from the forwarded Authorization header
    user_id = "anonymous"
    try:
        state = request.scope.get("state", {})
        user  = state.get("user") if isinstance(state, dict) else getattr(state, "user", None)
        if user:
            user_id = str(user.get("sub") or user.get("_id") or user.get("id", "anonymous"))
        else:
            # Fallback: decode JWT directly from header
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                payload = verify_jwt_token(auth.split(" ", 1)[1])
                user_id = str(payload.get("sub", "anonymous"))
    except Exception:
        pass

    in_cost  = (input_tokens  / 1_000_000) * 0.10   # Gemini 1.5/2.0 Flash pricing
    out_cost = (output_tokens / 1_000_000) * 0.40
    session_id = f"{source}_{user_id}_{uuid.uuid4().hex[:10]}"
    payload = {
        "userId":             user_id,
        "sessionId":          session_id,
        "model":              key_manager.get_gemini_model(),
        "inputTokens":        input_tokens,
        "outputTokens":       output_tokens,
        "totalTokens":        input_tokens + output_tokens,
        "costUsd":            in_cost + out_cost,
        "inputCostUsd":       in_cost,
        "outputCostUsd":      out_cost,
        "subscriptionStatus": "free",
        "source":             source,
        "interviewType":      "",
    }
    print(f"[CV-Upload] Reporting {source} usage to analytics (Gemini): in={input_tokens}, out={output_tokens}, user={user_id}")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{settings.BACKEND_URL}/analytics/ai-usage",
                json=payload,
                timeout=15.0,
            )
        print(f"[CV-Upload] analytics POST status={r.status_code}")
    except Exception as e:
        print(f"[CV-Upload] Failed to report usage: {type(e).__name__}: {e}")

# -----------------------
# Helpers
# -----------------------

def extract_text_from_pdf(file_path: str, max_pages: int = 7) -> str:
    """Extract text from PDF using pdfplumber with a configurable page limit."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            actual_pages = len(pdf.pages)
            if actual_pages > max_pages:
                raise ValueError(
                    f"Resume has {actual_pages} pages which exceeds the {max_pages}-page limit for your plan. "
                    f"Please trim your resume or upgrade to a plan that supports up to 20 pages."
                )
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception as e:
        if "page limit" in str(e) or "exceeds the" in str(e):
            raise
        # Fallback or other errors
        raise RuntimeError(f"Failed to extract PDF: {str(e)}")
        
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_doc(file_path: str) -> str:
    """
    Extract text from .doc files (old MS Word format).
    Uses antiword (available in Docker image) via subprocess.
    """
    import subprocess
    try:
        result = subprocess.run(["antiword", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return result.stdout.decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Failed to extract .doc file: {e}")


def extract_text_from_txt(file_path: str) -> str:
    """Extract plain text from .txt or .md files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_rtf(file_path: str) -> str:
    """
    Extract text from RTF files by stripping control words.
    (Simple fallback; avoids textract.)
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    import re
    return re.sub(r"{\\.*?}", "", data)


def extract_text_from_html(file_path: str) -> str:
    """Extract visible text from HTML using BeautifulSoup."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n")


def extract_text_from_odt(file_path: str) -> str:
    """
    Extract text from ODT using odfpy (if installed),
    or fallback to zip XML extraction.
    """
    try:
        from odf.opendocument import load
        from odf.text import P
        doc = load(file_path)
        return "\n".join([p.firstChild.data for p in doc.getElementsByType(P) if p.firstChild])
    except Exception:
        import zipfile, xml.etree.ElementTree as ET
        text = ""
        with zipfile.ZipFile(file_path) as z:
            with z.open("content.xml") as f:
                tree = ET.parse(f)
                for elem in tree.iter():
                    if elem.text:
                        text += elem.text + "\n"
        return text.strip()


def extract_text_from_image(file_path: str) -> str:
    """Extract text from images (JPG/PNG/TIFF) via Tesseract OCR."""
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from image: {e}")


def extract_text(file_path: str, max_pages: int = 7) -> str:
    """Dispatch extractor based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path, max_pages=max_pages)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".doc":
        return extract_text_from_doc(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".rtf":
        return extract_text_from_rtf(file_path)
    elif ext == ".md":
        return extract_text_from_txt(file_path)
    elif ext in [".html", ".htm"]:
        return extract_text_from_html(file_path)
    elif ext == ".odt":
        return extract_text_from_odt(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_and_extract(upload: UploadFile, max_pages: int = 7) -> str:
    """Temporarily save uploaded file, extract its text, and enforce plan-based limits."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.filename)[1]) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path, max_pages=max_pages)
        
        # Char-based overflow guard for non-PDF formats (can't count pages accurately)
        # 1 page ≈ 3000-4000 chars; free=7 pages≈21000 chars, paid=20 pages≈60000 chars
        MAX_CHARS = max_pages * 3500
        if len(text) > MAX_CHARS and not upload.filename.lower().endswith('.pdf'):
            raise ValueError(
                f"Extracted content exceeds the {max_pages}-page limit for your plan "
                f"(~{MAX_CHARS:,} characters allowed; your document has ~{len(text):,}). "
                f"Please trim your document or upgrade your plan."
            )
    finally:
        os.remove(tmp_path)
    return text


# -----------------------
# Endpoints
# -----------------------

@router.post("/cv_evaluate")
async def upload_and_evaluate_cv(
    request: Request,
    file: UploadFile = File(...),
    jd_text: str = Form("", description="Optional JD text"),
    jd_file: UploadFile = File(None),
    max_pages: int = Form(7, description="Max pages allowed (7=free, 20=premium)")
):
    """Upload CV (PDF/DOC/DOCX) and optionally JD for evaluation."""
    try:
        # Reset token counter for this request
        evaluation_engine.llm_scorer.reset_usage()

        cv_text = save_and_extract(file, max_pages=max_pages)

        if jd_text and jd_text.strip():
            result = evaluation_engine.evaluate(cv_text, jd_text)
        elif jd_file is not None:
            jd_extracted = save_and_extract(jd_file, max_pages=max_pages)
            result = evaluation_engine.evaluate(cv_text, jd_extracted)
        else:
            result = evaluation_engine.evaluate(cv_text)

        # Report token usage to analytics
        usage = evaluation_engine.llm_scorer.get_usage()
        await _report_cv_token_usage(request, usage, source="cv")
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {str(e)}")


# -----------------------
# Improvement Endpoint
# -----------------------

improvement_engine = Improvement()


@router.post("/cv_improvement")
async def upload_and_improve_cv(
    request: Request,
    file: UploadFile = File(...),
    jd_text: str = Form("", description="JD text (optional)"),
    jd_file: UploadFile = File(None),
    max_pages: int = Form(7, description="Max pages allowed (7=free, 20=premium)")
):
    """Upload CV and optionally JD → generate improved resume, benchmark, and cover letter."""
    try:
        # Reset token counter for this request
        improvement_engine.llm_scorer.reset_usage()

        cv_text = save_and_extract(file, max_pages=max_pages)

        if jd_text and jd_text.strip():
            result = improvement_engine.evaluate(cv_text, jd_text)
        elif jd_file is not None:
            jd_extracted = save_and_extract(jd_file, max_pages=max_pages)
            result = improvement_engine.evaluate(cv_text, jd_extracted)
        else:
            result = improvement_engine.evaluate(cv_text, "")

        # Report token usage to analytics
        usage = improvement_engine.llm_scorer.get_usage()
        await _report_cv_token_usage(request, usage, source="cv")
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Improvement failed: {str(e)}")


@router.post("/extract-text")
async def extract_text_endpoint(
    file: UploadFile = File(...)
):
    """Extract plain text from an uploaded document (PDF, DOCX, TXT, etc.) — used for JD upload."""
    try:
        text = save_and_extract(file, max_pages=20)
        return {"text": text.strip(), "filename": file.filename}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text extraction failed: {str(e)}")
