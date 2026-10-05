# AI Interview FastAPI Backend Documentation

This document outlines the API endpoints available in the newly architected FastAPI backend. This service replaces the legacy `ai-interview-dua` and `ai-interview-rahat` systems, bundling high-performance AI WebSockets and Resume Evaluation under one roof running on port `8001`.

## 1. WebSockets (Real-Time Interview)
These endpoints require a valid JWT token passed as a query parameter (`?token=YOUR_JWT`).


### `ws://localhost:8001/api/v1/interview/chat`
* **Purpose**: Primary conversational backbone for the AI interview.
* **Usage**: On connection, the client sends an initialized payload containing the candidate's resume, job description, role, etc. From there, it duplexes bidirectional audio and text messages, pushing the user's spoken words or chat into the Langchain Gemini Engine and returning the AI's spoken words or chat.

### `ws://localhost:8001/api/v1/interview/transcribe`
* **Purpose**: Active Real-time Speech-To-Text (STT) transcriber.
* **Usage**: Accepts continuous raw PCM 16-bit audio chunk streams. Evaluates pauses and chunks dynamically, communicating with Deepgram to return lightning-fast, highly accurate live transcripts.

---

## 2. Resume & CV Evaluation (REST API)
These endpoints require a standard `Authorization: Bearer <token>` header. They utilize blazingly fast Groq Llama models and Gemini 2.5 Pro for analysis.

### `POST /api/v1/resume/final-enhanced`
* **Purpose**: Core Resume Builder logic.
* **Payload**: Accepts a full `ResumeAnalysisRequest` containing CV sub-score arrays and existing metrics.
* **Returns**: A completely structured and formatted JSON object containing professional `Summary`, `Skills`, `Experience`, and `Projects` sections using Gemini 2.5 Pro.

### `POST /api/v1/cv/score`
* **Purpose**: Score purely the CV Quality (standalone, without JD matching).
* **Payload**: `{ "cv_text": "string" }`
* **Returns**: A JSON mapping the user's resume into dimension scores (Technical Depth, Formatting, Impact) with a banded rating.

### `POST /api/v1/cv/fit-index`
* **Purpose**: Calculate match percentage between Resume and Job Description.
* **Payload**: `{ "cv_text": "string", "jd_text": "string" }`
* **Returns**: Generates a unified JSON payload representing the `cv_quality`, `jd_match`, and an overarching `fit_index` representing their probability of passing ATS sweeps.

### `POST /api/v1/cv/improvement`
* **Purpose**: Actionable insights generator.
* **Payload**: `{ "cv_text": "string", "jd_text": "string" }`
* **Returns**: Generates the "Top 1% Gaps" feedback mapping out exactly what the candidate lacks in comparison to world-class applicant profiles. 

---

## 3. Upload & Extraction Tools 
### `POST /api/v1/upload/cv_evaluate` & `POST /api/v1/upload/cv_improvement`
* **Purpose**: Handles multi-part file uploads (PDF, DOCX, DOC, TXT, Images) using the underlying `pdfplumber` and `pytesseract` engines. 
* **Process**: Seamlessly extracts text upon upload and pipelines it immediately into the `evaluation_engine` or `improvement_engine` to prevent frontend client processing burdens.
