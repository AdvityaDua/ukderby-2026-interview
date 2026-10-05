# Product Requirements Document: RAG Implementation


# Overview

This backend now supports a specialized Retrieval-Augmented Generation flow for company-specific interview rounds. The RAG implementation is not a generic vector-search layer inside the interview graph; it is a Vertex AI managed corpus pipeline that indexes topic documents, retrieves relevant grounding chunks, and injects them into the interview prompt as company intelligence.

In practice, the system uses two layers of context:

1. Resume + JD summarization for normal interview personalization.
2. Vertex AI RAG grounding for company/topic-specific interview knowledge banks.

The RAG layer is only activated when the interview is running in a specialized company mode.

## Problem Statement

The interview engine needs a reliable way to ground questions in company-specific material without manually hardcoding interview questions into prompts. It must:

- Accept topic documents, especially PDFs.
- Store them in a retrievable knowledge corpus.
- Retrieve semantically relevant passages on demand.
- Inject those passages into the interview context.
- Keep the interview prompt focused on company-specific standards rather than generic questions.

## Goals

- Index topic documents into a dedicated Vertex AI RAG corpus.
- Retrieve semantically relevant passages by topic.
- Use retrieved knowledge to steer interview questions.
- Support company-specific interview modes without affecting standard interviews.
- Keep the system simple to operate through REST endpoints.

## Non-Goals

- This is not a full document management platform.
- This is not a manual chunking or custom embedding pipeline.
- This is not used for normal resume-to-JD matching in the standard interview flow.
- This does not currently expose a custom ranking layer beyond Vertex AI retrieval.

## Core User Stories

1. As an operator, I can upload a PDF for a topic or company knowledge bank.
2. As an operator, I can query a topic corpus and see grounded results.
3. As an interviewer, I can start a company-specific interview and have the model use the company RAG context.
4. As the system, I can fall back to the standard resume/JD flow if no company knowledge exists.

## System Architecture

The implementation is split into three main parts:

- API routes for indexing, querying, and deleting RAG corpora.
- A Vertex AI RAG service wrapper that owns corpus lifecycle and retrieval.
- A company interview session that fetches grounding and appends it to the prompt context.

### Main Files

- [app/services/ai/rag_service.py](../app/services/ai/rag_service.py)
- [app/api/v1/endpoints/rag.py](../app/api/v1/endpoints/rag.py)
- [app/api/v1/router.py](../app/api/v1/router.py)
- [app/services/ai/company_interviewer.py](../app/services/ai/company_interviewer.py)
- [app/services/ai/interview_graph.py](../app/services/ai/interview_graph.py)
- [app/api/v1/ws_interview.py](../app/api/v1/ws_interview.py)

## Functional Flow

### 1. Corpus Creation and Indexing

When a PDF is uploaded to the RAG endpoint, the backend saves it to a temporary file and passes it to the RAG service.

Flow:

```text
POST /api/v1/rag/index-file
  -> save temp PDF
  -> rag_service.index_pdf()
  -> get or create Vertex AI corpus for topic_<topic_id>
  -> Vertex AI uploads file
  -> Vertex AI handles parsing, chunking, and embedding
  -> temp file is removed
```

Implementation details:

- The endpoint lives in [app/api/v1/endpoints/rag.py](../app/api/v1/endpoints/rag.py).
- The temp upload directory is `temp_rag_uploads`.
- The corpus is named `topic_<topic_id>`.
- The service relies on Vertex AI RAG managed storage rather than custom local vector indexing.

### 2. Retrieval

When a topic is queried, the backend locates the matching corpus and performs a retrieval query.

Flow:

```text
GET /api/v1/rag/query?topic_id=...&query=...&k=...
  -> rag_service.query_topic()
  -> locate corpus by display name
  -> Vertex AI retrieval_query()
  -> return top-k contexts with source metadata
```

Returned result shape:

- `content`: the retrieved passage text
- `metadata.source_document`: source URI if available, otherwise `Vertex RAG`

### 3. Grounding Injection for Interviews

The RAG knowledge bank becomes active in company-specific interview sessions.

Flow:

```text
WebSocket interview starts
  -> ws_interview.py chooses CompanyInterviewSession when company is present
  -> CompanyInterviewSession.initialize_session()
  -> normal resume + JD summarization runs first
  -> company name is resolved to a backend knowledge topic
  -> rag_service.get_topic_grounding(topic_id)
  -> retrieved chunks are appended to context_summary
  -> interview prompt is built with company RAG content included
  -> interview_graph.py forces RAG-first question behavior for company rounds
```

This means the model does not directly query the corpus during every turn. Instead, it retrieves a representative grounding bundle once at session initialization and injects that bundle into the prompt context for the rest of the interview.

## Runtime Behavior

### Standard Interview Mode

If no company is specified:

- The system uses [app/services/ai/streaming_session.py](../app/services/ai/streaming_session.py).
- It generates a resume + JD summary with Gemini.
- It caches the summary and extracted skills in Redis.
- It uses that context in the interview graph.

This is personalization, but not Vertex RAG.

### Company Interview Mode

If a company name is provided:

- The system uses [app/services/ai/company_interviewer.py](../app/services/ai/company_interviewer.py).
- It first runs the standard session initialization.
- It then looks up a knowledge topic for that company.
- If grounding exists, it appends the company RAG block to `context_summary`.
- The prompt rules in [app/services/ai/interview_graph.py](../app/services/ai/interview_graph.py) instruct the model to prioritize the RAG bank over generic resume exploration.

## Data Lifecycle

### Input

- PDF document upload for RAG indexing.
- Topic ID used as the corpus namespace.
- Query text for retrieval.
- Company name for interview grounding lookup.

### Storage

- Vertex AI RAG corpus stores the indexed document collection.
- Redis stores resume + JD context summaries and extracted skills.
- Temporary PDFs are stored locally only during upload processing.

### Output

- Indexed corpus entry in Vertex AI.
- Retrieved grounding chunks from topic queries.
- Injected company knowledge in the interview prompt.
- Interview questions that follow the company-specific intelligence bank.

## Configuration Dependencies

The RAG stack depends on the following settings in [app/core/config.py](../app/core/config.py):

- `GCP_PROJECT_ID`
- `GCP_RAG_LOCATION`
- `GOOGLE_APPLICATION_CREDENTIALS`

The middleware explicitly bypasses the RAG routes so they can be accessed without standard JWT authentication in the current setup.

## Operational Concerns

- If no corpus exists for a topic, retrieval returns an empty list and the interview falls back to normal context.
- If retrieval fails, the system logs the error and continues with the base interview flow.
- The company grounding is added once during session initialization, which keeps turn-by-turn latency lower than querying the corpus every turn.
- The current implementation is limited to PDFs for indexing.

## Acceptance Criteria

- A PDF can be indexed into a topic corpus successfully.
- A query returns the top-k relevant chunks for that topic.
- A company interview session can fetch RAG grounding and append it to the prompt context.
- The interview graph uses that context to prioritize RAG-backed questions.
- Standard interviews still work when no company knowledge exists.

## Summary

The current RAG implementation is a managed Vertex AI knowledge bank used for company-specific interview grounding. It is triggered by the REST RAG endpoints for indexing and retrieval, then consumed by the interview session initialization when a company round is selected. The standard resume/JD pipeline still exists separately and should be thought of as prompt personalization, while the Vertex AI path is the actual retrieval layer.