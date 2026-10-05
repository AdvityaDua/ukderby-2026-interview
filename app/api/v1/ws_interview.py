import json
import asyncio
import os
import re
from dataclasses import asdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.api.ws_manager import manager
from app.api.ws_dependencies import get_ws_current_user
from app.services.ai.stt_service import stt_service
from app.services.ai.audio_analyzer import AudioAnalyzer
from app.services.ai.gemini_client import GeminiClient
from app.services.ai.streaming_session import StreamingInterviewSession
from app.models.db import SessionLocal, Company, Topic, GithubLink
from app.services.ai.company_interviewer import CompanyGeminiClient, CompanyInterviewSession

ws_router = APIRouter()


def _extract_text_from_file(file_path: str) -> str:
    """Extract plain text from a PDF, DOCX, or TXT file on disk."""
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".pdf":
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts).strip()
        elif ext in (".docx", ".doc"):
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"[WS] ⚠️  Could not extract text from {file_path}: {e}")
    return ""


def _resolve_resume_url_to_path(resume_url: str) -> str | None:
    """
    Try to map a resume URL like http://api.aiforjob.ai/uploads/users/xyz/file.pdf
    to a local filesystem path like ./uploads/users/xyz/file.pdf.
    Also handles paths that look like relative paths already.
    """
    if not resume_url:
        return None

    # URL-decode in case the frontend sent encoded characters (e.g. spaces as %20)
    from urllib.parse import unquote
    resume_url_decoded = unquote(resume_url)
    print(f"[WS] URL resolve: original={resume_url!r}")
    print(f"[WS] URL resolve: decoded ={resume_url_decoded!r}")

    # Strip the protocol + host to get the path segment
    # e.g. http://api.aiforjob.ai/uploads/users/... -> uploads/users/...
    path_part = re.sub(r'^https?://[^/]+/', '', resume_url_decoded)
    print(f"[WS] URL resolve: path_part={path_part!r}")

    # Try both relative (from CWD) and the NestJS backend root
    candidates = [
        path_part,                                          # relative to CWD
        os.path.join("..", "backend", path_part),          # ../backend/uploads/...
        os.path.join("..", path_part),
        os.path.abspath(os.path.join("..", "backend", path_part)),
    ]

    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        print(f"[WS] URL resolve: checking {normalized}  exists={os.path.isfile(normalized)}")
        if os.path.isfile(normalized):
            print(f"[WS] ✅ Resolved resume URL -> {normalized}")
            return normalized

    print(f"[WS] ❌ Could not resolve resume URL to local file: {resume_url_decoded}")
    return None

answer_counts = {}

# Shared analyzer instance
audio_analyzer = AudioAnalyzer()


async def _analyze_and_cleanup(user_id: str, audio_path: str, answer_idx: int):
    """Background task: analyze audio, store metrics, delete file."""
    try:
        metrics = await audio_analyzer.analyze_audio_file(audio_path, answer_idx)
        metrics_dict = asdict(metrics)
        manager.add_audio_metrics(user_id, metrics_dict)

        # Persist updated metrics to Redis
        await manager.save_session_to_cache(user_id)

        pass
    except Exception as e:
        pass
    finally:
        # Delete the audio file to free disk space
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            pass


@ws_router.websocket("/transcribe/{client_id}")
async def stt_stream_endpoint(
    websocket: WebSocket, 
    client_id: str,
    user_payload: dict = Depends(get_ws_current_user)
):
    """
    **Real-time Speech-To-Text WebSocket**
    Connect here with `?token=YOUR_JWT` to stream raw audio bytes.
    Receives JSON with transcription updates (`"type": "partial"` or `"type": "final"`).
    After each answer, fires a background task to analyze audio quality.
    """
    user_id = user_payload.get("sub")
    await websocket.accept()

    if user_id not in answer_counts:
        answer_counts[user_id] = 0
    answer_counts[user_id] += 1
    answer_idx = answer_counts[user_id]

    dg_stream = None

    try:
        async def on_partial(text):
            try:
                await websocket.send_json({"type": "partial", "text": text})
            except Exception:
                pass

        async def on_final(text):
            try:
                await websocket.send_json({"type": "final", "text": text})
            except Exception:
                pass

        dg_stream = await stt_service.create_stream(on_partial, on_final)

        while True:
            message = await websocket.receive()
            if "bytes" in message:
                await dg_stream.send(message["bytes"])
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "end":
                    await dg_stream.finish()
                    audio_path = dg_stream.save_audio(user_id, answer_idx)
                    if audio_path:
                        # Fire background analysis — don't await it
                        asyncio.create_task(
                            _analyze_and_cleanup(user_id, audio_path, answer_idx)
                        )
                    dg_stream = None
                    break 

    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass
    finally:
        if dg_stream:
            try:
                await dg_stream.finish()
                audio_path = dg_stream.save_audio(user_id, answer_idx)
                if audio_path:
                    asyncio.create_task(
                        _analyze_and_cleanup(user_id, audio_path, answer_idx)
                    )
            except Exception:
                pass


def _aggregate_audio_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate per-answer audio metrics into a session summary."""
    if not metrics_list:
        return {}

    valid = [m for m in metrics_list if m.get("word_count", 0) > 0]
    if not valid:
        return {"total_answers_analyzed": 0, "per_answer": metrics_list}

    n = len(valid)
    return {
        "total_answers_analyzed": len(metrics_list),
        "overall_confidence": round(sum(m["composite_confidence"] for m in valid) / n, 1),
        "avg_wpm": round(sum(m["words_per_minute"] for m in valid) / n, 1),
        "total_filler_count": sum(m["filler_word_count"] for m in valid),
        "avg_filler_rate": round(sum(m["filler_words_per_minute"] for m in valid) / n, 2),
        "avg_word_confidence": round(sum(m["avg_word_confidence"] for m in valid) / n, 4),
        "avg_speech_ratio": round(sum(m["speech_to_silence_ratio"] for m in valid) / n, 1),
        "per_answer": metrics_list,
    }


@ws_router.websocket("/chat/{client_id}")
async def stream_interview_endpoint(
    websocket: WebSocket, 
    client_id: str,
    user_payload: dict = Depends(get_ws_current_user)
):
    """
    **Full-Duplex Interview Chat WebSocket**
    """
    user_id = user_payload.get("sub")
    session: StreamingInterviewSession | None = None

    def _user_message_count(active_session: StreamingInterviewSession | None) -> int:
        if not active_session:
            return 0
        return sum(
            1 for msg in active_session.history
            if msg.get("role") in ("user", "candidate")
        )

    await manager.connect(websocket, user_id)
    print(f"[WS] Client {client_id} (User: {user_id}) connected.")
    
    try:
        session, restored = await manager.get_or_create_session(user_id)
        
        if restored and session.history:
            print(f"[WS] Restoring session for {user_id} with {len(session.history)} messages.")
            restored_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in session.history
            ]
            await manager.send_json({
                "type": "restored",
                "messages": restored_messages,
                "interview_type": session.interview_type,
                "role": session.role,
                "company": session.company,
            }, user_id)
        
        elif not session.history:
            print(f"[WS] Waiting for init payload from {user_id}...")
            # We wrap this in a timeout to prevent hanging forever if client doesn't send init
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                init_payload = json.loads(data)
            except asyncio.TimeoutError:
                print(f"[WS] Timeout waiting for init from {user_id}")
                await manager.send_json({"type": "error", "content": "Initialization timeout. Please try again."}, user_id)
                await manager.clear_session(user_id)
                await websocket.close()
                return

            if init_payload.get("type") == "init":
                # Check if we should actually clear or if this is a "redundant" init after restoration
                is_reconnection = False
                if restored and session.history:
                    # If we restored a session with history, we usually want to KEEP it.
                    # We check if the incoming init matches the existing session's context.
                    if (session.interview_type == init_payload.get("interview_type", "technical") and
                        session.role == init_payload.get("role", "") and
                        session.company == init_payload.get("company", "")):
                        print(f"[WS] ⚡ Valid restoration detected for {user_id}. Keeping history.")
                        is_reconnection = True

                if not is_reconnection:
                    # Clear any existing session to ensure fresh start with new resume
                    print(f"[WS] Clearing any existing session for {user_id} before new init")
                    await manager.clear_session(user_id)
                    
                    resume_text = init_payload.get("resume_text", "")
                    resume_url  = init_payload.get("resume_url", "")
                    resume_path = init_payload.get("resume_path", "")  # raw relative path from MongoDB
                    jd_text = init_payload.get("jd_text", "")
                    interview_type = init_payload.get("interview_type", "technical")
                    interview_mode = init_payload.get("interview_mode", "general")
                    topic_id = init_payload.get("topic_id", "")
                    topic_name = init_payload.get("topic_name", "")
                    role = init_payload.get("role", "")
                    company = init_payload.get("company", "")
                    duration = init_payload.get("duration", 0)
                    
                    # --- DB QUERY FOR GITHUB LINKS ---
                    github_urls = []
                    try:
                        db = SessionLocal()
                        
                        # 1. Fetch by company name
                        if company:
                            comp_record = db.query(Company).filter(Company.name.ilike(company)).first()
                            if comp_record:
                                for link in comp_record.github_links:
                                    github_urls.append(link.url)
                        
                        # 2. Fetch by topic_id
                        if topic_id:
                            try:
                                tid = int(topic_id)
                                topic_record = db.query(Topic).filter(Topic.id == tid).first()
                                if topic_record:
                                    for link in topic_record.github_links:
                                        if link.url not in github_urls:
                                            github_urls.append(link.url)
                            except ValueError:
                                pass
                                
                        db.close()
                    except Exception as db_err:
                        print(f"[WS] DB Error fetching github links: {db_err}")
                        
                    if github_urls:
                        print(f"[WS] Found GitHub Links: {github_urls}")
                        # Inject into jd_text so the AI graph gets it
                        jd_text = (jd_text or "") + "\n\nIMPORTANT CONTEXT: The following GitHub repositories are highly relevant for this interview round. Use your tools to analyze these repos for generating coding questions:\n" + "\n".join(github_urls)
                    # ---------------------------------
                    candidate_name = init_payload.get("candidate_name", "")

                    # ── Debug: print everything received ──
                    print(f"[WS] ============ INIT PAYLOAD RECEIVED ============")
                    print(f"[WS] User: {user_id} | Round: {interview_type} | Role: {role} | Company: {company}")
                    print(f"[WS] Mode: {interview_mode} | TopicId: {topic_id} | TopicName: {topic_name}")
                    print(f"[WS] Candidate name: {candidate_name!r}")
                    print(f"[WS] resume_text length from frontend: {len(resume_text)} chars")
                    print(f"[WS] resume_url:  {resume_url!r}")
                    print(f"[WS] resume_path: {resume_path!r}")
                    print(f"[WS] jd_text length: {len(jd_text)} chars")

                    # ── Fallback: extract resume text from file on disk if empty ──
                    if len(resume_text.strip()) < 50:
                        local_path = None

                        # Strategy 1: use the raw relative path directly (most reliable)
                        if resume_path:
                            print(f"[WS] Strategy 1: trying raw resume_path directly: {resume_path!r}")
                            candidates = [
                                resume_path,
                                os.path.join("..", "backend", resume_path),
                                os.path.abspath(os.path.join("..", "backend", resume_path)),
                            ]
                            for c in candidates:
                                norm = os.path.normpath(c)
                                print(f"[WS]   checking {norm} ... exists={os.path.isfile(norm)}")
                                if os.path.isfile(norm):
                                    local_path = norm
                                    print(f"[WS] ✅ Found via resume_path: {local_path}")
                                    break

                        # Strategy 2: fall back to URL-based resolution
                        if not local_path and resume_url:
                            print(f"[WS] Strategy 2: trying URL resolution for: {resume_url!r}")
                            local_path = _resolve_resume_url_to_path(resume_url)

                        if local_path:
                            resume_text = await asyncio.to_thread(_extract_text_from_file, local_path)
                            print(f"[WS] Extracted {len(resume_text)} chars from {local_path}")
                        else:
                            print(f"[WS] ⚠️  Could not locate resume file via path or URL.")

                    # ── Final debug: print resume content ──
                    if resume_text and len(resume_text.strip()) > 10:
                        print(f"[WS] ✅ RESUME TEXT OK ({len(resume_text)} chars total)")
                        print(f"[WS] --- RESUME PREVIEW (first 600 chars) ---")
                        print(resume_text[:600])
                        print(f"[WS] -------------------------------------------")
                    else:
                        print(f"[WS] ❌ RESUME TEXT IS EMPTY! AI will not be personalized.")

                    if jd_text and len(jd_text.strip()) > 10:
                        print(f"[WS] ✅ JD TEXT OK ({len(jd_text)} chars). Preview: {jd_text[:200]}")
                    else:
                        print(f"[WS] ⚠️  JD TEXT is empty.")

                    print(f"[WS] ================================================")
                    print(f"[WS] Initializing context for {user_id} ({interview_type} round)...")
                    
                    # Choose interviewer type.
                    is_topic_mode = (
                        str(interview_mode).lower() == "topic"
                        or str(interview_type).lower() == "topic-specific"
                        or bool(topic_id)
                    )

                    # Topic mode must not be routed to CompanyInterviewer.
                    if is_topic_mode:
                        print(f"[WS] 📚 Topic interviewer mode chosen for topic '{topic_name or topic_id or role}'")
                        client = GeminiClient()
                        session = StreamingInterviewSession(client)
                        company = ""
                        if not role and (topic_name or topic_id):
                            role = f"{topic_name or topic_id} Engineer"
                        if str(interview_type).lower() != "topic-specific":
                            interview_type = "topic-specific"
                    elif company and len(company.strip()) > 1:
                        print(f"[WS] 🚀 Specialized Company Interviewer chosen for '{company}'")
                        client = CompanyGeminiClient()
                        session = CompanyInterviewSession(client)
                    else:
                        print(f"[WS] 🤖 Standard Interviewer chosen")
                        client = GeminiClient()
                        session = StreamingInterviewSession(client)

                    manager.sessions[user_id] = session
                    manager.audio_metrics[user_id] = []
                    
                    try:
                        await manager.send_json({"type": "info", "content": "Analyzing resume & role..."}, user_id)
                        await session.initialize_session(
                            user_id,
                            client_id,
                            resume_text, 
                            jd_text, 
                            interview_type, 
                            role, 
                            company, 
                            duration,
                            candidate_name
                        )
                        await manager.send_json({"type": "info", "content": "Context initialized."}, user_id)
                        
                        await manager.save_session_to_cache(user_id)
                        
                        print(f"[WS] Starting first AI response for {user_id}")
                        await manager.send_json({"type": "stream_start"}, user_id)
                        async for item in session.stream_response(None):
                            if item["type"] == "metadata":
                                await manager.send_json({"type": "metadata", "is_coding": item["is_coding"]}, user_id)
                            else:
                                await manager.send_json({"type": "text", "content": item["content"]}, user_id)
                        await manager.send_json({"type": "stream_end"}, user_id)
                        
                        await manager.save_session_to_cache(user_id)
                    except Exception as e:
                        print(f"[WS] Initialization error for {user_id}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        try:
                            await manager.send_json({"type": "error", "content": f"AI Engine failed to initialize: {str(e)}"}, user_id)
                        except Exception:
                            pass
                        await manager.clear_session(user_id)
                        return  # ← EXIT: do NOT fall into the main message loop after init failure
            
        # Main message loop
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("type") == "message":
                user_text = payload.get("content")
                print(f"[WS] Message from {user_id}: {user_text[:50]}...")
                
                try:
                    await manager.send_json({"type": "stream_start"}, user_id)
                    async for item in session.stream_response(user_text):
                        if item["type"] == "metadata":
                            await manager.send_json({"type": "metadata", "is_coding": item["is_coding"]}, user_id)
                        elif item["type"] == "audio":
                            await manager.send_json({"type": "audio", "base64": item["base64"]}, user_id)
                        else:
                            await manager.send_json({"type": "text", "content": item["content"]}, user_id)
                    await manager.send_json({"type": "stream_end"}, user_id)
                    
                    await manager.save_session_to_cache(user_id)
                    
                    if getattr(session, 'ended', False):
                        await manager.send_json({"type": "info", "content": "Interview complete. Generating feedback..."}, user_id)
                        
                        try:
                            feedback, feedback_usage = await session.client.generate_feedback(session.history, session.context_summary)
                            
                            # Add feedback generation tokens to session
                            session.input_tokens += feedback_usage.get("input_tokens", 0)
                            session.output_tokens += feedback_usage.get("output_tokens", 0)
                            print(f"[WS] Final tokens after feedback: in={session.input_tokens}, out={session.output_tokens}")
                            
                            audio_summary = _aggregate_audio_metrics(manager.get_audio_metrics(user_id))
                            
                            await manager.send_json({
                                "type": "end_interview",
                                "feedback": feedback.model_dump(),
                                "audio_analysis": audio_summary,
                            }, user_id)
                        except Exception as fb_err:
                            print(f"[WS] Feedback generation error for {user_id}: {str(fb_err)}")
                            await manager.send_json({
                                "type": "end_interview",
                                "feedback": None,
                                "error": str(fb_err),
                                "audio_analysis": _aggregate_audio_metrics(manager.get_audio_metrics(user_id)),
                            }, user_id)
                        
                        # REPORT USAGE TO BACKEND
                        asyncio.create_task(session.report_usage(user_id, client_id))

                        await manager.clear_session(user_id)
                        break 
                except Exception as e:
                    print(f"[WS] Error during response for {user_id}: {str(e)}")
                    await manager.send_json({"type": "error", "content": f"AI Engine error: {str(e)}"}, user_id)
                    await manager.send_json({"type": "stream_end"}, user_id)
            
            elif payload.get("type") == "end_session":
                print(f"[WS] Manual/auto end session for {user_id}")

                # Count how many messages the user actually sent
                user_msg_count = _user_message_count(session)
                print(f"[WS] Session history has {len(session.history)} total messages, {user_msg_count} from user")

                if user_msg_count == 0:
                    print(f"[WS] ❌ User answered 0 questions — refusing to generate fake feedback.")
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": None,
                        "error": "no_answers",
                        "audio_analysis": {},
                    }, user_id)
                    await manager.clear_session(user_id)
                    break

                await manager.send_json({"type": "info", "content": "Ending session. Generating analysis..."}, user_id)
                try:
                    feedback, feedback_usage = await session.client.generate_feedback(session.history, session.context_summary)
                    
                    # Add feedback generation tokens to session
                    session.input_tokens += feedback_usage.get("input_tokens", 0)
                    session.output_tokens += feedback_usage.get("output_tokens", 0)
                    print(f"[WS] Final tokens after feedback: in={session.input_tokens}, out={session.output_tokens}")
                    
                    audio_summary = _aggregate_audio_metrics(manager.get_audio_metrics(user_id))
                    
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": feedback.model_dump(),
                        "audio_analysis": audio_summary,
                    }, user_id)
                except Exception as fb_err:
                    print(f"[WS] Feedback generation error for {user_id}: {str(fb_err)}")
                    await manager.send_json({
                        "type": "end_interview",
                        "feedback": None,
                        "audio_analysis": _aggregate_audio_metrics(manager.get_audio_metrics(user_id)),
                        "error": str(fb_err)
                    }, user_id)
                
                # REPORT USAGE TO BACKEND
                asyncio.create_task(session.report_usage(user_id, client_id))

                await manager.clear_session(user_id)
                break

            elif payload.get("type") == "ping":
                await manager.send_json({"type": "pong"}, user_id)

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected.")
        # If user exits before answering, treat it as an ended session and purge cache.
        active_session = session or manager.sessions.get(user_id)
        if _user_message_count(active_session) == 0:
            print(f"[WS] Clearing session for {user_id} after early disconnect (no user answers).")
            await manager.clear_session(user_id)
        manager.disconnect(user_id)
    except Exception as e:
        print(f"[WS] Unexpected error in stream endpoint: {str(e)}")
        active_session = session or manager.sessions.get(user_id)
        if _user_message_count(active_session) == 0:
            print(f"[WS] Clearing session for {user_id} after unexpected early error (no user answers).")
            await manager.clear_session(user_id)
        manager.disconnect(user_id)
        try:
            await websocket.close()
        except:
            pass
