"""Source-backed interview planning utilities.

This module builds interview topic maps and question candidates from the source
families provided by the user: GitHub question-bank repositories plus public
web interview-prep sources. It avoids using local exported artifacts as a
question source.
"""

from __future__ import annotations

import hashlib
import logging
import re
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .langgraph_agent import InterviewPlan

logger = logging.getLogger(__name__)
_CACHE_PATH = Path(__file__).resolve().parents[3] / ".cache" / "source_interview_documents.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60


SOURCE_REPOSITORIES = [
    "https://github.com/liquidslr/interview-company-wise-problems",
    "https://github.com/snehasishroy/leetcode-companywise-interview-questions",
    "https://github.com/krishnadey30/LeetCode-Questions-CompanyWise",
    "https://github.com/realabbas/big-companies-interview-questions",
    "https://github.com/jwasham/coding-interview-university",
]

SOURCE_WEB_URLS = [
    "https://leetcode.com/",
    "https://www.geeksforgeeks.org/tag/interview-experience/",
    "https://www.glassdoor.com/Interview/index.htm",
    "https://www.reddit.com/r/leetcode/",
]

_QUESTION_PREFIXES = (
    "why",
    "what",
    "how",
    "tell me",
    "describe",
    "explain",
    "walk me through",
    "can you",
    "would you",
)

_TOPIC_RULES = {
    "System Design": ["system design", "scalability", "architecture", "distributed", "microservices", "cache"],
    "Algorithms": ["algorithm", "algorithms", "complexity", "leetcode", "dynamic programming", "graph"],
    "Data Structures": ["array", "linked list", "stack", "queue", "hash", "tree", "trie", "heap"],
    "Databases": ["database", "sql", "postgres", "mysql", "mongodb", "redis"],
    "Behavioral": ["behavioral", "teamwork", "leadership", "conflict", "communication", "resume"],
    "Frontend": ["react", "frontend", "typescript", "ui", "performance", "accessibility"],
    "Backend": ["api", "backend", "service", "rest", "graphql", "websocket"],
    "DevOps": ["docker", "kubernetes", "deployment", "ci/cd", "aws", "gcp"],
    "Machine Learning": ["machine learning", "ml", "model", "training", "inference", "embedding"],
    "Career": ["resume", "interview", "salary", "career", "job", "offer"],
}


@dataclass(frozen=True)
class SourceQuestionCandidate:
    question: str
    topic: str
    source_label: str
    source_url: str
    fingerprint: str
    evidence: str
    is_coding_question: bool


def question_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _parse_repo_url(repo_url: str) -> tuple[str, str]:
    parsed = urlparse(repo_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
    return parts[0], parts[1]


def _guess_default_branch(owner: str, repo: str) -> str:
    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=15)
    response.raise_for_status()
    return response.json().get("default_branch", "main")


def _github_tree_files(owner: str, repo: str, branch: str) -> list[str]:
    response = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
        timeout=20,
    )
    response.raise_for_status()
    tree = response.json().get("tree", [])
    candidates: list[str] = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = str(entry.get("path", ""))
        lower = path.lower()
        if not lower.endswith((".md", ".txt", ".rst", ".org")):
            continue
        if any(token in lower for token in ("readme", "interview", "question", "design", "algorithm", "behavior", "system")):
            candidates.append(path)
    if not candidates:
        for entry in tree:
            if entry.get("type") == "blob" and str(entry.get("path", "")).lower().endswith((".md", ".txt")):
                candidates.append(str(entry.get("path", "")))
    return candidates[:40]


def _fetch_raw_github(owner: str, repo: str, branch: str, path: str) -> str:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    response = requests.get(raw_url, timeout=20)
    response.raise_for_status()
    return response.text


def _load_github_documents(repo_url: str) -> list[Document]:
    owner, repo = _parse_repo_url(repo_url)
    branch = _guess_default_branch(owner, repo)
    files = _github_tree_files(owner, repo, branch)
    documents: list[Document] = []
    for path in files:
        try:
            content = _fetch_raw_github(owner, repo, branch, path)
        except Exception as exc:
            logger.debug("Skipping GitHub file %s/%s: %s", repo_url, path, exc)
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source_type": "github",
                    "source_url": repo_url,
                    "source_label": f"{owner}/{repo}:{path}",
                    "repo": f"{owner}/{repo}",
                    "path": path,
                },
            )
        )
    return documents


def _load_web_documents(urls: Iterable[str]) -> list[Document]:
    documents: list[Document] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_type": "web",
                        "source_url": url,
                        "source_label": url,
                    },
                )
            )
        except Exception as exc:
            logger.debug("Skipping web source %s: %s", url, exc)
    return documents


def _extract_topics(text: str) -> list[str]:
    lowered = text.lower()
    topics: list[str] = []
    for topic, keywords in _TOPIC_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            topics.append(topic)
    headings = re.findall(r"^#{2,4}\s+(.+)$", text, flags=re.MULTILINE)
    for heading in headings:
        cleaned = re.sub(r"\s+", " ", heading).strip()
        if cleaned and cleaned not in topics and len(cleaned) <= 80:
            if any(keyword in cleaned.lower() for keyword in ("system", "interview", "design", "coding", "behavior", "resume", "algorithm", "data")):
                topics.append(cleaned)
    return list(dict.fromkeys(topics))


def _extract_question_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-•*").strip()
        if not line:
            continue
        lower = line.lower()
        if line.endswith("?") or any(lower.startswith(prefix) for prefix in _QUESTION_PREFIXES):
            if 18 <= len(line) <= 180 and "http" not in lower and "[" not in line and "]" not in line:
                lines.append(line)
    return lines


def _clean_context_fragment(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    return " ".join(words[:12])


def _company_tokens(company_name: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", company_name.lower()) if len(token) >= 3]
    if company_name:
        tokens.extend([company_name.lower().strip()])
    return list(dict.fromkeys(tokens))


def _document_relevance_score(document: Document, query_tokens: set[str], company_tokens: list[str]) -> int:
    content = document.page_content.lower()
    source_label = str(document.metadata.get("source_label", "")).lower()
    path = str(document.metadata.get("path", "")).lower()
    score = 0

    for token in query_tokens:
        if token and token in content:
            score += 1

    for token in company_tokens:
        if token and (token in content or token in source_label or token in path):
            score += 8

    if any(token in source_label for token in company_tokens):
        score += 5

    if document.metadata.get("source_type") == "github":
        score += 2

    return score


def _theme_question_from_topic(topic: str, context_line: str) -> str:
    topic_lower = topic.lower()
    context = _clean_context_fragment(context_line) or topic
    if "system" in topic_lower or "design" in topic_lower:
        return f"How would you design a scalable {context} system, and what trade-offs would you call out?"
    if "behavior" in topic_lower or "resume" in topic_lower or "interview" in topic_lower:
        return f"Tell me about a time you worked on {context}, and what you learned from it."
    if "algorithm" in topic_lower or "data structure" in topic_lower or "coding" in topic_lower or "leetcode" in topic_lower:
        return f"Walk me through a problem in {context} from first principles, including complexity and edge cases."
    if "database" in topic_lower or "sql" in topic_lower or "redis" in topic_lower:
        return f"How would you design and justify the data model for {context}?"
    return f"What would you focus on first when preparing for {context} interviews?"


def _is_coding_topic(topic: str) -> bool:
    topic_lower = topic.lower()
    return any(marker in topic_lower for marker in ("algorithm", "data structure", "coding", "leetcode", "system design", "database", "sql", "redis"))


def _summarize_documents(documents: list[Document]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for document in documents[:5]:
        content = re.sub(r"\s+", " ", document.page_content).strip()
        snippets = []
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            if len(sentence) > 50:
                snippets.append(sentence.strip())
            if len(snippets) >= 2:
                break
        summaries.append(
            {
                "source_label": document.metadata.get("source_label", document.metadata.get("source_url", "source")),
                "source_url": document.metadata.get("source_url", ""),
                "snippets": snippets[:2],
            }
        )
    return summaries


def _interleave_candidates(candidate_groups: list[list[SourceQuestionCandidate]], limit: int) -> list[SourceQuestionCandidate]:
    merged: list[SourceQuestionCandidate] = []
    seen: set[str] = set()
    positions = [0 for _ in candidate_groups]

    while len(merged) < limit:
        progressed = False
        for group_index, group in enumerate(candidate_groups):
            while positions[group_index] < len(group):
                candidate = group[positions[group_index]]
                positions[group_index] += 1
                if candidate.fingerprint in seen:
                    continue
                seen.add(candidate.fingerprint)
                merged.append(candidate)
                progressed = True
                break
        if not progressed:
            break

    return merged[:limit]


class SourceInterviewPlanner:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
        self._documents: list[Document] = []

    def load_documents(self) -> list[Document]:
        if self._documents:
            return self._documents

        cached = self._load_cached_documents()
        if cached:
            self._documents = cached
            return self._documents

        documents: list[Document] = []
        for repo_url in SOURCE_REPOSITORIES:
            try:
                documents.extend(_load_github_documents(repo_url))
            except Exception as exc:
                logger.warning("Failed to load GitHub source %s: %s", repo_url, exc)

        try:
            documents.extend(_load_web_documents(SOURCE_WEB_URLS))
        except Exception as exc:
            logger.warning("Failed to load web sources: %s", exc)

        self._documents = self._splitter.split_documents(documents)
        self._save_cached_documents(self._documents)
        return self._documents

    def _load_cached_documents(self) -> list[Document]:
        try:
            if not _CACHE_PATH.exists():
                return []
            if (_CACHE_PATH.stat().st_mtime + _CACHE_TTL_SECONDS) < time.time():
                return []
            with _CACHE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            documents = [
                Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}))
                for item in payload
                if item.get("page_content")
            ]
            if documents:
                logger.info("Loaded %s source documents from cache", len(documents))
            return documents
        except Exception as exc:
            logger.debug("Could not load source corpus cache: %s", exc)
            return []

    def _save_cached_documents(self, documents: list[Document]) -> None:
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = [
                {"page_content": document.page_content, "metadata": document.metadata}
                for document in documents
            ]
            with _CACHE_PATH.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception as exc:
            logger.debug("Could not save source corpus cache: %s", exc)

    def build_question_candidates(
        self,
        *,
        query: str,
        company_name: str = "",
        excluded_fingerprints: set[str] | None = None,
        limit: int = 6,
    ) -> list[SourceQuestionCandidate]:
        excluded_fingerprints = excluded_fingerprints or set()
        documents = self.load_documents()
        scored: list[tuple[int, Document]] = []
        query_lower = query.lower()
        query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
        company_terms = _company_tokens(company_name)

        for document in documents:
            score = _document_relevance_score(document, query_tokens, company_terms)
            if score:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_window = scored[: max(limit * 5, 20)]
        if len(top_window) > 1:
            random.SystemRandom().shuffle(top_window)
        candidates: list[SourceQuestionCandidate] = []
        company_hit = False
        for _, document in top_window[: max(limit * 3, 12)]:
            if str(document.metadata.get("source_type", "")) == "web":
                continue
            source_label = str(document.metadata.get("source_label", document.metadata.get("source_url", "source")))
            source_url = str(document.metadata.get("source_url", ""))
            topics = _extract_topics(document.page_content)
            explicit_questions = _extract_question_lines(document.page_content)
            if len(explicit_questions) > 1:
                random.SystemRandom().shuffle(explicit_questions)
            evidence = re.sub(r"\s+", " ", document.page_content).strip()[:350]
            company_matched = any(token and token in f"{source_label} {source_url} {document.page_content}".lower() for token in company_terms)
            company_hit = company_hit or company_matched

            if explicit_questions:
                for question in explicit_questions:
                    fingerprint = question_fingerprint(question)
                    if fingerprint in excluded_fingerprints:
                        continue
                    topic = topics[0] if topics else "General"
                    candidates.append(
                        SourceQuestionCandidate(
                            question=question,
                            topic=topic,
                            source_label=source_label,
                            source_url=source_url,
                            fingerprint=fingerprint,
                            evidence=evidence,
                            is_coding_question=_is_coding_topic(topic),
                        )
                    )
                    if len(candidates) >= limit:
                        return candidates

            for topic in topics[:3]:
                question = _theme_question_from_topic(topic, evidence[:100])
                fingerprint = question_fingerprint(question)
                if fingerprint in excluded_fingerprints:
                    continue
                candidates.append(
                    SourceQuestionCandidate(
                        question=question,
                        topic=topic,
                        source_label=source_label,
                        source_url=source_url,
                        fingerprint=fingerprint,
                        evidence=evidence,
                        is_coding_question=_is_coding_topic(topic),
                    )
                )
                if len(candidates) >= limit:
                    return candidates

        if not candidates:
            fallback_topic = "Algorithms" if "technical" in query_lower or "problem" in query_lower else "System Design"
            fallback_question = _theme_question_from_topic(fallback_topic, query)
            fingerprint = question_fingerprint(fallback_question)
            if fingerprint not in excluded_fingerprints:
                candidates.append(
                    SourceQuestionCandidate(
                        question=fallback_question,
                        topic=fallback_topic,
                        source_label="fallback",
                        source_url="",
                        fingerprint=fingerprint,
                        evidence=query,
                        is_coding_question=True,
                    )
                )
        return candidates[:limit]

    async def build_plan(
        self,
        *,
        resume_text: str,
        jd_text: str,
        company_name: str,
        role: str = "",
        interview_type: str = "technical",
        duration_minutes: int = 30,
        excluded_question_hashes: Iterable[str] | None = None,
    ) -> "InterviewPlan":
        excluded = set(excluded_question_hashes or [])
        company_query = "\n".join([company_name, role, interview_type])
        jd_query = "\n".join([jd_text, resume_text])
        if interview_type == "technical":
            company_candidates = self.build_question_candidates(
                query=company_query,
                company_name=company_name,
                excluded_fingerprints=excluded,
                limit=3,
            )
            resume_candidates = self.build_question_candidates(
                query=f"{resume_text}\n{role}\n{interview_type}",
                company_name="",
                excluded_fingerprints=excluded,
                limit=2,
            )
            jd_candidates = self.build_question_candidates(
                query=jd_query,
                company_name="",
                excluded_fingerprints=excluded,
                limit=2,
            )
            candidates = _interleave_candidates([company_candidates, resume_candidates, jd_candidates], limit=6)
            company_hit = bool(company_candidates and any(candidate.source_label != "fallback" for candidate in company_candidates))
        else:
            candidates = self.build_question_candidates(
                query=company_query,
                company_name=company_name,
                excluded_fingerprints=excluded,
                limit=6,
            )
            company_hit = bool(candidates and any(candidate.source_label != "fallback" for candidate in candidates))
            if not company_hit:
                fallback_candidates = self.build_question_candidates(
                    query=jd_query,
                    company_name="",
                    excluded_fingerprints=excluded,
                    limit=4,
                )
                for candidate in fallback_candidates:
                    if candidate.fingerprint not in {item.fingerprint for item in candidates}:
                        candidates.append(candidate)
        source_documents = self.load_documents()
        source_summaries = _summarize_documents(source_documents)

        from .langgraph_agent import generate_interview_plan

        base_plan = generate_interview_plan(
            resume_text=resume_text,
            jd_text=jd_text,
            company_name=company_name,
            role=role,
            interview_type=interview_type,
            duration_minutes=duration_minutes,
        )

        topics = list(base_plan.get("topics", []))
        for candidate in candidates:
            if candidate.topic and candidate.topic not in topics:
                topics.append(candidate.topic)

        greeting_line = f"Hi, thanks for joining us today. Let's get started with an interview focused on {role or interview_type} and the role fit for {company_name or 'your target company'}."
        closing_line = "Thank you so much for interviewing with us today. We really appreciate your time, and we’ll share the next steps soon."

        base_plan.update(
            {
                "topics": topics[:12],
                "source_evidence": source_summaries,
                "question_candidates": [candidate.__dict__ for candidate in candidates],
                "opening_line": greeting_line,
                "closing_line": closing_line,
                "question_strategy": {
                    **base_plan.get("question_strategy", {}),
                    "use_source_candidates": True,
                    "avoid_repeating_hashes": True,
                    "source_priority": ["github_repositories", "web_sources"],
                    "company_first": True,
                    "jd_fallback_only": interview_type != "technical",
                    "source_mix": {
                        "company": 3 if company_name.strip() else 1,
                        "resume": 2,
                        "jd": 2,
                    },
                    "requires_coding_question": interview_type in {"technical", "problem"},
                },
                "question_source_mode": "company_resume_jd_mix" if interview_type == "technical" else "company_first_jd_fallback",
                "company_match_found": company_hit,
                "needs_coding_question": interview_type in {"technical", "problem"},
            }
        )
        return base_plan


source_interview_planner = SourceInterviewPlanner()
