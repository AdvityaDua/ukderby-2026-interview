"""Topic-specific interview planning.

Builds a lightweight topic interview plan from admin-managed topic metadata.
"""

from __future__ import annotations

import logging
import httpx
from typing import Any, Dict, List, Optional

from app.core.config import settings
from .langgraph_agent import InterviewPlan

logger = logging.getLogger(__name__)


async def fetch_topic_config(topic_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch TopicInterview configuration from the admin panel's Nest.js API.
    
    Args:
        topic_id: MongoDB ObjectId or topic identifier
        
    Returns:
        Topic configuration dict or None if not found
    """
    try:
        nest_api_base = settings.NEST_API_BASE_URL or settings.BACKEND_URL
        nest_api_base = nest_api_base.rstrip("/")
        api_url = f"{nest_api_base}/api/topic-interviews/{topic_id}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(api_url)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Topic config not found: {topic_id}")
                return None
            else:
                logger.error(f"Failed to fetch topic config: {response.status_code}")
                return None
                
    except Exception as exc:
        logger.error(f"Error fetching topic config: {exc}")
        return None


async def generate_topic_plan(
    topic_id: str,
    topic_config: Optional[Dict[str, Any]] = None,
    difficulty: str = "mid",
    duration_minutes: int = 30,
    subtopics: Optional[List[str]] = None,
) -> InterviewPlan:
    """
    Generate an interview plan for a topic-specific interview.
    
    Fetches the topic configuration from admin panel if not provided,
    then uses the configured resources and prompt templates.
    
    Args:
        topic_id: Unique identifier for the topic
        topic_config: Pre-fetched topic configuration (optional)
        difficulty: Interview difficulty level (junior, mid, senior, expert)
        duration_minutes: Interview duration guidance
        subtopics: Optional list of subtopics to focus on
        
    Returns:
        InterviewPlan compatible with existing InterviewGraph
    """
    difficulty = difficulty.lower().strip()
    if difficulty not in ("junior", "mid", "senior", "expert"):
        difficulty = "mid"
    
    # Fetch config from admin panel if not provided
    if not topic_config:
        topic_config = await fetch_topic_config(topic_id)
    
    topic_name = topic_config.get("name", topic_id.title()) if topic_config else topic_id.title()
    links = topic_config.get("links", []) if topic_config else []
    topic_focus = [topic_name]

    questions_for_level: List[str] = []
    
    # Build question candidates in format expected by token_optimized_agent
    question_candidates: List[Dict[str, Any]] = [
        {
            "question": q,
            "topic": topic_id,
            "difficulty": difficulty,
            "is_coding_question": _is_coding_question(q, topic_id),
            "source": "topic_admin_links",
        }
        for q in questions_for_level
    ]
    
    # Calculate question strategy based on duration
    max_questions = _max_questions_for_duration(duration_minutes)
    max_per_topic = _max_questions_per_topic(duration_minutes)
    
    plan: InterviewPlan = {
        "company": topic_name,
        "role": f"{topic_name} Engineer",
        "interview_type": "topic-specific",
        "topics": subtopics or topic_focus,
        "question_strategy": {
            "question_budget": max_questions,
            "max_questions_per_topic": max_per_topic,
            "max_confusion_retries": 2,
            "difficulty": difficulty,
        },
        "evaluation_criteria": {
            "focus_areas": subtopics or topic_focus,
            "depth_expected": difficulty,
            "resources_available": len(links) > 0,
        },
        "company_signals": {
            "topic_specific": True,
            "topic_id": topic_id,
            "has_resources": len(links) > 0,
        },
        "token_budget": {
            "init": 2000,
            "per_turn": 1500,
            "max_total": 15000,
        },
        "metadata": {
            "topic_config": {
                "name": topic_name,
                "links": links,
            },
            "created_from": "topic_admin_links",
        },
        "source_evidence": [
            {
                "source": "topic_admin_links",
                "topic": topic_id,
                "difficulty": difficulty,
                "resources_count": len(links),
            }
        ],
        "question_candidates": question_candidates,
        "opening_line": _get_opening_line(topic_name, difficulty),
        "closing_line": _get_closing_line(topic_name),
    }
    
    return plan


def _max_questions_for_duration(duration_minutes: int) -> int:
    """Calculate max questions based on duration."""
    if duration_minutes <= 0:
        return 0
    if duration_minutes <= 15:
        return 8
    if duration_minutes <= 30:
        return 13
    if duration_minutes <= 45:
        return 16
    return 18


def _max_questions_per_topic(duration_minutes: int) -> int:
    """Calculate max questions per topic based on duration."""
    if duration_minutes <= 0:
        return 0
    if duration_minutes <= 15:
        return 1
    if duration_minutes <= 30:
        return 2
    if duration_minutes <= 45:
        return 2
    return 3


def _is_coding_question(question: str, topic_id: str) -> bool:
    """Determine if a question typically requires coding."""
    coding_topics = {"dsa", "algorithms", "python", "node", "typescript", "react", "java"}
    coding_keywords = {"write", "implement", "code", "design", "build"}
    
    question_lower = question.lower()
    return (topic_id.lower() in coding_topics) or any(kw in question_lower for kw in coding_keywords)


def _should_include_coding(topic_id: str) -> bool:
    """Determine if topic should include coding questions."""
    coding_topics = {"react", "node", "python", "typescript", "java", "dsa", "algorithms"}
    return topic_id.lower() in coding_topics


def _get_opening_line(topic_name: str, difficulty: str) -> str:
    """Generate opening line for the interview."""
    difficulty_context = {
        "junior": "fundamentals and basic understanding",
        "mid": "practical experience and problem-solving",
        "senior": "system design and architectural thinking",
        "expert": "advanced patterns and optimization",
    }
    
    context = difficulty_context.get(difficulty, "technical expertise")
    return f"Welcome! Today we'll be assessing your {topic_name} skills, focusing on {context}. Let's get started!"


def _get_closing_line(topic_name: str) -> str:
    """Generate closing line for the interview."""
    return f"That concludes our {topic_name} interview. Thank you for your time and thoughtful answers. We'll be in touch soon!"
