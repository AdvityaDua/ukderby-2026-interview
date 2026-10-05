"""
Question Bank Service
=====================
Parses admin-uploaded Markdown files into a structured question bank.
When an interview session starts the system picks a RANDOM starting question
and passes it to the LLM, which then drives follow-up / counter questions.

Expected Markdown format (admin uploads this):
------------------------------------------------
# Topic Name

## Entry Questions
- What is a Widget in Flutter?
- Explain the difference between StatelessWidget and StatefulWidget.

## Interview Questions

### Widgets & UI
- Explain the widget tree and element tree.
- How does Flutter's rendering pipeline work?

### State Management
- Compare setState, Provider, and Bloc.
- When would you choose Riverpod over Provider?

### Performance
- How do you profile a Flutter app?
- What causes jank and how do you fix it?
------------------------------------------------

Any heading level 2 (##) becomes a section.
Any heading level 3 (###) becomes a sub-section / category.
Bullet lines (- or *) under any section are extracted as questions.
"""

import os
import re
import json
import random
import pathlib
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Where parsed question banks are persisted as JSON
_BANK_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "question_banks"
_BANK_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_question_bank_md(content: str) -> Dict:
    """
    Parse an admin markdown file into a structured dict:
    {
        "title": "Flutter Development",
        "sections": {
            "Entry Questions": ["q1", "q2"],
            "Widgets & UI": ["q3", "q4"],
            ...
        },
        "entry_questions": ["q1", "q2"],   # shortcut — the ## Entry Questions section
        "all_questions": ["q1", "q2", ...]  # flat list of every question
    }
    """
    sections: Dict[str, List[str]] = {}
    current_section = "General"
    title = ""

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        # Top-level title
        if line.startswith("# ") and not line.startswith("##"):
            title = line[2:].strip()
            continue

        # Section heading (## or ###)
        if line.startswith("## "):
            current_section = line[3:].strip()
            if current_section not in sections:
                sections[current_section] = []
            continue

        if line.startswith("### "):
            current_section = line[4:].strip()
            if current_section not in sections:
                sections[current_section] = []
            continue

        # Question bullet
        if line.startswith("- ") or line.startswith("* "):
            question = line[2:].strip()
            if question:
                sections.setdefault(current_section, []).append(question)
            continue

        # Numbered list: "1. Question"
        numbered = re.match(r"^\d+\.\s+(.+)$", line)
        if numbered:
            question = numbered.group(1).strip()
            if question:
                sections.setdefault(current_section, []).append(question)

    # Build flat lists
    entry_questions: List[str] = []
    for key, qs in sections.items():
        if "entry" in key.lower():
            entry_questions.extend(qs)

    all_questions: List[str] = [q for qs in sections.values() for q in qs]

    return {
        "title": title,
        "sections": sections,
        "entry_questions": entry_questions,
        "all_questions": all_questions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bank_path(topic_id: str) -> pathlib.Path:
    return _BANK_DIR / f"{topic_id}.json"


def save_question_bank(topic_id: str, bank: Dict) -> None:
    path = _bank_path(topic_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
    logger.info("Saved question bank for topic %s (%d questions)", topic_id, len(bank["all_questions"]))


def load_question_bank(topic_id: str) -> Optional[Dict]:
    path = _bank_path(topic_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_question_bank(topic_id: str) -> bool:
    path = _bank_path(topic_id)
    if path.exists():
        path.unlink()
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Random question picker
# ─────────────────────────────────────────────────────────────────────────────

def get_random_question(topic_id: str, prefer_entry: bool = True) -> Optional[str]:
    """
    Pick a random question from the bank.
    If prefer_entry=True and entry questions exist, pick from those first.
    """
    bank = load_question_bank(topic_id)
    if not bank:
        return None

    if prefer_entry and bank.get("entry_questions"):
        pool = bank["entry_questions"]
    elif bank.get("all_questions"):
        pool = bank["all_questions"]
    else:
        return None

    return random.choice(pool)


def get_random_questions(topic_id: str, n: int = 5, prefer_entry: bool = True) -> List[str]:
    """
    Pick up to n unique random questions from the bank.
    Entry questions are included first if prefer_entry=True.
    """
    bank = load_question_bank(topic_id)
    if not bank:
        return []

    pool: List[str] = []
    if prefer_entry and bank.get("entry_questions"):
        pool = list(bank["entry_questions"])
        others = [q for q in bank["all_questions"] if q not in set(pool)]
        random.shuffle(pool)
        random.shuffle(others)
        pool = pool + others
    else:
        pool = list(bank["all_questions"])
        random.shuffle(pool)

    return pool[:n]


def get_bank_summary(topic_id: str) -> str:
    """
    Build a compact summary of the question bank for injection into the
    LLM system prompt.  Includes all section headings + questions so the
    LLM knows the full scope.
    """
    bank = load_question_bank(topic_id)
    if not bank:
        return ""

    lines = []
    if bank.get("title"):
        lines.append(f"Topic: {bank['title']}")
    lines.append("")

    for section, questions in bank.get("sections", {}).items():
        if not questions:
            continue
        lines.append(f"[{section}]")
        for q in questions:
            lines.append(f"  • {q}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Index from file path (called by API endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def index_md_file(file_path: str, topic_id: str) -> Dict:
    """Parse a markdown file and persist the question bank. Returns the bank dict."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    bank = parse_question_bank_md(content)
    save_question_bank(topic_id, bank)
    return bank
