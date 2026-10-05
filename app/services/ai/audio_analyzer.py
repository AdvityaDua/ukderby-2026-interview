"""
Async Audio Confidence Analyzer using Deepgram Pre-recorded API.

Processes saved interview WAV recordings to extract speech metrics:
- Speech Pace (WPM)
- Filler Word Rate
- Pause Analysis
- Word-level Confidence
- Speech-to-Silence Ratio
"""

import os
import glob
import asyncio
import httpx
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

AUDIO_DIR = Path(__file__).parent.parent / "audio_recordings"

FILLER_WORDS = {
    "um", "uh", "uhm", "hmm", "hm",
    "like", "basically", "actually", "literally",
    "you know", "i mean", "sort of", "kind of",
    "right", "okay", "so yeah",
}

# Single-word fillers for fast lookup
SINGLE_FILLERS = {w for w in FILLER_WORDS if " " not in w}
# Multi-word fillers
MULTI_FILLERS = [w for w in FILLER_WORDS if " " in w]

DEEPGRAM_PRERECORDED_URL = "https://api.deepgram.com/v1/listen"


@dataclass
class AudioMetrics:
    """Computed speech metrics for a single answer recording."""
    answer_index: int
    duration_seconds: float = 0.0
    word_count: int = 0
    words_per_minute: float = 0.0
    filler_word_count: int = 0
    filler_words_per_minute: float = 0.0
    fillers_found: List[str] = field(default_factory=list)
    avg_word_confidence: float = 0.0
    pause_count: int = 0          # pauses > 1 second
    avg_pause_duration: float = 0.0
    longest_pause: float = 0.0
    speech_to_silence_ratio: float = 0.0
    # Individual sub-scores (0-100)
    pace_score: float = 0.0
    filler_score: float = 0.0
    pause_score: float = 0.0
    confidence_score_word: float = 0.0
    speech_ratio_score: float = 0.0
    # Composite
    composite_confidence: float = 0.0
    transcript: str = ""


@dataclass
class SessionAnalysis:
    """Aggregate analysis for all answers in a session."""
    session_id: str
    total_answers: int = 0
    overall_confidence: float = 0.0
    avg_wpm: float = 0.0
    total_filler_count: int = 0
    avg_filler_rate: float = 0.0
    avg_word_confidence: float = 0.0
    avg_speech_ratio: float = 0.0
    per_answer: List[dict] = field(default_factory=list)


class AudioAnalyzer:
    """Analyzes saved interview audio using Deepgram pre-recorded API."""

    def __init__(self):
        self.api_key = os.getenv("DEEPGRAM_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_KEY not found in environment")

    async def _transcribe_file(self, filepath: str) -> dict:
        """Send a WAV file to Deepgram pre-recorded API."""
        params = {
            "model": "nova-2",
            "language": "en",
            "smart_format": "true",
            "utterances": "true",
            "diarize": "false",
            "punctuate": "true",
            "filler_words": "true",    # Keep filler words in transcript
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(filepath, "rb") as f:
                audio_data = f.read()

            response = await client.post(
                DEEPGRAM_PRERECORDED_URL,
                params=params,
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "audio/wav",
                },
                content=audio_data,
            )

            if response.status_code != 200:
                return None

            return response.json()

    def _compute_metrics(self, dg_response: dict, answer_index: int) -> AudioMetrics:
        """Extract all metrics from Deepgram's response."""
        metrics = AudioMetrics(answer_index=answer_index)

        if not dg_response:
            return metrics

        results = dg_response.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            return metrics

        alt = channels[0].get("alternatives", [{}])[0]
        words = alt.get("words", [])
        transcript = alt.get("transcript", "")
        metrics.transcript = transcript

        if not words:
            return metrics

        # ── Duration ──
        metadata = dg_response.get("metadata", {})
        metrics.duration_seconds = metadata.get("duration", 0.0)
        if metrics.duration_seconds == 0 and words:
            metrics.duration_seconds = words[-1].get("end", 0.0)

        duration_min = metrics.duration_seconds / 60.0 if metrics.duration_seconds > 0 else 1.0

        # ── Word Count & WPM ──
        metrics.word_count = len(words)
        metrics.words_per_minute = round(metrics.word_count / duration_min, 1)

        # ── Filler Words ──
        transcript_lower = transcript.lower()
        fillers_found = []

        # Check multi-word fillers in full transcript
        for mf in MULTI_FILLERS:
            count = transcript_lower.count(mf)
            fillers_found.extend([mf] * count)

        # Check single-word fillers
        for w in words:
            word_text = w.get("word", "").lower().strip(".,!?")
            if word_text in SINGLE_FILLERS:
                fillers_found.append(word_text)

        metrics.filler_word_count = len(fillers_found)
        metrics.fillers_found = fillers_found
        metrics.filler_words_per_minute = round(metrics.filler_word_count / duration_min, 2)

        # ── Word Confidence ──
        confidences = [w.get("confidence", 0.0) for w in words]
        metrics.avg_word_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        # ── Pause Analysis ──
        pauses = []
        for i in range(1, len(words)):
            gap = words[i].get("start", 0) - words[i - 1].get("end", 0)
            if gap > 1.0:
                pauses.append(gap)

        metrics.pause_count = len(pauses)
        metrics.avg_pause_duration = round(sum(pauses) / len(pauses), 2) if pauses else 0.0
        metrics.longest_pause = round(max(pauses), 2) if pauses else 0.0

        # ── Speech-to-Silence Ratio ──
        total_speech_time = sum(
            w.get("end", 0) - w.get("start", 0) for w in words
        )
        metrics.speech_to_silence_ratio = round(
            (total_speech_time / metrics.duration_seconds * 100) if metrics.duration_seconds > 0 else 0, 1
        )

        # ── Sub-Scores (each 0-100) ──
        metrics.pace_score = self._score_pace(metrics.words_per_minute)
        metrics.filler_score = self._score_fillers(metrics.filler_words_per_minute)
        metrics.pause_score = self._score_pauses(metrics.pause_count, duration_min)
        metrics.confidence_score_word = round(metrics.avg_word_confidence * 100, 1)
        metrics.speech_ratio_score = self._score_speech_ratio(metrics.speech_to_silence_ratio)

        # ── Composite Confidence Score ──
        metrics.composite_confidence = round(
            metrics.pace_score * 0.15 +
            metrics.filler_score * 0.20 +
            metrics.pause_score * 0.20 +
            metrics.confidence_score_word * 0.25 +
            metrics.speech_ratio_score * 0.20,
            1
        )

        return metrics

    # ── Scoring Functions ──

    @staticmethod
    def _score_pace(wpm: float) -> float:
        """120-150 WPM = ideal (100). Falls off outside that range."""
        if 120 <= wpm <= 150:
            return 100.0
        elif 100 <= wpm < 120:
            return 70 + (wpm - 100) * 1.5  # 70-100
        elif 150 < wpm <= 180:
            return 70 + (180 - wpm) * 1.0  # 70-100
        elif 80 <= wpm < 100:
            return 40 + (wpm - 80) * 1.5   # 40-70
        elif 180 < wpm <= 200:
            return 40 + (200 - wpm) * 1.5   # 40-70
        else:
            return max(10, 40 - abs(wpm - 135) * 0.5)

    @staticmethod
    def _score_fillers(fillers_per_min: float) -> float:
        """<1/min = excellent, >6/min = poor."""
        if fillers_per_min <= 1:
            return 100.0
        elif fillers_per_min <= 2:
            return 85.0
        elif fillers_per_min <= 3:
            return 70.0
        elif fillers_per_min <= 5:
            return 50.0
        elif fillers_per_min <= 8:
            return 30.0
        else:
            return 10.0

    @staticmethod
    def _score_pauses(pause_count: int, duration_min: float) -> float:
        """Score based on pauses per minute (>1s gaps)."""
        if duration_min <= 0:
            return 50.0
        pauses_per_min = pause_count / duration_min
        if pauses_per_min <= 1:
            return 100.0
        elif pauses_per_min <= 2:
            return 80.0
        elif pauses_per_min <= 4:
            return 60.0
        elif pauses_per_min <= 6:
            return 40.0
        else:
            return 20.0

    @staticmethod
    def _score_speech_ratio(ratio_pct: float) -> float:
        """70%+ = great, <30% = very poor."""
        if ratio_pct >= 70:
            return 100.0
        elif ratio_pct >= 55:
            return 80.0
        elif ratio_pct >= 40:
            return 60.0
        elif ratio_pct >= 25:
            return 40.0
        else:
            return 20.0

    # ── Main Entry Points ──

    async def analyze_audio_file(self, filepath: str, answer_index: int) -> AudioMetrics:
        """Analyze a single audio file and return metrics."""
        dg_response = await self._transcribe_file(filepath)
        metrics = self._compute_metrics(dg_response, answer_index)
        return metrics

    async def analyze_session(self, session_id: str) -> SessionAnalysis:
        """
        Analyze all audio files for a given session.
        Runs all file analyses concurrently.
        """
        pattern = str(AUDIO_DIR / f"{session_id}_answer_*.wav")
        files = sorted(glob.glob(pattern))

        if not files:
            return SessionAnalysis(session_id=session_id)

        if files:
            pass # Pattern match logic below

        # Run all analyses concurrently
        tasks = [
            self.analyze_audio_file(fp, idx + 1)
            for idx, fp in enumerate(files)
        ]
        results: List[AudioMetrics] = await asyncio.gather(*tasks)

        # Aggregate
        valid = [r for r in results if r.word_count > 0]
        analysis = SessionAnalysis(
            session_id=session_id,
            total_answers=len(files),
            per_answer=[asdict(r) for r in results],
        )

        if valid:
            analysis.overall_confidence = round(
                sum(r.composite_confidence for r in valid) / len(valid), 1
            )
            analysis.avg_wpm = round(
                sum(r.words_per_minute for r in valid) / len(valid), 1
            )
            analysis.total_filler_count = sum(r.filler_word_count for r in valid)
            analysis.avg_filler_rate = round(
                sum(r.filler_words_per_minute for r in valid) / len(valid), 2
            )
            analysis.avg_word_confidence = round(
                sum(r.avg_word_confidence for r in valid) / len(valid), 4
            )
            analysis.avg_speech_ratio = round(
                sum(r.speech_to_silence_ratio for r in valid) / len(valid), 1
            )

        return analysis
