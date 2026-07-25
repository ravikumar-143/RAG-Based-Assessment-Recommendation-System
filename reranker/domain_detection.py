"""Domain detection for balancing recommendations."""
from __future__ import annotations
import re
from typing import Dict

TECHNICAL_KEYWORDS = [
    "developer", "engineering", "software", "cloud", "data", "sql", "python", "java",
    "aws", "azure", "devops", "network", "security", "qa", "tester", "full stack",
    "frontend", "backend", "ios", "android", "ml", "ai", "analytics",
]

BEHAVIORAL_KEYWORDS = [
    "leadership", "communication", "teamwork", "collaboration", "manager", "people",
    "behavior", "culture", "values", "stakeholder", "conflict", "emotional", "resilience",
]

COGNITIVE_KEYWORDS = [
    "aptitude", "reasoning", "logic", "cognitive", "numerical", "verbal", "abstract",
    "situational", "problem solving", "critical thinking", "assessment", "ability",
]


def detect_domains(query: str) -> Dict[str, int]:
    text = query.lower()
    def count_keywords(keywords):
        return sum(bool(re.search(rf"\b{re.escape(k)}\b", text)) for k in keywords)

    return {
        "technical": count_keywords(TECHNICAL_KEYWORDS),
        "behavioral": count_keywords(BEHAVIORAL_KEYWORDS),
        "cognitive": count_keywords(COGNITIVE_KEYWORDS),
    }
