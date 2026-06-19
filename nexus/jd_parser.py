"""
NEXUS Layer 1 — JD INTELLIGENCE PARSER
Parses the Job Description into a machine-readable scoring contract.
The JD profile is hard-coded from the Redrob Senior AI/ML Engineer JD.
"""
from __future__ import annotations

from nexus.config import (
    CONSULTING_COMPANIES,
    PREFERRED_CITIES,
    PREFERRED_COUNTRY,
    AI_CURIOUS_PHRASES,
    JD_QUERY_TERMS,
)

# ─────────────────────────────────────────────────────────────────────────────
# JD Profile — derived from careful reading of job_description.docx
# ─────────────────────────────────────────────────────────────────────────────

JD_PROFILE = {
    "role": "Senior AI/ML Engineer — Intelligence Layer",
    "company": "Redrob",
    "experience_years": {"min": 5, "max": 9},  # Range, not hard cutoff
    "experience_ideal": {"min": 6, "max": 8},
    "ai_ml_years_ideal": {"min": 4, "max": 5},  # Of which 4-5 in applied ML/AI

    "must_have_skills": [
        # Embeddings-based retrieval
        "sentence transformers", "openai embeddings", "bge", "e5",
        "embedding retrieval", "embeddings",
        # Vector databases / hybrid search
        "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch",
        "vector search", "hybrid search", "dense retrieval",
        # Evaluation
        "ndcg", "mrr", "map", "a/b testing", "ranking evaluation",
        # Python
        "python",
    ],

    "nice_to_have_skills": [
        "lora", "qlora", "peft", "fine-tuning", "llm fine-tuning",
        "learning to rank", "ltr", "xgboost", "lightgbm",
        "open source", "github",
        "distributed systems", "large-scale inference",
        "hr-tech", "recruiting", "marketplace",
        "reranking", "cross-encoder", "bm25",
    ],

    # Hard disqualifiers — immediately push candidate far down the ranking
    "hard_disqualifiers": {
        "consulting_only_career": True,     # Entire career at TCS/Wipro/etc.
        "pure_research_no_production": True, # Academic-only, never shipped
        "code_inactive_18_months": True,     # No coding in 18+ months
        "cv_speech_robotics_only": True,     # Primary domain is not NLP/IR
    },

    # The 'ideal candidate' paragraph from the JD
    "ideal_profile": {
        "total_years": (6, 8),
        "ai_ml_years_at_product_company": (4, 5),
        "shipped_ranking_search_recommendation": True,
        "strong_opinions_on": ["retrieval", "evaluation", "llm_integration"],
        "location_preference": ["noida", "pune", "hyderabad", "mumbai", "delhi"],
        "active_on_platform": True,
    },

    # Salary range inferred from senior ML role in India
    "salary_range_inr_lpa": {"min": 25, "max": 80},

    "preferred_locations": list(PREFERRED_CITIES),
    "preferred_country": PREFERRED_COUNTRY,

    "bm25_query": " ".join(JD_QUERY_TERMS),
}


# ─────────────────────────────────────────────────────────────────────────────
# Disqualifier checks
# ─────────────────────────────────────────────────────────────────────────────

def is_hard_disqualified(candidate: dict) -> tuple[bool, str]:
    """Return (disqualified: bool, reason: str).

    Applies the explicit JD disqualifiers. A disqualified candidate gets
    a near-zero score (not zero — they still appear in output but ranked last).
    """
    profile = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    summary = profile.get("summary", "").lower()

    # ── Disqualifier 1: Entire career at consulting/services firms ─────────────
    if career:
        non_consulting_months = 0
        total_months = 0
        for role in career:
            company = role.get("company", "").lower().strip()
            duration = role.get("duration_months", 0)
            total_months += duration
            is_consulting = any(
                c in company for c in CONSULTING_COMPANIES
            )
            if not is_consulting:
                non_consulting_months += duration

        if total_months > 0:
            consulting_ratio = 1.0 - (non_consulting_months / total_months)
            if consulting_ratio >= 0.98 and total_months >= 36:  # 98%+ consulting, 3+ years
                return True, "Entire career at consulting/IT-services firms"

    # ── Disqualifier 2: AI-curious language in summary (non-engineer) ──────────
    ai_curious_count = sum(1 for phrase in AI_CURIOUS_PHRASES if phrase in summary)
    if ai_curious_count >= 2:
        return True, "Summary indicates AI-curious non-engineer (JD-explicit disqualifier)"

    return False, ""


def compute_disqualifier_penalty(candidate: dict) -> float:
    """Return a multiplier [0.05, 1.0] based on soft disqualifiers.

    Soft disqualifiers lower the score but don't eliminate the candidate entirely.
    """
    profile = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])
    skills: list[dict] = candidate.get("skills", [])
    summary = profile.get("summary", "").lower()
    penalty = 1.0

    # Soft penalty: primarily CV/Speech/Robotics domain
    NON_DOMAIN_SKILLS = {
        "image classification", "object detection", "computer vision",
        "speech recognition", "tts", "text-to-speech", "asr",
        "gans", "robotics"
    }
    skill_names = {s.get("name", "").lower() for s in skills}
    non_domain_count = len(skill_names & NON_DOMAIN_SKILLS)
    total_skills = max(1, len(skills))
    if non_domain_count / total_skills > 0.6:
        penalty *= 0.7  # Primarily wrong domain

    # Soft penalty: one AI-curious phrase
    ai_curious_count = sum(1 for phrase in AI_CURIOUS_PHRASES if phrase in summary)
    if ai_curious_count == 1:
        penalty *= 0.8

    return max(0.05, penalty)
