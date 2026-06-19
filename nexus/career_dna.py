"""
NEXUS Layer 2 — CAREER DNA EXTRACTOR
Extracts semantic understanding of a candidate's career trajectory.

This is the heart of NEXUS. It goes far beyond skills matching to understand:
  - Product company vs. consulting/services background
  - Genuine AI/ML engineering depth vs. AI-curious domain experts
  - Keyword stuffing detection (claims skills not evidenced in career)
  - Role trajectory (IC growth vs. manager drift)
  - Career-level AI/ML seniority estimation
"""
from __future__ import annotations

import re
from typing import Any

from nexus.config import (
    CONSULTING_COMPANIES,
    AI_CURIOUS_PHRASES,
    AI_SENIORITY_W,
    PRODUCT_RATIO_W,
    STUFFING_PENALTY_W,
    ROLE_TRAJECTORY_W,
    EDUCATION_W,
)

# ─────────────────────────────────────────────────────────────────────────────
# AI/ML career evidence keywords (in role descriptions — NOT skill tags)
# ─────────────────────────────────────────────────────────────────────────────

_STRONG_AI_EVIDENCE = [
    # Retrieval & ranking
    "embedding", "embeddings", "vector search", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "elasticsearch", "opensearch", "hybrid search",
    "dense retrieval", "sparse retrieval", "bm25", "reranking", "cross-encoder",
    "recommendation system", "ranking system", "search system",
    "information retrieval", "semantic search",
    # Production ML
    "deployed to production", "deployed in production", "shipped",
    "a/b test", "a/b testing", "online experiment",
    "ndcg", "mrr", "recall@", "precision@", "map@",
    "model serving", "model deployment", "inference", "latency",
    "mlflow", "bentoml", "ray serve", "triton", "torchserve",
    # LLM Engineering
    "fine-tun", "lora", "qlora", "peft", "rlhf",
    "rag", "retrieval augmented", "langchain", "llamaindex",
    "llm", "large language model",
    # ML fundamentals
    "neural network", "deep learning", "transformer",
    "pytorch", "tensorflow", "huggingface", "sklearn",
    "feature engineering", "feature store",
    "gradient boosting", "xgboost", "lightgbm",
]

_MODERATE_AI_EVIDENCE = [
    "machine learning", "ml model", "data science", "nlp",
    "natural language", "text classification", "sentiment",
    "pipeline", "spark", "airflow", "kafka",
    "sql", "python", "data pipeline", "etl",
    "model training", "model evaluation",
    "kaggle", "competition",
]

_AI_ROLE_TITLES = {
    "ml engineer", "machine learning engineer",
    "ai engineer", "data scientist", "applied scientist",
    "research engineer", "research scientist",
    "nlp engineer", "search engineer",
    "recommendation engineer", "ranking engineer",
    "mlops engineer", "platform engineer",
    "backend engineer", "software engineer",  # Potentially relevant
    "data engineer", "analytics engineer",
}

_NON_AI_TITLES = {
    "hr manager", "human resources", "recruiter",
    "marketing manager", "brand manager",
    "accountant", "finance manager",
    "graphic designer", "ux designer",
    "content writer", "copywriter",
    "sales executive", "business development",
    "operations manager", "supply chain",
    "mechanical engineer", "civil engineer",
    "electrical engineer", "customer support",
    "project manager",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_career_dna_score(candidate: dict[str, Any]) -> dict[str, float]:
    """Compute the full Career DNA score breakdown.

    Returns a dict with keys:
      total          - weighted composite [0, 1]
      ai_seniority   - depth of genuine AI/ML experience [0, 1]
      product_ratio  - fraction of career at product companies [0, 1]
      stuffing_clean - inverse of keyword-stuffing likelihood [0, 1]
      trajectory     - IC growth trajectory score [0, 1]
      education      - education prestige score [0, 1]
    """
    ai_seniority  = _score_ai_seniority(candidate)
    product_ratio = _score_product_ratio(candidate)
    stuffing_clean = _score_stuffing_cleanliness(candidate)
    trajectory    = _score_role_trajectory(candidate)
    education     = _score_education(candidate)

    total = (
        AI_SENIORITY_W     * ai_seniority
        + PRODUCT_RATIO_W  * product_ratio
        + STUFFING_PENALTY_W * stuffing_clean
        + ROLE_TRAJECTORY_W * trajectory
        + EDUCATION_W      * education
    )

    return {
        "total":         min(1.0, total),
        "ai_seniority":  ai_seniority,
        "product_ratio": product_ratio,
        "stuffing_clean":stuffing_clean,
        "trajectory":    trajectory,
        "education":     education,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_ai_seniority(candidate: dict[str, Any]) -> float:
    """Estimate genuine AI/ML engineering depth.

    Sources evidence from:
    - career_history role descriptions (strong vs. moderate keywords)
    - career_history job titles (AI/ML title bonus)
    - profile summary (AI-evidence vs. AI-curious language)
    - years_of_experience in appropriate range
    """
    career: list[dict] = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    summary = profile.get("summary", "").lower()

    strong_hits = 0
    moderate_hits = 0
    ai_role_months = 0
    total_months = max(1, sum(r.get("duration_months", 0) for r in career))

    for role in career:
        description = role.get("description", "").lower()
        title = role.get("title", "").lower()
        duration = role.get("duration_months", 0)

        # Count keyword hits in description
        role_strong = sum(1 for kw in _STRONG_AI_EVIDENCE if kw in description)
        role_moderate = sum(1 for kw in _MODERATE_AI_EVIDENCE if kw in description)

        strong_hits += min(role_strong, 10)  # Cap per role to avoid double-counting
        moderate_hits += min(role_moderate, 5)

        # AI role title bonus
        if any(t in title for t in _AI_ROLE_TITLES):
            ai_role_months += duration

    # Score from keyword density
    keyword_score = min(1.0, (strong_hits * 2 + moderate_hits) / 20.0)

    # AI role ratio score
    ai_role_ratio = ai_role_months / total_months
    role_score = min(1.0, ai_role_ratio * 1.2)  # Slight bonus for high ratio

    # Summary evidence (from profile.summary)
    summary_strong = sum(1 for kw in _STRONG_AI_EVIDENCE if kw in summary)
    summary_score = min(0.3, summary_strong * 0.05)

    # Penalty: AI-curious language
    curious_count = sum(1 for p in AI_CURIOUS_PHRASES if p in summary)
    curious_penalty = min(0.5, curious_count * 0.25)

    # Years of experience fit — ideal 6-8 years total
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 3 <= yoe < 5 or 9 < yoe <= 12:
        yoe_score = 0.8
    elif yoe > 12:
        yoe_score = 0.65  # Over-experienced but can still be relevant
    elif 2 <= yoe < 3:
        yoe_score = 0.5
    else:
        yoe_score = max(0.0, yoe / 5.0)

    combined = (
        0.40 * keyword_score
        + 0.35 * role_score
        + 0.15 * yoe_score
        + 0.10 * summary_score
        - curious_penalty
    )
    return max(0.0, min(1.0, combined))


def _score_product_ratio(candidate: dict[str, Any]) -> float:
    """Calculate fraction of career spent at product companies.

    Product company = NOT a consulting/IT-services firm.
    Returns higher score for more time at product companies.
    """
    career: list[dict] = candidate.get("career_history", [])
    if not career:
        return 0.3  # Unknown — give benefit of doubt

    product_months = 0
    total_months = 0

    for role in career:
        company_name = role.get("company", "").lower()
        duration = role.get("duration_months", 0)
        total_months += duration

        is_consulting = any(c in company_name for c in CONSULTING_COMPANIES)
        if not is_consulting:
            product_months += duration

        # Company size bonus — larger product companies signal real product work
        company_size = role.get("company_size", "")
        if not is_consulting and company_size in ("51-200", "201-500", "501-1000"):
            product_months += duration * 0.1  # Bonus for product-stage size

    if total_months == 0:
        return 0.3

    ratio = product_months / total_months
    # Apply slight bonus: even 60% product is significantly better than consulting-only
    if ratio >= 0.8:
        return 1.0
    elif ratio >= 0.6:
        return 0.85
    elif ratio >= 0.4:
        return 0.65
    elif ratio >= 0.2:
        return 0.45
    else:
        return max(0.05, ratio)


def _score_stuffing_cleanliness(candidate: dict[str, Any]) -> float:
    """Detect keyword stuffing and return inverse (high = clean candidate).

    Keyword stuffing = claiming many AI skills in the skills[] array
    but career_history contains NO AI/ML evidence whatsoever.
    The JD explicitly calls this out as the main trap.
    """
    skills: list[dict] = candidate.get("skills", [])
    career: list[dict] = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()

    # ── 1. Count AI skills claimed ─────────────────────────────────────────────
    AI_SKILL_KEYWORDS = set(_STRONG_AI_EVIDENCE + _MODERATE_AI_EVIDENCE)
    claimed_ai_skills = 0
    for skill in skills:
        skill_name = skill.get("name", "").lower()
        if any(kw in skill_name for kw in AI_SKILL_KEYWORDS):
            claimed_ai_skills += 1

    if claimed_ai_skills == 0:
        return 0.9  # No AI skills claimed at all — not a stuffer

    # ── 2. Measure AI evidence in career history ───────────────────────────────
    combined_desc = " ".join(
        role.get("description", "").lower() for role in career
    )
    career_evidence = sum(1 for kw in _STRONG_AI_EVIDENCE if kw in combined_desc)

    # ── 3. Check if current title is non-AI ───────────────────────────────────
    title_is_non_ai = any(t in current_title for t in _NON_AI_TITLES)

    # ── 4. Compute stuffing score ─────────────────────────────────────────────
    # Ratio of claims to evidence — high ratio = more likely stuffing
    evidence_ratio = career_evidence / max(1, claimed_ai_skills)

    if evidence_ratio >= 2.0:
        stuffing_risk = 0.0  # Excellent evidence-to-claim ratio
    elif evidence_ratio >= 1.0:
        stuffing_risk = 0.2
    elif evidence_ratio >= 0.5:
        stuffing_risk = 0.5
    elif evidence_ratio >= 0.2:
        stuffing_risk = 0.75
    else:
        stuffing_risk = 0.90  # Many AI claims, almost no career evidence

    # Non-AI title with many AI skills = strong stuffing signal
    if title_is_non_ai and claimed_ai_skills >= 4:
        stuffing_risk = min(1.0, stuffing_risk + 0.3)

    return max(0.0, 1.0 - stuffing_risk)


def _score_role_trajectory(candidate: dict[str, Any]) -> float:
    """Assess the career trajectory for IC (individual contributor) growth.

    We want candidates who stayed technical / grew as ICs.
    Penalise those who drifted entirely into management with no recent coding.
    """
    career: list[dict] = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").lower()
    yoe = profile.get("years_of_experience", 0)

    if not career:
        return 0.3

    MANAGEMENT_KEYWORDS = {
        "director", "vp ", "vice president", "cto", "ceo", "coo",
        "head of", "chief", "vp of",
    }
    SENIOR_IC_KEYWORDS = {
        "senior", "staff", "principal", "lead", "architect",
        "sr.", "sr ",
    }

    # Check recent roles (first 2 in list = most recent)
    recent_roles = career[:2]
    current_is_manager = any(kw in current_title for kw in MANAGEMENT_KEYWORDS)
    current_is_senior_ic = any(kw in current_title for kw in SENIOR_IC_KEYWORDS)
    current_is_ai = any(t in current_title for t in _AI_ROLE_TITLES)

    # Recent descriptions should contain coding evidence
    recent_desc = " ".join(r.get("description", "").lower() for r in recent_roles)
    has_recent_code_evidence = any(
        kw in recent_desc for kw in ["implemented", "built", "developed", "wrote", "designed",
                                      "deployed", "engineered", "optimized", "refactored"]
    )

    score = 0.5  # Base

    if current_is_manager and yoe > 8:
        score -= 0.3  # Likely drifted to management
    if current_is_senior_ic:
        score += 0.2
    if current_is_ai:
        score += 0.2
    if has_recent_code_evidence:
        score += 0.2
    if not has_recent_code_evidence and yoe > 5:
        score -= 0.2  # 5+ YOE with no coding language in recent roles

    return max(0.0, min(1.0, score))


def _score_education(candidate: dict[str, Any]) -> float:
    """Score educational background for prestige and relevance.

    Uses the tier field directly from the schema.
    Also rewards CS/EE/Math/Stats fields over unrelated fields.
    """
    education: list[dict] = candidate.get("education", [])
    if not education:
        return 0.3  # No education listed — neutral

    RELEVANT_FIELDS = {
        "computer science", "cs", "software engineering",
        "machine learning", "artificial intelligence",
        "data science", "statistics", "mathematics",
        "electronics", "electrical engineering",
        "information technology", "it",
        "computational", "physics",
    }

    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5,
                   "tier_4": 0.3, "unknown": 0.35}

    best_tier_score = 0.0
    best_field_score = 0.0

    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        t_score = tier_scores.get(tier, 0.35)
        best_tier_score = max(best_tier_score, t_score)

        field_relevant = any(f in field for f in RELEVANT_FIELDS)
        degree_bonus = 0.1 if "ph.d" in degree or "m.tech" in degree or "m.e" in degree else 0.0
        if field_relevant:
            best_field_score = max(best_field_score, 0.8 + degree_bonus)
        else:
            best_field_score = max(best_field_score, 0.3 + degree_bonus)

    return min(1.0, 0.6 * best_tier_score + 0.4 * best_field_score)
