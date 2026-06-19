"""
NEXUS Layer 5 — ENSEMBLE RANKER + EXPLAINER
Combines all sub-scores into a final ranking with natural-language reasoning.
"""
from __future__ import annotations

from typing import Any

from nexus.config import (
    CAREER_WEIGHT,
    SKILLS_WEIGHT,
    LOCATION_WEIGHT,
    BM25_W,
)


# ─────────────────────────────────────────────────────────────────────────────
# Final score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(
    career_scores: dict[str, float],
    skills_scores: dict[str, float],
    behavioral_scores: dict[str, float],
    honeypot_modifier: float,
    disqualifier_penalty: float,
    bm25_normalized: float = 0.0,
) -> float:
    """Compute the final candidate score in [0, 1].

    Formula:
      base   = CAREER_W × career + SKILLS_W × skills_with_bm25 + LOC_W × location
      final  = base × behavioral_modifier × honeypot_modifier × disqualifier_penalty
    """
    # Skills score includes BM25 component
    skills_total = (
        (1 - BM25_W) * skills_scores["total"]
        + BM25_W * bm25_normalized
    )
    skills_total = min(1.0, skills_total)

    base = (
        CAREER_WEIGHT  * career_scores["total"]
        + SKILLS_WEIGHT * skills_total
        + LOCATION_WEIGHT * behavioral_scores["location_fit"]
    )

    final = base * behavioral_scores["modifier"] * honeypot_modifier * disqualifier_penalty
    return max(0.0, min(1.0, final))


# ─────────────────────────────────────────────────────────────────────────────
# Score calibration (ensures non-increasing ranks)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_scores(ranked_candidates: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Ensure scores are strictly non-increasing and add tie-break.

    If two candidates have the same score, the one with the lower
    candidate_id (lexicographically) gets a tiny bonus to ensure
    consistent tie-breaking per spec.
    """
    if not ranked_candidates:
        return []

    # Sort by score desc, then candidate_id asc for ties
    ranked_candidates.sort(
        key=lambda x: (-x[0], x[1].get("candidate_id", "ZZZZ"))
    )

    # Ensure non-increasing: if score[i] > score[i-1], cap it
    calibrated = []
    prev_score = ranked_candidates[0][0]
    for score, cand in ranked_candidates:
        if score > prev_score:
            score = prev_score
        calibrated.append((score, cand))
        prev_score = score

    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(
    candidate: dict[str, Any],
    career_scores: dict[str, float],
    skills_scores: dict[str, float],
    behavioral_scores: dict[str, float],
    honeypot_risk: float,
    final_score: float,
) -> str:
    """Generate a factual, 1–2 sentence reasoning string for the submission CSV.

    Format mirrors what a senior recruiter would write:
    '[Role] with [YOE] yrs; [AI-evidence summary]; [behavioral summary].'
    """
    profile = candidate.get("profile", {})
    sig     = candidate.get("redrob_signals", {})
    career  = candidate.get("career_history", [])

    title = profile.get("current_title", "Professional")
    yoe   = profile.get("years_of_experience", 0)

    parts: list[str] = []

    # ── Part 1: Role + experience ──────────────────────────────────────────────
    parts.append(f"{title} with {yoe:.1f} yrs experience")

    # ── Part 2: AI/ML evidence ─────────────────────────────────────────────────
    matched_a = skills_scores.get("matched_tier_a", [])
    if matched_a:
        top3 = ", ".join(matched_a[:3])
        parts.append(f"core retrieval/ML skills: {top3}")

    # Career DNA highlights
    if career_scores["ai_seniority"] >= 0.7:
        parts.append("strong AI/ML production background")
    if career_scores["product_ratio"] >= 0.7:
        parts.append("primarily at product companies")
    if career_scores["stuffing_clean"] >= 0.85:
        parts.append("skills corroborated by career evidence")

    # ── Part 3: Behavioral summary ─────────────────────────────────────────────
    beh_parts = []
    if sig.get("open_to_work_flag"):
        beh_parts.append("actively job-seeking")
    rrr = sig.get("recruiter_response_rate", 0)
    if rrr >= 0.6:
        beh_parts.append(f"{rrr:.0%} recruiter response rate")
    elif rrr < 0.2 and rrr > 0:
        beh_parts.append(f"low response rate ({rrr:.0%})")
    gh = sig.get("github_activity_score", -1)
    if gh >= 50:
        beh_parts.append(f"GitHub active ({gh:.0f}/100)")
    notice = sig.get("notice_period_days", 60)
    if notice <= 30:
        beh_parts.append(f"{notice}d notice")

    if beh_parts:
        parts.append("; ".join(beh_parts[:2]))

    # ── Part 4: Honeypot note (for transparency in reasoning) ──────────────────
    if honeypot_risk < 0.2:
        parts.append("profile integrity verified")

    # Compose into 1-2 sentences
    sentence = "; ".join(parts[:4]) + "."
    # Truncate to reasonable length
    if len(sentence) > 200:
        sentence = sentence[:197] + "..."
    return sentence
