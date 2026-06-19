"""
NEXUS Layer 0 — INTEGRITY SHIELD
Detects honeypot candidates and profiles with impossible / inconsistent data.

Honeypot patterns used in this dataset:
  1. Temporal impossibility  — career duration exceeds claimed years_of_experience
  2. Skill-experience paradox — 'expert' proficiency with 0 duration_months
  3. Title-career mismatch    — current_title totally inconsistent with career history
  4. Cumulative overlap       — career entries have overlapping/impossible dates
  5. Statistical extremes     — suspiciously perfect assessment scores across all skills
  6. Skills inflation         — claimed 'expert' in 10+ skills simultaneously
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from nexus.config import HONEYPOT_DISQUALIFY_THRESHOLD, HONEYPOT_PENALIZE_THRESHOLD

# Reference date — treat as "today" for all temporal calculations
_TODAY = date(2026, 6, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_honeypot_risk(candidate: dict[str, Any]) -> float:
    """Return a honeypot risk score in [0, 1].

    0.0 = definitely a real candidate
    1.0 = almost certainly a honeypot / trap profile

    The score is a weighted average of six independent red-flag checks.
    Any single flag above its own threshold is enough to raise overall risk.
    """
    flags = [
        (0.25, _check_temporal_impossibility(candidate)),
        (0.25, _check_skill_experience_paradox(candidate)),
        (0.20, _check_cumulative_tenure_mismatch(candidate)),
        (0.15, _check_title_career_mismatch(candidate)),
        (0.10, _check_statistical_extremes(candidate)),
        (0.05, _check_skills_inflation(candidate)),
    ]
    total_weight = sum(w for w, _ in flags)
    weighted_sum = sum(w * s for w, s in flags)
    return min(1.0, weighted_sum / total_weight)


def is_honeypot(candidate: dict[str, Any]) -> bool:
    """Return True if candidate should be treated as a definite honeypot."""
    return compute_honeypot_risk(candidate) >= HONEYPOT_DISQUALIFY_THRESHOLD


def honeypot_score_modifier(candidate: dict[str, Any]) -> float:
    """Return a multiplier [0.05, 1.0] applied to the final candidate score.

    - Risk < penalize_threshold → multiplier = 1.0 (no penalty)
    - Risk in [penalize_threshold, disqualify_threshold] → linear penalty
    - Risk >= disqualify_threshold → multiplier = 0.05 (effectively eliminated)
    """
    risk = compute_honeypot_risk(candidate)
    if risk >= HONEYPOT_DISQUALIFY_THRESHOLD:
        return 0.05
    if risk >= HONEYPOT_PENALIZE_THRESHOLD:
        # Linear interpolation from 1.0 → 0.15 across the penalize band
        band = HONEYPOT_DISQUALIFY_THRESHOLD - HONEYPOT_PENALIZE_THRESHOLD
        t = (risk - HONEYPOT_PENALIZE_THRESHOLD) / band
        return max(0.15, 1.0 - 0.85 * t)
    return 1.0


def get_honeypot_reasons(candidate: dict[str, Any]) -> list[str]:
    """Return a human-readable list of reasons for honeypot risk (for demo UI)."""
    reasons = []
    if _check_temporal_impossibility(candidate) > 0.5:
        reasons.append("Temporal impossibility in career timeline")
    if _check_skill_experience_paradox(candidate) > 0.5:
        reasons.append("Expert proficiency with near-zero usage duration")
    if _check_cumulative_tenure_mismatch(candidate) > 0.5:
        reasons.append("Total career duration greatly exceeds stated experience")
    if _check_title_career_mismatch(candidate) > 0.5:
        reasons.append("Current title inconsistent with career history")
    if _check_statistical_extremes(candidate) > 0.5:
        reasons.append("Suspiciously perfect assessment / endorsement scores")
    if _check_skills_inflation(candidate) > 0.5:
        reasons.append("Implausible number of expert-level skills")
    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# Private checks
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _check_temporal_impossibility(candidate: dict[str, Any]) -> float:
    """Detect when career entries have impossible dates.

    Checks:
    - An end_date before start_date in the same role
    - A start_date in the future (after _TODAY)
    - Overlapping non-current roles that together imply more months
      than the candidate has been alive in the workforce (>600 months = 50 years)
    """
    career: list[dict] = candidate.get("career_history", [])
    score = 0.0
    flags = 0

    for role in career:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        if start is None:
            continue

        # Flag 1: end before start
        if end is not None and end < start:
            score += 0.8
            flags += 1

        # Flag 2: start date in the future
        if start > _TODAY:
            score += 0.6
            flags += 1

        # Flag 3: duration_months is much larger than date range implies
        claimed_duration = role.get("duration_months", 0)
        if start and (end or _TODAY):
            actual_end = end if end else _TODAY
            actual_months = max(0, (actual_end.year - start.year) * 12
                                + (actual_end.month - start.month))
            # Allow 2-month leeway for rounding
            if claimed_duration > actual_months + 2 and claimed_duration > 12:
                overshoot_ratio = (claimed_duration - actual_months) / max(1, claimed_duration)
                score += min(0.9, overshoot_ratio)
                flags += 1

    return min(1.0, score / max(1, flags)) if flags else 0.0


def _check_skill_experience_paradox(candidate: dict[str, Any]) -> float:
    """Flag skills where proficiency is 'expert' or 'advanced' but duration is 0.

    The honeypot spec says: 'expert proficiency in 10 skills with 0 years used'.
    """
    skills: list[dict] = candidate.get("skills", [])
    paradox_count = 0
    total_expert_advanced = 0

    for skill in skills:
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 1)  # 1 as safe default

        if proficiency in ("expert", "advanced"):
            total_expert_advanced += 1
            if duration == 0:
                paradox_count += 1
            elif proficiency == "expert" and duration < 3:
                paradox_count += 0.5

    if total_expert_advanced == 0:
        return 0.0
    return min(1.0, paradox_count / total_expert_advanced)


def _check_cumulative_tenure_mismatch(candidate: dict[str, Any]) -> float:
    """Check if total career months exceeds plausible working life.

    Also checks if claimed years_of_experience is wildly inconsistent with
    the sum of career_history duration_months.
    """
    profile = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])

    claimed_years = profile.get("years_of_experience", 0)
    claimed_months = claimed_years * 12

    total_career_months = sum(r.get("duration_months", 0) for r in career)

    if total_career_months == 0:
        return 0.0

    # Overshoot: sum of tenure >> claimed years_of_experience
    # Allow 20% overshoot for overlapping roles (common in real data)
    if claimed_months > 0:
        ratio = total_career_months / claimed_months
        if ratio > 2.5:  # 2.5× more months than claimed YOE
            return min(1.0, (ratio - 2.5) / 2.5)

    # Absolute cap: total career > 55 years of work history
    if total_career_months > 660:
        return min(1.0, (total_career_months - 660) / 240)

    return 0.0


def _check_title_career_mismatch(candidate: dict[str, Any]) -> float:
    """Detect when the current_title is completely inconsistent with career history.

    Example trap: current_title = 'ML Engineer' but every career entry is
    'HR Manager', 'Accountant', 'Graphic Designer', etc.
    """
    profile = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])
    current_title = profile.get("current_title", "").lower()

    if not current_title or not career:
        return 0.0

    TECH_KEYWORDS = {
        "engineer", "scientist", "developer", "architect",
        "ml", "ai", "data", "analytics", "research",
        "backend", "frontend", "fullstack", "devops", "platform",
    }
    NON_TECH_KEYWORDS = {
        "hr", "human resources", "recruiter",
        "accountant", "accounting", "finance",
        "graphic", "designer", "creative",
        "writer", "content", "editor",
        "sales", "marketing", "brand",
        "operations", "manager", "support",
        "mechanical", "civil", "electrical",
    }

    current_is_tech = any(kw in current_title for kw in TECH_KEYWORDS)
    if not current_is_tech:
        return 0.0  # Non-tech current title is expected — no anomaly

    # Current title is tech — check if ANY career history role is also tech
    tech_career_count = 0
    for role in career:
        title = role.get("title", "").lower()
        if any(kw in title for kw in TECH_KEYWORDS):
            tech_career_count += 1

    if not career:
        return 0.0

    tech_ratio = tech_career_count / len(career)
    if tech_ratio == 0 and current_is_tech:
        # Tech title now, but ZERO tech roles in history — very suspicious
        return 0.85
    elif tech_ratio < 0.2 and current_is_tech:
        return 0.5
    return 0.0


def _check_statistical_extremes(candidate: dict[str, Any]) -> float:
    """Flag candidates with suspiciously perfect statistical signals.

    Patterns:
    - All skill assessment scores are 95-100
    - Every skill has maximum endorsements
    - Impossible combination of signals (e.g., 100% interview_completion AND 100% offer_acceptance)
    """
    signals = candidate.get("redrob_signals", {})
    skills: list[dict] = candidate.get("skills", [])
    score = 0.0

    # Check skill assessment scores — are they suspiciously uniform/perfect?
    assessments: dict = signals.get("skill_assessment_scores", {})
    if len(assessments) >= 3:
        values = list(assessments.values())
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        # Perfect scores with zero variance is suspicious
        if mean_val >= 95 and variance < 4:
            score += 0.6
        elif mean_val >= 90 and variance < 2:
            score += 0.4

    # Check endorsements — are all endorsements the same value?
    if len(skills) >= 5:
        endorsements = [s.get("endorsements", 0) for s in skills]
        if len(set(endorsements)) == 1 and endorsements[0] > 0:
            # All identical endorsement counts — synthetic
            score += 0.5

    # Check interview + offer combination
    icr = signals.get("interview_completion_rate", 0)
    oar = signals.get("offer_acceptance_rate", -1)
    if icr == 1.0 and oar == 1.0:
        score += 0.3  # Perfect on both is unusual but not impossible

    return min(1.0, score)


def _check_skills_inflation(candidate: dict[str, Any]) -> float:
    """Detect implausible number of expert-level skills.

    In real profiles, having 5+ 'expert' skills is very unusual.
    Having 10+ is a strong synthetic data signal.
    """
    skills: list[dict] = candidate.get("skills", [])
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    advanced_count = sum(1 for s in skills if s.get("proficiency") == "advanced")

    if expert_count >= 10:
        return 0.9
    elif expert_count >= 7:
        return 0.7
    elif expert_count >= 5:
        return 0.4
    elif expert_count >= 3 and advanced_count >= 8:
        return 0.3
    return 0.0
