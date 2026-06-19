"""
NEXUS Layer 4 — BEHAVIORAL READINESS SCORER
Converts the 23 Redrob platform signals into a behavioral readiness composite.

The 23 signals are grouped into 5 sub-indices:
  1. Availability  — Is the candidate actually reachable right now?
  2. Engagement    — Are they actively looking?
  3. Trust         — Are their contact details verified, do they show up?
  4. Capability    — What do platform assessments and GitHub say?
  5. Fit           — Location, salary, work mode alignment

The composite acts as a MULTIPLIER on the base fit score:
  - Highly engaged (>0.75) → 1.2× bonus
  - Moderately engaged     → 1.0× neutral
  - Disengaged (0.3-0.5)  → 0.8× penalty
  - Ghost candidate (<0.3) → 0.5× strong penalty
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from nexus.config import (
    ACTIVE_DECAY_START_DAYS,
    ACTIVE_DECAY_ZERO_DAYS,
    NOTICE_IDEAL_MAX_DAYS,
    NOTICE_ACCEPTABLE_DAYS,
    NOTICE_MAX_DAYS,
    PREFERRED_CITIES,
    PREFERRED_COUNTRY,
    BEHAVIORAL_AVAIL_W,
    BEHAVIORAL_ENGAGE_W,
    BEHAVIORAL_TRUST_W,
    BEHAVIORAL_CAPAB_W,
    BEHAVIORAL_LOC_W,
)

_TODAY = date(2026, 6, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_behavioral_score(candidate: dict[str, Any]) -> dict[str, float]:
    """Compute full behavioral readiness score.

    Returns:
      total        - weighted composite [0, 1]
      availability - availability sub-index
      engagement   - engagement sub-index
      trust        - trust sub-index
      capability   - capability sub-index
      location_fit - location/fit sub-index
      modifier     - final multiplier to apply to base score (0.5–1.2)
    """
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    avail   = _score_availability(sig)
    engage  = _score_engagement(sig)
    trust   = _score_trust(sig)
    capab   = _score_capability(sig)
    loc_fit = _score_location_fit(profile, sig)

    total = (
        BEHAVIORAL_AVAIL_W  * avail
        + BEHAVIORAL_ENGAGE_W * engage
        + BEHAVIORAL_TRUST_W  * trust
        + BEHAVIORAL_CAPAB_W  * capab
        + BEHAVIORAL_LOC_W    * loc_fit
    )
    total = min(1.0, total)

    return {
        "total":        total,
        "availability": avail,
        "engagement":   engage,
        "trust":        trust,
        "capability":   capab,
        "location_fit": loc_fit,
        "modifier":     _compute_modifier(total),
    }


def _compute_modifier(behavioral_score: float) -> float:
    """Convert behavioral composite to a score multiplier."""
    if behavioral_score >= 0.75:
        return 1.20
    elif behavioral_score >= 0.60:
        return 1.08
    elif behavioral_score >= 0.45:
        return 1.00
    elif behavioral_score >= 0.30:
        return 0.82
    else:
        return 0.50  # Ghost candidate — penalise heavily


# ─────────────────────────────────────────────────────────────────────────────
# Sub-index scorers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _log_scale(x: float, max_val: float = 100.0) -> float:
    """Log-normalize x to [0, 1] with saturation at max_val."""
    if x <= 0:
        return 0.0
    return min(1.0, math.log1p(x) / math.log1p(max_val))


def _score_availability(sig: dict[str, Any]) -> float:
    """How available and responsive is this candidate right now?

    Key signals: open_to_work, last_active_date, notice_period, response_time
    """
    score = 0.0

    # Signal 1: open_to_work_flag (0.35 weight)
    open_to_work = sig.get("open_to_work_flag", False)
    score += 0.35 * (1.0 if open_to_work else 0.2)

    # Signal 2: recency of last active date (0.30 weight)
    last_active = _parse_date(sig.get("last_active_date"))
    if last_active:
        days_ago = (_TODAY - last_active).days
        if days_ago <= ACTIVE_DECAY_START_DAYS:
            recency = 1.0
        elif days_ago >= ACTIVE_DECAY_ZERO_DAYS:
            recency = 0.0
        else:
            # Exponential decay between start and zero
            t = (days_ago - ACTIVE_DECAY_START_DAYS) / (
                ACTIVE_DECAY_ZERO_DAYS - ACTIVE_DECAY_START_DAYS
            )
            recency = math.exp(-3 * t)  # Exponential decay
        score += 0.30 * recency
    else:
        score += 0.30 * 0.3  # Unknown — assume somewhat active

    # Signal 3: notice period (0.20 weight)
    notice_days = sig.get("notice_period_days", 60)
    if notice_days <= NOTICE_IDEAL_MAX_DAYS:
        notice_score = 1.0
    elif notice_days <= NOTICE_ACCEPTABLE_DAYS:
        notice_score = 0.75
    elif notice_days <= NOTICE_MAX_DAYS:
        notice_score = 0.50
    else:
        notice_score = max(0.1, 1.0 - (notice_days - NOTICE_MAX_DAYS) / 90.0)
    score += 0.20 * notice_score

    # Signal 4: avg_response_time (0.15 weight) — lower is better
    resp_time = sig.get("avg_response_time_hours", 48)
    if resp_time <= 4:
        resp_score = 1.0
    elif resp_time <= 24:
        resp_score = 0.85
    elif resp_time <= 72:
        resp_score = 0.6
    elif resp_time <= 168:  # 1 week
        resp_score = 0.35
    else:
        resp_score = max(0.0, 1.0 - resp_time / 336)  # 0 at 2 weeks
    score += 0.15 * resp_score

    return min(1.0, score)


def _score_engagement(sig: dict[str, Any]) -> float:
    """How actively is the candidate engaging with the platform?"""
    score = 0.0

    # Signal 5: recruiter_response_rate (0.40 weight) — key hiring predictor
    rrr = sig.get("recruiter_response_rate", 0.0)
    score += 0.40 * rrr  # Already in [0, 1]

    # Signal 6: applications_submitted_30d (0.25 weight)
    apps = sig.get("applications_submitted_30d", 0)
    score += 0.25 * _log_scale(apps, max_val=15)

    # Signal 7: profile_views_received_30d (0.20 weight)
    views = sig.get("profile_views_received_30d", 0)
    score += 0.20 * _log_scale(views, max_val=50)

    # Signal 8: saved_by_recruiters_30d (0.15 weight)
    saved = sig.get("saved_by_recruiters_30d", 0)
    score += 0.15 * _log_scale(saved, max_val=20)

    return min(1.0, score)


def _score_trust(sig: dict[str, Any]) -> float:
    """How trustworthy and reliable is this candidate?"""
    score = 0.0

    # Signal 9: verified_email (0.30 weight)
    score += 0.30 * (1.0 if sig.get("verified_email") else 0.0)

    # Signal 10: verified_phone (0.25 weight)
    score += 0.25 * (1.0 if sig.get("verified_phone") else 0.0)

    # Signal 11: interview_completion_rate (0.30 weight) — shows up when they say they will
    icr = sig.get("interview_completion_rate", 0.5)
    score += 0.30 * icr

    # Signal 12: linkedin_connected (0.10 weight)
    score += 0.10 * (1.0 if sig.get("linkedin_connected") else 0.0)

    # Signal 13: offer_acceptance_rate (0.05 weight) — -1 means no prior offers
    oar = sig.get("offer_acceptance_rate", -1)
    if oar >= 0:
        score += 0.05 * oar
    else:
        score += 0.05 * 0.5  # Neutral for no-offer-history

    return min(1.0, score)


def _score_capability(sig: dict[str, Any]) -> float:
    """Platform-measured capability signals."""
    score = 0.0

    # Signal 14: profile_completeness_score (0.35 weight)
    completeness = sig.get("profile_completeness_score", 0)
    score += 0.35 * (completeness / 100.0)

    # Signal 15: github_activity_score (0.40 weight) — -1 means not linked
    gh = sig.get("github_activity_score", -1)
    if gh == -1:
        gh_score = 0.0  # No GitHub linked — treat as 0, not penalise unfairly
    else:
        gh_score = gh / 100.0
    score += 0.40 * gh_score

    # Signal 16: skill_assessment_scores (0.25 weight)
    assessments: dict = sig.get("skill_assessment_scores", {})
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        score += 0.25 * (avg_assessment / 100.0)
    else:
        score += 0.25 * 0.3  # No assessments — slight penalty

    return min(1.0, score)


def _score_location_fit(profile: dict[str, Any], sig: dict[str, Any]) -> float:
    """Score location and work-mode fit against the JD requirements."""
    location = profile.get("location", "").lower()
    country  = profile.get("country", "").lower()
    willing_to_relocate = sig.get("willing_to_relocate", False)
    work_mode = sig.get("preferred_work_mode", "flexible")

    # Location scoring
    in_preferred_city    = any(city in location for city in PREFERRED_CITIES)
    in_preferred_country = (country == PREFERRED_COUNTRY)

    if in_preferred_city:
        location_score = 1.0
    elif in_preferred_country and willing_to_relocate:
        location_score = 0.85
    elif in_preferred_country:
        location_score = 0.70
    elif willing_to_relocate:
        location_score = 0.50
    else:
        location_score = 0.25

    # Work mode scoring — JD mentions offices but is flexible
    MODE_SCORES = {"flexible": 1.0, "hybrid": 0.9, "remote": 0.7, "onsite": 0.85}
    mode_score = MODE_SCORES.get(work_mode, 0.7)

    return min(1.0, 0.70 * location_score + 0.30 * mode_score)
