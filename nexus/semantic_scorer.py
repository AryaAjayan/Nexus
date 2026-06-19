"""
NEXUS Layer 3 — SEMANTIC FIT SCORER
CPU-optimized 3-tier cascading scorer:
  Tier 1: Hard filter (eliminates ~85% of candidates)
  Tier 2: BM25 pre-ranking over career text (narrows to top-N)
  Tier 3: Full precision scoring with skill taxonomy

Designed to run 100K candidates in < 60 seconds on 8-core CPU.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from nexus.config import (
    SKILL_TAXONOMY_PATH,
    JD_QUERY_TERMS,
    TIER_A_W,
    TIER_B_W,
    BM25_W,
    HARD_FILTER_MIN_AI_EVIDENCE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load skill taxonomy
# ─────────────────────────────────────────────────────────────────────────────

_TAXONOMY: dict = {}

def _load_taxonomy() -> dict:
    global _TAXONOMY
    if not _TAXONOMY:
        with open(SKILL_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            _TAXONOMY = json.load(f)
    return _TAXONOMY


# ─────────────────────────────────────────────────────────────────────────────
# BM25 implementation (pure Python, no external deps beyond math)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Index:
    """Lightweight BM25 index for scoring candidate documents against a query.

    Implements BM25+ (lower-bounded BM25) to handle zero-term-frequency docs.
    k1=1.5, b=0.75 are standard Lucene defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 0.5):
        self.k1    = k1
        self.b     = b
        self.delta = delta
        self.docs: list[list[str]] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.df: dict[str, int] = {}  # document frequency
        self.avgdl: float = 0.0
        self.N: int = 0

    def add_documents(self, texts: list[str]) -> None:
        """Build index from a list of raw text strings."""
        for text in texts:
            tokens = _tokenize(text)
            self.docs.append(tokens)
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freqs.append(freq)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25+ score of query against document at doc_idx."""
        tokens = _tokenize(query)
        doc_freq = self.doc_freqs[doc_idx]
        doc_len  = len(self.docs[doc_idx])
        score = 0.0

        for term in tokens:
            if term not in self.df:
                continue
            tf   = doc_freq.get(term, 0)
            df_t = self.df[term]
            idf  = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1)
            numerator   = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avgdl))
            score += idf * (numerator / denominator + self.delta)

        return score

    def batch_score(self, query: str) -> list[float]:
        """Score all documents against query. Returns list of scores."""
        return [self.score(query, i) for i in range(self.N)]


# ─────────────────────────────────────────────────────────────────────────────
# Skill taxonomy matching
# ─────────────────────────────────────────────────────────────────────────────

def _get_candidate_skill_set(candidate: dict[str, Any]) -> set[str]:
    """Return the set of skill names (lowercase) from skills[] array."""
    return {
        s.get("name", "").lower().strip()
        for s in candidate.get("skills", [])
    }


def _get_candidate_skill_text(candidate: dict[str, Any]) -> str:
    """Combine all textual evidence from a candidate for BM25 indexing."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("current_title", ""))

    for role in candidate.get("career_history", []):
        parts.append(role.get("description", ""))
        parts.append(role.get("title", ""))
        parts.append(role.get("company", ""))

    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", ""))

    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " ".join(parts)


def compute_taxonomy_score(
    candidate: dict[str, Any],
    taxonomy: dict | None = None,
) -> dict[str, float]:
    """Score candidate skills against the tiered taxonomy.

    Returns:
      total         - weighted skill taxonomy score [0, 1]
      tier_a_score  - fraction of tier-A skills matched [0, 1]
      tier_b_score  - fraction of tier-B skills matched [0, 1]
      matched_tier_a - list of matched tier-A skills (for reasoning)
    """
    if taxonomy is None:
        taxonomy = _load_taxonomy()

    skill_set = _get_candidate_skill_set(candidate)

    # Build lookup from all career text too (skills evidenced in descriptions)
    career_text = _get_candidate_skill_text(candidate).lower()

    tier_a_skills = taxonomy.get("tier_a", {}).get("skills", [])
    tier_b_skills = taxonomy.get("tier_b", {}).get("skills", [])

    def _match_skill(kw: str, skill_set: set, career_text: str) -> float:
        """Return match strength [0, 1] for a keyword against candidate."""
        kw_lower = kw.lower()
        # Direct skill tag match (strongest signal)
        if kw_lower in skill_set:
            return 1.0
        # Partial skill tag match
        if any(kw_lower in s for s in skill_set):
            return 0.85
        # Career text mention
        if kw_lower in career_text:
            return 0.5
        return 0.0

    tier_a_matches = [
        (kw, _match_skill(kw, skill_set, career_text))
        for kw in tier_a_skills
    ]
    tier_b_matches = [
        (kw, _match_skill(kw, skill_set, career_text))
        for kw in tier_b_skills
    ]

    tier_a_score = sum(v for _, v in tier_a_matches) / max(1, len(tier_a_skills))
    tier_b_score = sum(v for _, v in tier_b_matches) / max(1, len(tier_b_skills))

    # Scale up — having ANY tier-A match is significant
    tier_a_score = min(1.0, tier_a_score * 3.5)  # ~30% match → full score
    tier_b_score = min(1.0, tier_b_score * 5.0)  # ~20% match → full score

    matched_tier_a = [kw for kw, v in tier_a_matches if v >= 0.5]

    total = TIER_A_W * tier_a_score + TIER_B_W * tier_b_score
    # BM25 component added externally in pipeline

    return {
        "total":          min(1.0, total),
        "tier_a_score":   tier_a_score,
        "tier_b_score":   tier_b_score,
        "matched_tier_a": matched_tier_a,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hard filter (Tier-1 — eliminates bulk of irrelevant candidates)
# ─────────────────────────────────────────────────────────────────────────────

def passes_hard_filter(candidate: dict[str, Any], taxonomy: dict | None = None) -> bool:
    """Return True if candidate passes the minimum AI-evidence bar.

    ~85% of candidates (HR managers, accountants, etc.) fail this and are
    excluded from expensive BM25 scoring.
    """
    if taxonomy is None:
        taxonomy = _load_taxonomy()

    skill_set = _get_candidate_skill_set(candidate)
    career_text = _get_candidate_skill_text(candidate).lower()

    # Must have at least 1 tier-A skill OR 2 tier-B skills
    tier_a_hits = sum(
        1 for kw in taxonomy["tier_a"]["skills"]
        if kw.lower() in skill_set or kw.lower() in career_text
    )
    tier_b_hits = sum(
        1 for kw in taxonomy["tier_b"]["skills"]
        if kw.lower() in skill_set or kw.lower() in career_text
    )

    # Also fast-pass anyone with relevant title
    title = candidate.get("profile", {}).get("current_title", "").lower()
    FAST_PASS_TITLES = ["ml engineer", "ai engineer", "data scientist",
                        "machine learning", "nlp", "search engineer"]
    has_relevant_title = any(t in title for t in FAST_PASS_TITLES)

    return tier_a_hits >= 1 or tier_b_hits >= 2 or has_relevant_title


# ─────────────────────────────────────────────────────────────────────────────
# BM25 index builder (for full corpus)
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_index(candidates: list[dict[str, Any]]) -> BM25Index:
    """Build a BM25 index over the candidate corpus."""
    texts = [_get_candidate_skill_text(c) for c in candidates]
    idx = BM25Index()
    idx.add_documents(texts)
    return idx


def bm25_normalized_score(raw_score: float, max_score: float) -> float:
    """Normalize BM25 score to [0, 1]."""
    if max_score <= 0:
        return 0.0
    return min(1.0, raw_score / max_score)
