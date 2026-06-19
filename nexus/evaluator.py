"""
NEXUS — Local NDCG Evaluation Harness
Estimates ranking quality locally WITHOUT access to ground truth.

Approach:
  Derives relevance annotations from the JD requirements programmatically.
  Relevance tiers:
    3 = Ideal: strong AI/ML product-company background, engaged
    2 = Good: some AI/ML evidence, mixed background
    1 = Marginal: AI-curious, domain expert
    0 = Irrelevant/Honeypot

This lets you tune score weights BEFORE submitting.
"""
from __future__ import annotations

import math
from typing import Any

from nexus.integrity import compute_honeypot_risk
from nexus.career_dna import compute_career_dna_score
from nexus.jd_parser import is_hard_disqualified


def estimate_relevance(candidate: dict[str, Any]) -> int:
    """Estimate relevance tier (0-3) for a candidate against the JD.

    Used only for LOCAL evaluation — not submitted.
    """
    # Tier 0: Honeypot or hard disqualified
    if compute_honeypot_risk(candidate) >= 0.6:
        return 0
    disqualified, _ = is_hard_disqualified(candidate)
    if disqualified:
        return 0

    dna = compute_career_dna_score(candidate)
    ai_score      = dna["ai_seniority"]
    product_score = dna["product_ratio"]
    stuffing_clean = dna["stuffing_clean"]

    # Tier 3: Ideal candidate
    if ai_score >= 0.65 and product_score >= 0.55 and stuffing_clean >= 0.7:
        return 3

    # Tier 2: Good candidate
    if ai_score >= 0.40 and (product_score >= 0.35 or stuffing_clean >= 0.5):
        return 2

    # Tier 1: Marginal
    if ai_score >= 0.15 or dna["total"] >= 0.25:
        return 1

    return 0


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # +2 because i is 0-indexed
    return dcg


def ndcg_at_k(ranked_candidates: list[dict], all_relevances: list[int], k: int) -> float:
    """Compute NDCG@k for a ranked list given pre-computed relevances."""
    if not ranked_candidates:
        return 0.0

    # Build ranked relevance list
    ranked_rels = all_relevances[:len(ranked_candidates)]

    # Ideal DCG: sorted relevances
    ideal_rels = sorted(ranked_rels, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0

    actual_dcg = dcg_at_k(ranked_rels, k)
    return actual_dcg / idcg


def evaluate_ranking(
    ranked_top100: list[dict[str, Any]],
    full_corpus: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Evaluate the top-100 ranking using local relevance estimation.

    Args:
        ranked_top100: Candidates in rank order (rank 1 first)
        full_corpus:   Optional full corpus for MAP calculation

    Returns dict with ndcg_10, ndcg_50, p_10, estimated_composite
    """
    # Estimate relevances for ranked candidates
    ranked_rels = [estimate_relevance(c) for c in ranked_top100]

    # NDCG@10
    ndcg10 = dcg_at_k(ranked_rels, 10) / max(
        1e-6, dcg_at_k(sorted(ranked_rels, reverse=True), 10)
    )

    # NDCG@50
    ndcg50 = dcg_at_k(ranked_rels, 50) / max(
        1e-6, dcg_at_k(sorted(ranked_rels, reverse=True), 50)
    )

    # P@10: fraction of top-10 that are tier 2+ ("relevant")
    p10 = sum(1 for r in ranked_rels[:10] if r >= 2) / 10.0

    # Honeypot rate in top 100
    honeypot_count = sum(
        1 for c in ranked_top100 if compute_honeypot_risk(c) >= 0.5
    )
    honeypot_rate = honeypot_count / max(1, len(ranked_top100))

    # Estimated composite (per spec formula)
    estimated_composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.05 * p10

    return {
        "ndcg_10":             round(ndcg10, 4),
        "ndcg_50":             round(ndcg50, 4),
        "p_10":                round(p10, 4),
        "honeypot_rate":       round(honeypot_rate, 4),
        "estimated_composite": round(estimated_composite, 4),
    }
