"""
NEXUS — Main Pipeline Orchestrator
Ties all 6 layers together into a single, efficient ranking run.

Architecture (3-tier cascading):
  Tier 1 (Hard filter):   100K → ~15K candidates in ~10s
  Tier 2 (BM25 pre-rank): 15K  → top-600 candidates in ~20s
  Tier 3 (Full scoring):  600  → final top-100 in ~15s
  Total wall time target: < 60 seconds on 8-core CPU
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from nexus.config import BM25_TOP_N
from nexus.integrity import compute_honeypot_risk, honeypot_score_modifier
from nexus.jd_parser import is_hard_disqualified, compute_disqualifier_penalty
from nexus.career_dna import compute_career_dna_score
from nexus.semantic_scorer import (
    passes_hard_filter,
    compute_taxonomy_score,
    build_bm25_index,
    bm25_normalized_score,
    _load_taxonomy,
)
from nexus.behavioral import compute_behavioral_score
from nexus.ranker import compute_final_score, calibrate_scores, generate_reasoning


# ─── Streaming data loader ────────────────────────────────────────────────────

def stream_candidates(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    """Stream candidates one at a time from JSONL file (memory-efficient)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load all candidates into memory (use only for sample files)."""
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    else:
        return list(stream_candidates(path))


# ─── Tier 1: Hard filter ──────────────────────────────────────────────────────

def tier1_filter(candidates: list[dict], verbose: bool = True) -> list[dict]:
    """Filter to candidates with at least minimal AI relevance."""
    taxonomy = _load_taxonomy()
    passed = []
    for c in candidates:
        # Hard disqualified by JD rules → still include but mark
        disq, _ = is_hard_disqualified(c)
        if disq:
            continue  # Skip entirely — JD says these are not fits
        if passes_hard_filter(c, taxonomy):
            passed.append(c)
    if verbose:
        print(f"  Tier 1 filter: {len(candidates):,} -> {len(passed):,} candidates")
    return passed


# ─── Tier 2: BM25 pre-rank ────────────────────────────────────────────────────

def tier2_bm25_prerank(
    candidates: list[dict],
    top_n: int = BM25_TOP_N,
    verbose: bool = True,
) -> tuple[list[dict], list[float]]:
    """BM25 pre-rank to select top-N candidates for full scoring."""
    from nexus.config import JD_QUERY_TERMS
    query = " ".join(JD_QUERY_TERMS)

    idx = build_bm25_index(candidates)
    scores = idx.batch_score(query)

    # Pair and sort
    paired = sorted(zip(scores, candidates), key=lambda x: -x[0])
    top_paired = paired[:top_n]

    top_scores = [s for s, _ in top_paired]
    top_candidates = [c for _, c in top_paired]

    # Normalize BM25 scores to [0, 1]
    max_score = max(top_scores) if top_scores else 1.0
    normalized = [bm25_normalized_score(s, max_score) for s in top_scores]

    if verbose:
        print(f"  Tier 2 BM25:   {len(candidates):,} -> {len(top_candidates):,} candidates")
    return top_candidates, normalized


# ─── Tier 3: Full precision scoring ──────────────────────────────────────────

def tier3_full_score(
    candidates: list[dict],
    bm25_scores: list[float],
    verbose: bool = True,
) -> list[tuple[float, dict, dict]]:
    """Full multi-dimensional scoring on top-N candidates.

    Returns list of (final_score, candidate, score_breakdown) tuples.
    """
    taxonomy = _load_taxonomy()
    results = []

    for i, candidate in enumerate(candidates):
        bm25_norm = bm25_scores[i] if i < len(bm25_scores) else 0.0

        # Layer 0: Honeypot
        honeypot_risk = compute_honeypot_risk(candidate)
        hp_modifier   = honeypot_score_modifier(candidate)

        # Layer 1: Disqualifier penalty
        _, _ = is_hard_disqualified(candidate)  # Already filtered in tier1
        disq_penalty = compute_disqualifier_penalty(candidate)

        # Layer 2: Career DNA
        career_scores = compute_career_dna_score(candidate)

        # Layer 3: Semantic skills
        skills_scores = compute_taxonomy_score(candidate, taxonomy)

        # Layer 4: Behavioral
        behavioral_scores = compute_behavioral_score(candidate)

        # Layer 5: Final score
        final = compute_final_score(
            career_scores=career_scores,
            skills_scores=skills_scores,
            behavioral_scores=behavioral_scores,
            honeypot_modifier=hp_modifier,
            disqualifier_penalty=disq_penalty,
            bm25_normalized=bm25_norm,
        )

        # Layer 5: Reasoning
        reasoning = generate_reasoning(
            candidate=candidate,
            career_scores=career_scores,
            skills_scores=skills_scores,
            behavioral_scores=behavioral_scores,
            honeypot_risk=honeypot_risk,
            final_score=final,
        )

        breakdown = {
            "final_score":        final,
            "career_total":       career_scores["total"],
            "career_ai":          career_scores["ai_seniority"],
            "career_product":     career_scores["product_ratio"],
            "career_stuffing":    career_scores["stuffing_clean"],
            "skills_total":       skills_scores["total"],
            "skills_tier_a":      skills_scores["tier_a_score"],
            "skills_bm25":        bm25_norm,
            "behavioral_total":   behavioral_scores["total"],
            "behavioral_avail":   behavioral_scores["availability"],
            "behavioral_engage":  behavioral_scores["engagement"],
            "behavioral_trust":   behavioral_scores["trust"],
            "behavioral_capab":   behavioral_scores["capability"],
            "behavioral_loc":     behavioral_scores["location_fit"],
            "behavioral_modifier":behavioral_scores["modifier"],
            "honeypot_risk":      honeypot_risk,
            "matched_tier_a":     skills_scores.get("matched_tier_a", []),
            "reasoning":          reasoning,
        }

        results.append((final, candidate, breakdown))

    results.sort(key=lambda x: (-x[0], x[1].get("candidate_id", "ZZZZ")))

    if verbose:
        print(f"  Tier 3 scoring: {len(candidates):,} candidates fully scored")
    return results


# ─── CSV writer ──────────────────────────────────────────────────────────────

def write_submission_csv(
    ranked_results: list[tuple[float, dict, dict]],
    output_path: Path,
    top_n: int = 100,
) -> None:
    """Write the top-N results to submission CSV per spec."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (score, candidate, breakdown) in enumerate(ranked_results[:top_n], start=1):
            cid = candidate.get("candidate_id", "UNKNOWN")
            reasoning = breakdown.get("reasoning", "")
            # Escape commas in reasoning for CSV compatibility
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"  Submission CSV written to: {output_path}")


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> list[tuple[float, dict, dict]]:
    """Run the full NEXUS pipeline end-to-end.

    Args:
        input_path:  Path to candidates.jsonl (or sample_candidates.json)
        output_path: Path to write submission CSV
        verbose:     Print progress to stdout

    Returns:
        List of (score, candidate, breakdown) tuples, sorted by rank
    """
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  NEXUS - Intelligent Candidate Ranking Pipeline")
        print(f"{'='*60}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}\n")

    # ── Load candidates ──────────────────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print("[1/4] Loading candidates...")
    all_candidates = load_candidates(input_path)
    if verbose:
        print(f"  Loaded {len(all_candidates):,} candidates in {time.time()-t0:.1f}s")

    # ── Tier 1: Hard filter ──────────────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print("\n[2/4] Tier 1: Hard filter (AI relevance check)...")
    filtered = tier1_filter(all_candidates, verbose=verbose)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # ── Tier 2: BM25 pre-rank ────────────────────────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"\n[3/4] Tier 2: BM25 pre-ranking (top {BM25_TOP_N})...")
    top_candidates, bm25_scores = tier2_bm25_prerank(filtered, top_n=BM25_TOP_N, verbose=verbose)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # ── Tier 3: Full precision scoring ───────────────────────────────────────
    t0 = time.time()
    if verbose:
        print(f"\n[4/4] Tier 3: Full precision scoring...")
    results = tier3_full_score(top_candidates, bm25_scores, verbose=verbose)
    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    # ── Write output ─────────────────────────────────────────────────────────
    write_submission_csv(results, output_path)

    total_time = time.time() - start_time
    if verbose:
        print(f"\n{'='*60}")
        print(f"  [SUCCESS] Pipeline complete in {total_time:.1f}s")
        print(f"  Top candidate: {results[0][1].get('candidate_id')} "
              f"(score={results[0][0]:.4f})")
        print(f"  Top-10 scores: "
              + ", ".join(f"{r[0]:.3f}" for r in results[:10]))
        print(f"{'='*60}\n")

    return results
