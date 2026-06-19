"""
NEXUS — Neural Explainable Candidate Understanding System
Configuration constants used across all modules.
"""
from __future__ import annotations
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SKILL_TAXONOMY_PATH = DATA_DIR / "skill_taxonomy.json"

# ─── Scoring weights ──────────────────────────────────────────────────────────
# Final score = base_score × behavioral_modifier × honeypot_penalty
# base_score = CAREER_WEIGHT × career_score + SKILLS_WEIGHT × skills_score
#              + LOCATION_WEIGHT × location_score

CAREER_WEIGHT   = 0.45   # Career DNA (product bg, AI seniority, stuffing penalty)
SKILLS_WEIGHT   = 0.40   # Semantic skills fit (BM25 + taxonomy)
LOCATION_WEIGHT = 0.15   # Location + relocation fit

# Career DNA sub-weights (must sum to 1.0)
AI_SENIORITY_W      = 0.35
PRODUCT_RATIO_W     = 0.30
STUFFING_PENALTY_W  = 0.15   # Penalty applied
ROLE_TRAJECTORY_W   = 0.12
EDUCATION_W         = 0.08

# Skills sub-weights (must sum to 1.0)
TIER_A_W            = 0.55
TIER_B_W            = 0.25
BM25_W              = 0.20

# Behavioral sub-weights (must sum to 1.0)
BEHAVIORAL_AVAIL_W  = 0.30
BEHAVIORAL_ENGAGE_W = 0.25
BEHAVIORAL_TRUST_W  = 0.20
BEHAVIORAL_CAPAB_W  = 0.15
BEHAVIORAL_LOC_W    = 0.10

# ─── Pipeline thresholds ──────────────────────────────────────────────────────
# Tier 1 filter: hard disqualify below this score (applied to ~85% of candidates)
HARD_FILTER_MIN_AI_EVIDENCE  = 0.05   # Minimum AI evidence in career text

# BM25 pre-filter: keep top N before full scoring
BM25_TOP_N = 600

# Honeypot thresholds
HONEYPOT_DISQUALIFY_THRESHOLD = 0.65   # Above this → treated as honeypot, score floored
HONEYPOT_PENALIZE_THRESHOLD   = 0.35   # Above this → partial penalty

# ─── Behavioral signal ranges ─────────────────────────────────────────────────
# Days since last_active_date before candidate is penalized
ACTIVE_DECAY_START_DAYS  = 30    # Full score up to 30 days ago
ACTIVE_DECAY_ZERO_DAYS   = 180   # Zero score at 180+ days

# Notice period scoring
NOTICE_IDEAL_MAX_DAYS    = 30    # Sub-30 is premium
NOTICE_ACCEPTABLE_DAYS   = 60
NOTICE_MAX_DAYS          = 90    # 90+ days is a hiring friction signal

# ─── Geography ────────────────────────────────────────────────────────────────
PREFERRED_CITIES = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bombay", "bangalore", "bengaluru",
    "delhi ncr", "new delhi", "faridabad", "ghaziabad",
    "navi mumbai", "thane", "pune"
}
PREFERRED_COUNTRY = "india"

# ─── Services / consulting companies (penalise if 100% of career) ─────────────
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcl technologies", "tech mahindra",
    "ibm global services", "ibm consulting",
    "hexaware", "mphasis", "mindtree",
    "l&t infotech", "ltimindtree", "lti",
    "persistent systems", "coforge", "niit technologies",
    "zensar", "mastech", "syntel",
    "deloitte consulting", "kpmg consulting",
}

# ─── AI/ML career keywords for text scoring ───────────────────────────────────
# Used in BM25 query and career evidence extraction
JD_QUERY_TERMS = [
    "embedding retrieval vector search ranking recommendation",
    "faiss elasticsearch opensearch weaviate qdrant milvus pinecone",
    "sentence transformers dense retrieval hybrid search reranking",
    "production deployment deployed shipped a/b testing",
    "ndcg mrr map recall precision evaluation",
    "nlp information retrieval recommendation system",
    "lora qlora fine-tuning learning to rank",
    "python pytorch tensorflow machine learning",
]

# Phrases from JD that identify "AI-curious non-engineers" (negative signal)
AI_CURIOUS_PHRASES = [
    "curious about ai", "experimenting with chatgpt", "ai tools for productivity",
    "excited about ai", "exploring ai", "learning about ai",
    "ai for content creation", "using llm tools",
]

# ─── Evaluation settings ──────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 5000  # Subsample for local NDCG estimation
NDCG_K_VALUES    = [10, 50]
