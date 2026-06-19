# NEXUS — Neural Explainable Candidate Understanding System

> **Redrob Intelligent Candidate Discovery & Ranking Challenge**  
> Rank the top 100 of 100,000 candidates for a Senior AI/ML Engineer role.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ranker (produces submission.csv)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. Run the demo
streamlit run demo/app.py
```

**Reproduce command:**
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## 🧠 Architecture — 6-Layer Pipeline

```
Layer 0  INTEGRITY SHIELD       Honeypot + anomaly detection
Layer 1  JD INTELLIGENCE        Machine-readable JD scoring contract
Layer 2  CAREER DNA             Product vs services, AI seniority, keyword-stuffing detection
Layer 3  SEMANTIC SCORER        BM25+ cascade: 100K → 15K → 600 → 100
Layer 4  BEHAVIORAL SCORER      23 Redrob signals → 5 composite indices → multiplier
Layer 5  ENSEMBLE RANKER        Weighted fusion + reasoning generation
```

### 3-Tier Cascade (Why It's Fast)

| Tier | Input | Output | Method | Time |
|------|-------|--------|--------|------|
| Hard filter | 100,000 | ~15,000 | AI-evidence rule check | ~10s |
| BM25 pre-rank | ~15,000 | 600 | Custom BM25+ over career text | ~20s |
| Full scoring | 600 | 100 | All 6 layers, full precision | ~15s |
| **Total** | | | | **< 60s** |

---

## 📊 Scoring Formula

```
base_score  = 0.45 × career_DNA + 0.40 × skills_fit + 0.15 × location_fit
final_score = base_score × behavioral_modifier × honeypot_penalty × disqualifier_penalty
```

**Career DNA sub-scores:**
- `0.35` × AI/ML seniority (from career descriptions, not skill tags)
- `0.30` × Product company ratio (penalises consulting-only careers)
- `0.15` × Keyword-stuffing cleanliness (skill claims vs. career evidence)
- `0.12` × Role trajectory (IC growth vs. management drift)
- `0.08` × Education prestige (tier + field relevance)

**Behavioral modifier:** `0.50×` (ghost) → `1.00×` (neutral) → `1.20×` (highly engaged)

---

## 🛡️ Honeypot Detection

NEXUS detects 6 types of impossible/synthetic profiles:

| Check | Example Trap |
|-------|-------------|
| Temporal impossibility | End date before start date |
| Skill-experience paradox | "Expert" proficiency with 0 months usage |
| Cumulative tenure mismatch | 30 years career in 10 years |
| Title-career mismatch | "ML Engineer" with zero tech career history |
| Statistical extremes | All assessment scores = 100 ± 0 |
| Skills inflation | 10+ "expert"-level skills simultaneously |

Candidates scoring ≥ 0.65 honeypot risk are penalised to near-zero score.

---

## 🔧 Compute Environment

```
Platform: [Your platform]
CPU:      8 cores
RAM:      16 GB
Python:   3.11
GPU:      ❌ Not used
Network:  ❌ Not used during ranking
Pre-compute: ❌ Not required (BM25 builds at runtime in ~20s)
```

---

## 📁 Project Structure

```
nexus-ranker/
├── rank.py                   # ← CLI entrypoint (reproduce command)
├── requirements.txt
├── submission_metadata.yaml
├── nexus/
│   ├── config.py             # All constants & weights
│   ├── integrity.py          # Layer 0: Honeypot detection
│   ├── jd_parser.py          # Layer 1: JD intelligence
│   ├── career_dna.py         # Layer 2: Career DNA extractor
│   ├── semantic_scorer.py    # Layer 3: BM25 + skill taxonomy
│   ├── behavioral.py         # Layer 4: 23-signal behavioral scorer
│   ├── ranker.py             # Layer 5: Ensemble + reasoning
│   ├── evaluator.py          # Local NDCG harness
│   └── pipeline.py           # Orchestration
├── demo/
│   └── app.py                # Streamlit sandbox
└── data/
    └── skill_taxonomy.json   # Curated AI/ML skill taxonomy
```

---

## 📈 Local Evaluation

Run with `--eval` to see estimated NDCG scores:

```bash
python rank.py --candidates ./sample_candidates.json --out ./test.csv --eval
```

Output:
```
  Local Evaluation Metrics (estimated, not ground truth):
    NDCG@10:              0.87XX
    NDCG@50:              0.81XX
    P@10:                 0.90XX
    Honeypot rate:        0.0000
    Estimated composite:  0.84XX
```

---

## 🎯 Key Design Decisions

**Why BM25 over sentence-transformers?**
Sentence-transformers encoding 100K candidates on CPU ≈ 8-12 minutes — outside the 5-minute constraint. Our 3-tier cascade (hard filter → BM25 → full score) runs in <60 seconds.

**Why 'career DNA' instead of skills matching?**
The JD explicitly says: "A candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit, no matter how perfect their skill list looks." Career history analysis catches this; skills lists alone do not.

**Why a behavioral multiplier instead of additive?**
A perfect-on-paper candidate who hasn't logged in for 6 months is not actually available. Behavioral signals as a multiplier correctly capture this: zero availability → near-zero final score regardless of fit.
