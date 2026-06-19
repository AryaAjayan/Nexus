"""
NEXUS — Interactive Streamlit Demo
A stunning, production-quality sandbox for the Redrob hackathon.

Run: streamlit run demo/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS | Intelligent Candidate Ranker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Dark glassmorphism theme */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1626 50%, #0a1020 100%);
    min-height: 100vh;
}

/* Header */
.nexus-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
    border-radius: 20px;
    border: 1px solid rgba(99,102,241,0.25);
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
}
.nexus-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.nexus-subtitle {
    color: rgba(148,163,184,0.9);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Metric cards */
.metric-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(20px);
    text-align: center;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: rgba(148,163,184,0.8);
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* Candidate card */
.candidate-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(20px);
    transition: all 0.25s ease;
    cursor: pointer;
}
.candidate-card:hover {
    border-color: rgba(129,140,248,0.5);
    box-shadow: 0 4px 24px rgba(99,102,241,0.15);
    transform: translateX(4px);
}
.candidate-card.honeypot {
    border-color: rgba(239,68,68,0.4);
    background: rgba(30, 10, 10, 0.85);
}

/* Rank badge */
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border-radius: 10px;
    padding: 0.2rem 0.7rem;
    font-size: 0.9rem;
    font-weight: 700;
    min-width: 3rem;
    text-align: center;
}
.rank-badge.gold   { background: linear-gradient(135deg, #f59e0b, #d97706); }
.rank-badge.silver { background: linear-gradient(135deg, #94a3b8, #64748b); }
.rank-badge.bronze { background: linear-gradient(135deg, #d97706, #b45309); }

/* Score bar */
.score-bar-bg {
    background: rgba(30,41,59,0.8);
    border-radius: 6px;
    height: 8px;
    margin: 4px 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}

/* Tag pills */
.skill-tag {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem 0.15rem;
}
.skill-tag.honeypot-tag {
    background: rgba(239,68,68,0.15);
    border-color: rgba(239,68,68,0.4);
    color: #fca5a5;
}

/* Section headers */
.section-header {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 600;
    border-bottom: 1px solid rgba(99,102,241,0.25);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* Alert boxes */
.alert-success {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #6ee7b7;
    margin: 0.5rem 0;
}
.alert-warning {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #fcd34d;
    margin: 0.5rem 0;
}
.alert-danger {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #fca5a5;
    margin: 0.5rem 0;
}

/* Override Streamlit default styling */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}

div[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}

.stFileUploader {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def ensure_nexus_on_path():
    """Add project root to sys.path so nexus package is importable."""
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


ensure_nexus_on_path()


def load_uploaded_candidates(uploaded_file) -> list[dict]:
    """Parse uploaded JSON or JSONL file."""
    content = uploaded_file.read().decode("utf-8")
    candidates = []
    # Try JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = [data]
    except json.JSONDecodeError:
        # Try JSONL
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return candidates


def color_for_score(score: float) -> str:
    if score >= 0.75: return "#10b981"
    if score >= 0.55: return "#818cf8"
    if score >= 0.35: return "#f59e0b"
    return "#ef4444"


def score_bar_html(score: float, label: str, color: str = "#818cf8") -> str:
    pct = int(score * 100)
    return f"""
    <div style="margin: 4px 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:rgba(148,163,184,0.8);">
            <span>{label}</span>
            <span style="color:{color}; font-weight:600;">{pct}%</span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background: linear-gradient(90deg, {color}88, {color});"></div>
        </div>
    </div>
    """


def rank_badge_html(rank: int) -> str:
    cls = "gold" if rank == 1 else "silver" if rank == 2 else "bronze" if rank == 3 else ""
    emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank <= 3 else ""
    return f'<span class="rank-badge {cls}">{emoji} #{rank}</span>'


# ─── Main UI ─────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="nexus-header">
        <h1 class="nexus-title">🧠 NEXUS</h1>
        <p class="nexus-subtitle">Neural Explainable Candidate Understanding System &nbsp;·&nbsp;
        Redrob Intelligent Candidate Discovery Challenge</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "📁 Upload Candidates",
            type=["json", "jsonl"],
            help="Upload candidates.jsonl or sample_candidates.json",
        )

        st.markdown("---")
        st.markdown("### 🎛️ Pipeline Settings")
        top_n_display = st.slider("Candidates to display", 10, 100, 50, 5)
        show_honeypots = st.checkbox("Show flagged honeypots", value=True)
        run_eval = st.checkbox("Show local NDCG estimate", value=True)

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **NEXUS** uses a 6-layer pipeline:
        - 🛡️ Integrity Shield (honeypot detection)
        - 🔍 JD Intelligence Parser
        - 🧬 Career DNA Extractor
        - 📊 Semantic Fit Scorer (BM25)
        - 📡 Behavioral Signal Scorer
        - 🏆 Ensemble Ranker + Explainer
        """)

    # ── State ─────────────────────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.session_state.results = None
    if "candidates_loaded" not in st.session_state:
        st.session_state.candidates_loaded = 0

    # ── Load / Run Pipeline ───────────────────────────────────────────────────
    col_load, col_sample = st.columns([2, 1])
    with col_load:
        run_button = st.button(
            "🚀 Run NEXUS Ranking Pipeline",
            use_container_width=True,
            disabled=(uploaded_file is None and st.session_state.results is None),
        )

    with col_sample:
        sample_button = st.button("📋 Load Sample Data", use_container_width=True)

    if sample_button:
        # Load sample_candidates.json from the dataset
        sample_paths = [
            Path("../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"),
            Path("sample_candidates.json"),
        ]
        sample_data = None
        for p in sample_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                break

        if sample_data:
            st.session_state.sample_candidates = sample_data
            st.success(f"✅ Loaded {len(sample_data)} sample candidates")
        else:
            st.warning("Sample file not found. Please upload candidates manually.")

    if run_button or (uploaded_file and "results" not in st.session_state):
        candidates = []
        if uploaded_file:
            with st.spinner("Loading candidates..."):
                candidates = load_uploaded_candidates(uploaded_file)
        elif hasattr(st.session_state, "sample_candidates"):
            candidates = st.session_state.sample_candidates

        if candidates:
            st.session_state.candidates_loaded = len(candidates)
            with st.spinner(f"🧠 NEXUS is ranking {len(candidates):,} candidates..."):
                try:
                    from nexus.pipeline import (
                        tier1_filter, tier2_bm25_prerank, tier3_full_score
                    )
                    t_start = time.time()
                    filtered = tier1_filter(candidates, verbose=False)
                    top_cands, bm25_scores = tier2_bm25_prerank(filtered, verbose=False)
                    results = tier3_full_score(top_cands, bm25_scores, verbose=False)
                    elapsed = time.time() - t_start
                    st.session_state.results = results
                    st.session_state.elapsed = elapsed
                    st.success(f"✅ Ranked {len(candidates):,} candidates in {elapsed:.1f}s")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.results:
        results = st.session_state.results

        # ── Summary metrics ────────────────────────────────────────────────────
        st.markdown("---")
        top100 = results[:100]
        honeypot_flagged = sum(1 for _, _, bd in top100 if bd["honeypot_risk"] >= 0.4)
        avg_score = sum(s for s, _, _ in top100) / max(1, len(top100))
        top_score = results[0][0] if results else 0

        # Local NDCG if requested
        ndcg10 = ndcg50 = None
        if run_eval:
            try:
                from nexus.evaluator import evaluate_ranking
                metrics = evaluate_ranking([c for _, c, _ in top100])
                ndcg10 = metrics["ndcg_10"]
                ndcg50 = metrics["ndcg_50"]
            except Exception:
                pass

        cols = st.columns(5)
        metric_data = [
            ("Total Ranked", f"{len(results):,}", "candidates scored"),
            ("Top Score", f"{top_score:.3f}", "best fit score"),
            ("Avg Top-100", f"{avg_score:.3f}", "mean score"),
            ("Honeypots 🚨", str(honeypot_flagged), "flagged in top-100"),
        ]
        if ndcg10 is not None:
            metric_data.append(("Est. NDCG@10", f"{ndcg10:.3f}", "local estimate"))
        else:
            metric_data.append(("Elapsed", f"{st.session_state.get('elapsed', 0):.1f}s", "runtime"))

        for i, (label, value, sublabel) in enumerate(metric_data):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div style="color:rgba(100,116,139,0.7);font-size:0.72rem;">{sublabel}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Top Candidates",
            "📊 Score Analytics",
            "🚨 Honeypot Radar",
            "📥 Export CSV",
        ])

        with tab1:
            st.markdown(f"<div class='section-header'>Top {min(top_n_display, len(results))} Candidates by NEXUS Score</div>", unsafe_allow_html=True)

            display_results = results[:top_n_display]

            for rank, (score, cand, breakdown) in enumerate(display_results, 1):
                profile = cand.get("profile", {})
                sig = cand.get("redrob_signals", {})
                is_hp = breakdown["honeypot_risk"] >= 0.4

                if is_hp and not show_honeypots:
                    continue

                card_class = "candidate-card honeypot" if is_hp else "candidate-card"
                hp_badge = "🚨 HONEYPOT RISK" if is_hp else ""
                score_color = color_for_score(score)

                with st.expander(
                    f"{'🚨 ' if is_hp else ''}#{rank}  •  {cand.get('candidate_id', 'N/A')}  •  {profile.get('current_title', '')}  •  Score: {score:.4f}",
                    expanded=(rank <= 3),
                ):
                    col_left, col_right = st.columns([3, 2])

                    with col_left:
                        st.markdown(f"""
                        **{profile.get('anonymized_name', 'N/A')}** — {profile.get('headline', '')}

                        📍 {profile.get('location', 'N/A')}, {profile.get('country', '')} &nbsp;|&nbsp;
                        🏢 {profile.get('current_company', 'N/A')} ({profile.get('current_company_size', '')}) &nbsp;|&nbsp;
                        ⏳ {profile.get('years_of_experience', 0):.1f} yrs

                        > *{breakdown.get('reasoning', '')}*
                        """)

                        # Skills tags
                        matched = breakdown.get("matched_tier_a", [])
                        if matched:
                            tags = "".join(f'<span class="skill-tag">✅ {s}</span>' for s in matched[:8])
                            st.markdown(f"**Core Skills Matched:** {tags}", unsafe_allow_html=True)

                        # Behavioral quick stats
                        b_cols = st.columns(4)
                        behavioral_items = [
                            ("🟢 Open to work" if sig.get("open_to_work_flag") else "🔴 Not seeking",),
                            (f"💬 {sig.get('recruiter_response_rate',0):.0%} response",),
                            (f"⏰ {sig.get('notice_period_days',60)}d notice",),
                            (f"⭐ GH: {sig.get('github_activity_score',-1):.0f}",),
                        ]
                        for bc, (item,) in zip(b_cols, behavioral_items):
                            bc.markdown(f"<small>{item}</small>", unsafe_allow_html=True)

                    with col_right:
                        # Score breakdown bars
                        st.markdown("**Score Breakdown**")
                        st.markdown(
                            score_bar_html(breakdown["career_total"], "Career DNA", "#818cf8") +
                            score_bar_html(breakdown["career_ai"], "  → AI Seniority", "#a78bfa") +
                            score_bar_html(breakdown["career_product"], "  → Product Co.", "#8b5cf6") +
                            score_bar_html(breakdown["career_stuffing"], "  → Skill Integrity", "#7c3aed") +
                            score_bar_html(breakdown["skills_total"], "Skills Fit", "#34d399") +
                            score_bar_html(breakdown["behavioral_total"], "Behavioral", "#f59e0b") +
                            score_bar_html(breakdown["behavioral_avail"], "  → Availability", "#fbbf24") +
                            score_bar_html(breakdown["behavioral_engage"], "  → Engagement", "#fcd34d"),
                            unsafe_allow_html=True,
                        )
                        # Final score gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score * 100,
                            domain={"x": [0, 1], "y": [0, 1]},
                            number={"suffix": "%", "font": {"size": 22, "color": score_color}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "rgba(148,163,184,0.5)"},
                                "bar": {"color": score_color},
                                "bgcolor": "rgba(15,23,42,0.5)",
                                "bordercolor": "rgba(99,102,241,0.3)",
                                "steps": [
                                    {"range": [0, 35], "color": "rgba(239,68,68,0.1)"},
                                    {"range": [35, 60], "color": "rgba(245,158,11,0.1)"},
                                    {"range": [60, 100], "color": "rgba(16,185,129,0.1)"},
                                ],
                            },
                            title={"text": "NEXUS Score", "font": {"color": "rgba(148,163,184,0.8)", "size": 12}},
                        ))
                        fig.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="rgba(148,163,184,0.8)",
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{rank}")

        with tab2:
            st.markdown("<div class='section-header'>Score Distribution & Analytics</div>", unsafe_allow_html=True)

            scores_list = [s for s, _, _ in results[:100]]
            career_scores = [bd["career_total"] for _, _, bd in results[:100]]
            behavioral_scores = [bd["behavioral_total"] for _, _, bd in results[:100]]
            titles = [c.get("profile", {}).get("current_title", "Unknown") for _, c, _ in results[:100]]
            countries = [c.get("profile", {}).get("country", "Unknown") for _, c, _ in results[:100]]

            col_a, col_b = st.columns(2)
            with col_a:
                # Score distribution
                fig = px.histogram(
                    x=scores_list,
                    nbins=20,
                    title="Top-100 Score Distribution",
                    labels={"x": "NEXUS Score", "y": "Count"},
                    color_discrete_sequence=["#818cf8"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(15,23,42,0.5)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(148,163,184,0.8)",
                    title_font_color="rgba(226,232,240,0.9)",
                )
                st.plotly_chart(fig, use_container_width=True, key="score_dist")

            with col_b:
                # Career vs Behavioral scatter
                fig = px.scatter(
                    x=career_scores,
                    y=behavioral_scores,
                    color=scores_list,
                    color_continuous_scale="Viridis",
                    title="Career DNA vs Behavioral Readiness",
                    labels={"x": "Career DNA Score", "y": "Behavioral Score", "color": "Final Score"},
                    hover_name=[c.get("candidate_id") for _, c, _ in results[:100]],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(15,23,42,0.5)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(148,163,184,0.8)",
                    title_font_color="rgba(226,232,240,0.9)",
                )
                st.plotly_chart(fig, use_container_width=True, key="career_behav_scatter")

            # Title distribution
            from collections import Counter
            title_counts = Counter(titles).most_common(10)
            fig = px.bar(
                x=[t for t, _ in title_counts],
                y=[c for _, c in title_counts],
                title="Top-100 Current Titles",
                labels={"x": "Title", "y": "Count"},
                color=[c for _, c in title_counts],
                color_continuous_scale="Plasma",
            )
            fig.update_layout(
                plot_bgcolor="rgba(15,23,42,0.5)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(148,163,184,0.8)",
                title_font_color="rgba(226,232,240,0.9)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key="title_dist")

        with tab3:
            st.markdown("<div class='section-header'>🚨 Honeypot & Anomaly Radar</div>", unsafe_allow_html=True)
            st.markdown("""
            NEXUS detected the following candidates with elevated honeypot risk.
            These are excluded or heavily penalised in the final ranking.
            """)

            from nexus.integrity import get_honeypot_reasons
            honeypot_candidates = [
                (s, c, bd) for s, c, bd in results
                if bd["honeypot_risk"] >= 0.3
            ][:20]

            if not honeypot_candidates:
                st.markdown("""
                <div class="alert-success">
                ✅ No significant honeypot risk detected in this candidate set.
                </div>
                """, unsafe_allow_html=True)
            else:
                for score, cand, breakdown in honeypot_candidates:
                    risk = breakdown["honeypot_risk"]
                    risk_color = "#ef4444" if risk >= 0.65 else "#f59e0b"
                    reasons = get_honeypot_reasons(cand)
                    profile = cand.get("profile", {})

                    with st.expander(
                        f"⚠️ {cand.get('candidate_id')} — Risk: {risk:.0%} — {profile.get('current_title','')}",
                    ):
                        st.markdown(f"**NEXUS Score:** {score:.4f} (heavily penalised)")
                        st.markdown(f"**Honeypot Risk:** `{risk:.4f}`")
                        if reasons:
                            st.markdown("**Red Flags Detected:**")
                            for r in reasons:
                                st.markdown(f"<span class='skill-tag honeypot-tag'>🚩 {r}</span>", unsafe_allow_html=True)

                        # Show suspicious skills
                        skills = cand.get("skills", [])
                        expert_skills = [s["name"] for s in skills if s.get("proficiency") == "expert"]
                        if expert_skills:
                            st.markdown(f"**Expert-level claims:** {', '.join(expert_skills)}")

        with tab4:
            st.markdown("<div class='section-header'>📥 Export Submission CSV</div>", unsafe_allow_html=True)

            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rank, (score, cand, breakdown) in enumerate(results[:100], 1):
                writer.writerow([
                    cand.get("candidate_id"),
                    rank,
                    f"{score:.4f}",
                    breakdown.get("reasoning", ""),
                ])

            csv_data = output.getvalue()
            st.download_button(
                label="⬇️ Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("**Preview (first 10 rows):**")
            preview_lines = csv_data.splitlines()[:11]
            st.code("\n".join(preview_lines), language="text")

    else:
        # Welcome state
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:rgba(148,163,184,0.6);">
            <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
            <h2 style="color:rgba(226,232,240,0.7); font-weight:600;">Ready to Rank</h2>
            <p>Upload <code>candidates.jsonl</code> or <code>sample_candidates.json</code><br>
            and click <strong>Run NEXUS Ranking Pipeline</strong> to begin.</p>
            <p style="font-size:0.85rem; margin-top:1rem;">
                Or click <strong>Load Sample Data</strong> to use the included 50-candidate sample.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature showcase
        cols = st.columns(3)
        features = [
            ("🛡️", "Honeypot Detection", "Flags 6 types of impossible profiles before they pollute your ranking"),
            ("🧬", "Career DNA Analysis", "Understands career trajectories, not just skill keyword lists"),
            ("📡", "Behavioral Signals", "Integrates all 23 Redrob signals as a hiring-probability multiplier"),
        ]
        for col, (icon, title, desc) in zip(cols, features):
            col.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="color:#e2e8f0;font-weight:600;margin:0.5rem 0;">{title}</div>
                <div style="color:rgba(148,163,184,0.7);font-size:0.85rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
