import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explainer import analyze_article, load_model_bundle
from src.preprocessing import clean_text_for_nlp
from src.live_news import fetch_live_headlines, RSS_FEEDS
from sample_articles import SAMPLE_ARTICLES

# Page Configuration
st.set_page_config(
    page_title="NewsDetective AI — Live News Misinformation & Reliability Diagnostics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top banner card */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(120deg, #60a5fa, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.5;
        max-width: 850px;
    }

    /* KPI metric cards */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }

    /* Verdict Card */
    .verdict-card {
        border-radius: 16px;
        padding: 22px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 18px;
    }
    .verdict-reliable {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
    }
    .verdict-misleading {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-color: rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.15);
    }

    .badge-pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-reliable {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-misleading {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Live Feed Item Card */
    .live-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .live-card:hover {
        border-color: rgba(96, 165, 250, 0.35);
        background: rgba(30, 41, 59, 0.85);
    }
    .live-card-reliable {
        border-left: 4px solid #10b981;
    }
    .live-card-misleading {
        border-left: 4px solid #ef4444;
    }

    .source-tag {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Highlighted text container */
    .highlight-container {
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        line-height: 1.9;
        font-size: 0.98rem;
        color: #cbd5e1;
        max-height: 380px;
        overflow-y: auto;
    }

    .reasoning-box {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #60a5fa;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Load model metrics if available
@st.cache_resource
def get_metrics_and_bundle():
    bundle = load_model_bundle()
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return bundle, metrics


try:
    bundle, metrics = get_metrics_and_bundle()
except Exception as e:
    st.error(f"Error loading model bundle: {e}")
    st.stop()

# Header Section
st.markdown("""
<div class="hero-container">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 2rem;">🛡️</span>
        <div class="hero-title">NewsDetective AI</div>
    </div>
    <div class="hero-subtitle">
        Intelligent Real-Time News Reliability & Misinformation Diagnostic Platform. Automatically fetches live news headlines from global wire feeds,
        predicts authenticity, analyzes sensationalism signals, and exposes word-level explainability in real-time.
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Training Corpus</div>
        <div class="kpi-value">44,898 <span style="font-size: 0.85rem; color: #94a3b8; font-weight: normal;">articles</span></div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Classifier Accuracy</div>
        <div class="kpi-value" style="color: #34d399;">99.37%</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Passive-Aggressive F1</div>
        <div class="kpi-value" style="color: #38bdf8;">99.66%</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Live Stream Feeds</div>
        <div class="kpi-value" style="color: #c084fc;">Real-Time <span style="font-size: 0.85rem; color: #94a3b8; font-weight: normal;">RSS & Search</span></div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_live, tab_detect, tab_batch, tab_analytics, tab_architecture = st.tabs([
    "📡 Live Breaking News Stream & Predictions",
    "🔍 Real-Time Article Inspector",
    "📁 Batch Article Scanner",
    "📊 Model Benchmark & Analytics",
    "🧠 Methodology & XAI Engine"
])

# -------------------------------------------------------------
# TAB 1: LIVE BREAKING NEWS STREAM & PREDICTIONS
# -------------------------------------------------------------
with tab_live:
    st.subheader("📡 Live Global News Stream with Real-Time AI Predictions")
    st.caption("Fetches current live headlines from global news wire services and instantly predicts whether each story is potentially reliable or misleading.")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.8, 1, 0.8])
    with ctrl_col1:
        chosen_cat = st.selectbox("Select Topic Category:", list(RSS_FEEDS.keys()), index=0)
    with ctrl_col2:
        search_kw = st.text_input("Or Search Specific Topic / Keyword:", placeholder="e.g. Trump, AI, election, climate, crypto...")
    with ctrl_col3:
        max_feed_items = st.slider("Headlines count:", min_value=10, max_value=35, value=20, step=5)
    with ctrl_col4:
        st.write("")
        st.write("")
        refresh_live = st.button("🔄 Refresh Stream", type="primary", use_container_width=True)

    with st.spinner("Fetching live headlines & executing AI reliability inference..."):
        live_articles = fetch_live_headlines(
            category_or_url=chosen_cat,
            search_query=search_kw if search_kw.strip() else None,
            limit=max_feed_items,
            force_refresh=refresh_live
        )

    if not live_articles:
        st.warning("No live articles retrieved for this query. Try refreshing or changing search terms.")
    else:
        # Aggregated Metrics for the live feed
        total_live = len(live_articles)
        reliable_count = sum(1 for a in live_articles if a["verdict"] == "Reliable")
        misleading_count = total_live - reliable_count
        reliable_pct = round((reliable_count / total_live) * 100, 1)
        avg_sensationalism = round(sum(a["sensationalism_score"] for a in live_articles) / total_live, 1)

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Live Headlines Analyzed", total_live)
        with stat_col2:
            st.metric("Deemed Reliable", f"{reliable_count} ({reliable_pct}%)", delta="Credibility rate")
        with stat_col3:
            st.metric("Flagged Misleading / Clickbait", f"{misleading_count} ({100 - reliable_pct}%)", delta_color="inverse")
        with stat_col4:
            st.metric("Avg Sensationalism Index", f"{avg_sensationalism} / 100", help="Stylometric score measuring emotional/clickbait intensity")

        # Distribution Chart and Article Cards
        c_left, c_right = st.columns([1.2, 2.5])

        with c_left:
            st.markdown("##### 🥧 Live Stream Reliability Breakdown")
            df_pie = pd.DataFrame([
                {"Verdict": "Reliable", "Count": reliable_count},
                {"Verdict": "Misleading", "Count": misleading_count}
            ])
            fig_live_pie = px.pie(
                df_pie,
                names="Verdict",
                values="Count",
                color="Verdict",
                color_discrete_map={"Reliable": "#10b981", "Misleading": "#ef4444"},
                hole=0.5
            )
            fig_live_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260
            )
            st.plotly_chart(fig_live_pie, use_container_width=True)

            st.markdown("##### 🏷️ Filter Live Feed")
            filter_choice = st.radio("Display stories:", ["All Stories", "Only Reliable", "Only Misleading / Clickbait"], horizontal=True)

        with c_right:
            st.markdown("##### 📰 Real-Time Predicted News Stream")

            filtered_articles = live_articles
            if filter_choice == "Only Reliable":
                filtered_articles = [a for a in live_articles if a["verdict"] == "Reliable"]
            elif filter_choice == "Only Misleading / Clickbait":
                filtered_articles = [a for a in live_articles if a["verdict"] == "Misleading"]

            if not filtered_articles:
                st.info(f"No headlines matching filter '{filter_choice}'.")

            for idx, item in enumerate(filtered_articles):
                is_rel = item["verdict"] == "Reliable"
                card_border = "live-card-reliable" if is_rel else "live-card-misleading"
                badge_style = "badge-reliable" if is_rel else "badge-misleading"
                icon = "✅" if is_rel else "🚨"

                indicators_str = ""
                if item["top_indicators"]:
                    ind_names = ", ".join([f"{w['word']} ({w['score']:+.2f})" for w in item["top_indicators"]])
                    indicators_str = f"<span style='color: #94a3b8; font-size: 0.8rem;'>Influential tokens: {ind_names}</span>"

                st.markdown(f"""
                <div class="live-card {card_border}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span class="source-tag">📰 {item['publisher']} • {item['pub_date'][:16] if item['pub_date'] else 'Just now'}</span>
                        <span class="badge-pill {badge_style}">{icon} {item['verdict'].upper()} ({item['confidence_score']}%)</span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-bottom: 8px;">
                        <a href="{item['link']}" target="_blank" style="color: inherit; text-decoration: none;">{item['title']} ↗</a>
                    </div>
                    <div style="display: flex; gap: 16px; font-size: 0.8rem; color: #cbd5e1; flex-wrap: wrap;">
                        <span>Sensationalism: <b>{item['sensationalism_score']}/100</b></span>
                        <span>Caps: <b>{round(item['caps_ratio']*100, 1)}%</b></span>
                        <span>Subjectivity: <b>{item['subjectivity']}</b></span>
                        {indicators_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Optional deep dive expansion
                with st.expander(f"🔎 Inspect reasoning & token attribution for story #{idx+1}"):
                    ed1, ed2 = st.columns([1, 1])
                    with ed1:
                        st.markdown("**Highlighted Text Influence:**")
                        st.markdown(f"<div class='highlight-container'>{item['highlighted_html']}</div>", unsafe_allow_html=True)
                    with ed2:
                        st.markdown("**Algorithmic Reasoning:**")
                        for r in item["reasoning"]:
                            st.markdown(f"<div class='reasoning-box'>{r}</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: REAL-TIME ARTICLE INSPECTOR
# -------------------------------------------------------------
with tab_detect:
    col_input, col_preset = st.columns([2.5, 1])

    with col_preset:
        st.subheader("⚡ Quick Test Presets")
        st.caption("Select a sample news article to immediately observe model diagnostics:")
        selected_sample_title = st.selectbox(
            "Pre-loaded benchmark articles:",
            list(SAMPLE_ARTICLES.keys()),
            index=0
        )
        preset_text = SAMPLE_ARTICLES[selected_sample_title]

    with col_input:
        st.subheader("📰 Custom Article Text Input")
        headline_input = st.text_input(
            "Headline / Title (Optional):",
            placeholder="e.g. Breaking: Officials Announce New Energy Strategy..."
        )
        default_body = preset_text if preset_text else ""
        body_input = st.text_area(
            "Article Body / Text to Analyze:",
            value=default_body,
            height=160,
            placeholder="Paste news article content, press release, social media claim, or editorial copy here..."
        )

        analyze_button = st.button("🚀 Analyze Content Reliability", type="primary", use_container_width=True)

    combined_input = ""
    if headline_input.strip() and body_input.strip():
        combined_input = f"{headline_input.strip()}\n\n{body_input.strip()}"
    elif body_input.strip():
        combined_input = body_input.strip()
    elif headline_input.strip():
        combined_input = headline_input.strip()

    if analyze_button or combined_input:
        if not combined_input.strip():
            st.warning("Please enter some text or pick a pre-loaded sample above.")
        else:
            with st.spinner("Analyzing textual patterns, stylometry, and vocabulary markers..."):
                res = analyze_article(combined_input)

            if "error" in res:
                st.error(res["error"])
            else:
                st.write("")
                verdict = res["verdict"]
                conf = res["confidence_score"]
                tier = res["tier"]

                if verdict == "Reliable":
                    card_cls = "verdict-reliable"
                    pill_cls = "badge-reliable"
                    icon = "✅"
                else:
                    card_cls = "verdict-misleading"
                    pill_cls = "badge-misleading"
                    icon = "🚨"

                st.markdown(f"""
                <div class="verdict-card {card_cls}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                        <div>
                            <span class="badge-pill {pill_cls}">{icon} {verdict.upper()} CONTENT</span>
                            <h2 style="margin: 10px 0 4px 0; font-size: 1.8rem; font-weight: 800; color: #ffffff;">{tier}</h2>
                            <p style="color: #cbd5e1; margin: 0; font-size: 0.95rem;">
                                The AI engine evaluated this content against authentic news and deceptive clickbait corpora.
                            </p>
                        </div>
                        <div style="text-align: right; min-width: 160px;">
                            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Prediction Confidence</div>
                            <div style="font-size: 2.4rem; font-weight: 800; color: #f8fafc;">{conf}%</div>
                            <div style="font-size: 0.85rem; color: #cbd5e1;">Reliable: {res['prob_reliable']}% | Misleading: {res['prob_misleading']}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics Row
                lp = res["linguistic_patterns"]
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Sensationalism Score", f"{lp['sensationalism_score']} / 100", 
                              help="Composite index based on emotional triggers, punctuation exaggeration, and ALL-CAPS usage.")
                with m2:
                    st.metric("Caps / Shouting Ratio", f"{round(lp['caps_ratio']*100, 1)}%", 
                              help="Percentage of uppercase characters. Tabloid and clickbait content typically exceeds 8%.")
                with m3:
                    st.metric("Exclamation Count", lp['exclamation_count'],
                              help="Frequent exclamation marks indicate emotional rhetoric rather than objective journalism.")
                with m4:
                    st.metric("Tone Subjectivity", f"{lp['subjectivity']:.2f}",
                              help="0.0 = completely objective/factual, 1.0 = highly subjective/opinionated.")
                with m5:
                    st.metric("Word Count", lp['word_count'],
                              help="Total token count analyzed in this passage.")

                st.divider()

                col_left, col_right = st.columns([1.2, 1])

                with col_left:
                    st.subheader("💡 Key Indicators & Feature Attribution")
                    st.caption("Top word markers pushing the prediction towards Reliable (+) or Misleading (-):")

                    pos_words = res.get("top_reliable_indicators", [])[:7]
                    neg_words = res.get("top_misleading_indicators", [])[:7]

                    chart_data = []
                    for item in pos_words:
                        chart_data.append({
                            "word": item["word"],
                            "impact": item["score"],
                            "class": "Supports Reliable"
                        })
                    for item in neg_words:
                        chart_data.append({
                            "word": item["word"],
                            "impact": item["score"],
                            "class": "Supports Misleading"
                        })

                    if chart_data:
                        df_chart = pd.DataFrame(chart_data)
                        df_chart = df_chart.sort_values(by="impact", ascending=True)

                        fig = px.bar(
                            df_chart,
                            x="impact",
                            y="word",
                            orientation="h",
                            color="class",
                            color_discrete_map={
                                "Supports Reliable": "#10b981",
                                "Supports Misleading": "#ef4444"
                            },
                            text_auto=".2f"
                        )
                        fig.update_layout(
                            height=360,
                            margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#94a3b8"),
                            xaxis_title="Attribution Score (TF-IDF × Coefficient)",
                            yaxis_title=None,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No dominant vocabulary indicators exceeded statistical threshold.")

                    st.markdown("#### 🔎 Reasoning & Algorithmic Rationale")
                    for reason in res["reasoning"]:
                        st.markdown(f'<div class="reasoning-box">{reason}</div>', unsafe_allow_html=True)

                with col_right:
                    st.subheader("🖍️ In-Context Vocabulary Highlighter")
                    st.caption("Green pills indicate credible vocabulary correlates; Red pills indicate deceptive/clickbait patterns.")
                    st.markdown(f"""
                    <div class="highlight-container">
                        {res['highlighted_html']}
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")
                    st.subheader("🤖 Multi-Model Consensus")
                    consensus = res.get("model_consensus", {})
                    if consensus:
                        cons_rows = []
                        for m_name, m_data in consensus.items():
                            cons_rows.append({
                                "Model": m_name,
                                "Verdict": m_data["prediction"],
                                "Reliable Prob": f"{round(m_data['prob_reliable']*100, 1)}%",
                                "Misleading Prob": f"{round(m_data['prob_misleading']*100, 1)}%"
                            })
                        st.dataframe(pd.DataFrame(cons_rows), use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 3: BATCH ARTICLE SCANNER
# -------------------------------------------------------------
with tab_batch:
    st.subheader("📁 Batch News Article Processing")
    st.caption("Upload a CSV file containing articles to classify hundreds of items in bulk.")

    sample_csv_col, upload_col = st.columns([1, 2])
    with sample_csv_col:
        st.markdown("##### 📥 Download Sample CSV Template")
        sample_df = pd.DataFrame({
            "title": [
                "Senate advances landmark green energy infrastructure initiative",
                "SHOCKING: Leaked documents prove aliens running world government!",
                "Quarterly inflation figures stabilize as consumer retail spending ticks upward"
            ],
            "text": [
                "Federal lawmakers reached consensus after lengthy debate regarding renewable energy grants.",
                "Mainstream media blackout! Whistleblower exposes unbelievable conspiracy they hide from you!",
                "Government statistical bureaus released consumer price index metrics this morning."
            ]
        })
        st.download_button(
            label="Download Template CSV",
            data=sample_df.to_csv(index=False),
            file_name="news_sample_template.csv",
            mime="text/csv"
        )

    with upload_col:
        uploaded_file = st.file_uploader("Upload CSV file (must contain 'text' or 'title' column):", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV with {len(batch_df)} rows!")

            cols = batch_df.columns.tolist()
            default_col = "text" if "text" in cols else ("title" if "title" in cols else cols[0])
            chosen_col = st.selectbox("Select column containing article text:", cols, index=cols.index(default_col))

            if st.button("⚡ Run Batch Diagnostics", type="primary"):
                progress_bar = st.progress(0)
                results_list = []
                total = min(len(batch_df), 500)

                for i in range(total):
                    row_text = str(batch_df.iloc[i][chosen_col])
                    res = analyze_article(row_text)
                    results_list.append({
                        "Original Index": i,
                        "Snippet": row_text[:90] + "...",
                        "Verdict": res.get("verdict", "N/A"),
                        "Confidence (%)": res.get("confidence_score", 0.0),
                        "Reliable Prob (%)": res.get("prob_reliable", 0.0),
                        "Misleading Prob (%)": res.get("prob_misleading", 0.0),
                        "Sensationalism Index": res.get("linguistic_patterns", {}).get("sensationalism_score", 0.0),
                        "Subjectivity": res.get("linguistic_patterns", {}).get("subjectivity", 0.0)
                    })
                    progress_bar.progress((i + 1) / total)

                res_df = pd.DataFrame(results_list)
                st.subheader("📋 Batch Evaluation Results")

                b_col1, b_col2 = st.columns([1, 2])
                with b_col1:
                    counts = res_df["Verdict"].value_counts().reset_index()
                    counts.columns = ["Verdict", "Count"]
                    fig_pie = px.pie(
                        counts,
                        names="Verdict",
                        values="Count",
                        color="Verdict",
                        color_discrete_map={"Reliable": "#10b981", "Misleading": "#ef4444"},
                        hole=0.45
                    )
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#cbd5e1"),
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with b_col2:
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

                csv_data = res_df.to_csv(index=False)
                st.download_button(
                    label="💾 Export Evaluated Results to CSV",
                    data=csv_data,
                    file_name="news_detective_batch_results.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as ex:
            st.error(f"Error parsing uploaded file: {ex}")

# -------------------------------------------------------------
# TAB 4: MODEL BENCHMARK & ANALYTICS
# -------------------------------------------------------------
with tab_analytics:
    st.subheader("📊 Rigorous Model Comparison & Evaluation Metrics")
    st.caption("Trained on 44,898 articles from ISOT fake and real news datasets with stratified 80/20 train-test split.")

    if metrics and "model_performance" in metrics:
        perf = metrics["model_performance"]
        perf_rows = []
        for m_name, stats in perf.items():
            perf_rows.append({
                "Classifier": m_name,
                "Accuracy": f"{stats['accuracy']*100:.2f}%",
                "Precision": f"{stats['precision']*100:.2f}%",
                "Recall": f"{stats['recall']*100:.2f}%",
                "F1 Score": f"{stats['f1_score']*100:.2f}%",
                "Training Latency": f"{stats['train_time_sec']}s"
            })
        df_perf = pd.DataFrame(perf_rows)
        st.dataframe(df_perf, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎯 Model Accuracy & F1 Comparison")
            comp_chart = []
            for m_name, stats in perf.items():
                comp_chart.append({"Model": m_name, "Metric": "Accuracy (%)", "Score": stats["accuracy"] * 100})
                comp_chart.append({"Model": m_name, "Metric": "F1 Score (%)", "Score": stats["f1_score"] * 100})
            df_comp = pd.DataFrame(comp_chart)
            fig_bar = px.bar(
                df_comp,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                color_discrete_sequence=["#38bdf8", "#818cf8"],
                text_auto=".1f"
            )
            fig_bar.update_layout(
                yaxis=dict(range=[90, 101]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            st.markdown("#### 🔲 Logistic Regression Confusion Matrix")
            cm = perf["Logistic Regression"]["confusion_matrix"]
            cm_labels = ["Misleading", "Reliable"]
            fig_cm = px.imshow(
                cm,
                x=cm_labels,
                y=cm_labels,
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Sample Count"),
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.divider()
        st.subheader("📖 Global Vocabulary Lexicon: Top Discriminative Words")
        lex1, lex2 = st.columns(2)

        with lex1:
            st.markdown("##### 🟢 Top Credibility Correlates")
            st.caption("Words most strongly associated with genuine reporting across 21k+ articles:")
            rel_tokens = metrics.get("top_reliable_indicators", [])
            df_rel = pd.DataFrame(rel_tokens)
            st.dataframe(df_rel, use_container_width=True, hide_index=True)

        with lex2:
            st.markdown("##### 🔴 Top Deceptive / Misleading Correlates")
            st.caption("Words most strongly associated with fabricated/sensational news across 23k+ articles:")
            fake_tokens = metrics.get("top_fake_indicators", [])
            df_fake = pd.DataFrame(fake_tokens)
            st.dataframe(df_fake, use_container_width=True, hide_index=True)

# -------------------------------------------------------------
# TAB 5: METHODOLOGY & EXPLAINABILITY ENGINE
# -------------------------------------------------------------
with tab_architecture:
    st.subheader("🧠 System Architecture & Methodology")
    st.markdown("""
    The **NewsDetective AI** pipeline operates across four decoupled layers to guarantee speed, accuracy, and interpretability:
    """)

    st.markdown("""
    ```
    ┌───────────────────────────┐      ┌───────────────────────────┐
    │   Live RSS News Feed      │ ───> │ Text Preprocessing Engine │
    │   & Custom Ingestion      │      │ • Dateline & HTML removal │
    └───────────────────────────┘      │ • Punctuation & URL strip │
                                       └─────────────┬─────────────┘
                                                     │
                                                     ▼
    ┌───────────────────────────┐      ┌───────────────────────────┐
    │ Linguistic Diagnostics    │      │  N-Gram TF-IDF Vectorizer │
    │ • Sensationalism Index    │ <──> │  • Sublinear Term Freq    │
    │ • Caps / Shouting Ratio   │      │  • Unigrams + Bigrams     │
    │ • Emotional Subjectivity  │      │  • 25,000 Dimensions      │
    └─────────────┬─────────────┘      └─────────────┬─────────────┘
                  │                                  │
                  ▼                                  ▼
    ┌───────────────────────────┐      ┌───────────────────────────┐
    │  Reasoning & Indicators   │      │ Logistic & Ensemble Clf   │
    │  • Token Highlighting     │ <─── │ • Calibrated Probabilities│
    │  • Word-Level Log-Odds    │      │ • 99.37% Accuracy         │
    └─────────────┬─────────────┘      └─────────────┬─────────────┘
                  │                                  │
                  └─────────────────┬────────────────┘
                                    ▼
                      ┌───────────────────────────┐
                      │ Unified Reliability Report│
                      │  • Verdict & Confidence   │
                      │  • Visual Risk Meter      │
                      └───────────────────────────┘
    ```
    """)

    st.markdown("""
    ### 🔬 Scientific Innovations in Text Preprocessing
    1. **Live Global RSS Ingestion Engine**:
       - Continuously streams live breaking news across 7 categories (World, Politics, Tech, Business, Science, Health, Top Stories) and arbitrary search queries.
       - Runs real-time inference on each incoming headline with zero external API key requirements.
    2. **Agency Dateline Stripping**:
       - Naive models memorize publisher attributions like `"WASHINGTON (Reuters) -"` or wire prefixes, creating severe dataset leakage.
       - NewsDetective's `clean_text_for_nlp` function explicitly strips dateline patterns, ensuring classification decisions are founded on **linguistic style, journalistic neutrality, syntax, and rhetoric** rather than publisher brand names.
    3. **Linear Feature Attribution ($w_i \cdot x_i$)**:
       - Each token's influence is derived directly from its TF-IDF frequency multiplied by its model coefficient.
       - Positive scores contribute towards genuine reporting probability, while negative scores indicate sensational or fabricated patterns.
    4. **Composite Sensationalism Index**:
       - Evaluates stylometric features including ALL-CAPS ratios, excessive exclamation marks, clickbait phrase lexicons, and sentiment polarity.
    """)
