import re
from datetime import datetime

import streamlit as st


st.set_page_config(
    page_title="MindCare AI",
    page_icon="MC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #24122e;
      --muted: #74637b;
      --paper: #fbf8ff;
      --panel: rgba(255,255,255,.88);
      --line: rgba(42,14,60,.13);
      --navy: #2a0e3c;
      --gold: #d8b96a;
      --rose: #a87ca0;
    }
    .stApp {
      background: linear-gradient(135deg, #fbf8ff 0%, #f2edf7 48%, #fffaf2 100%);
      color: var(--ink);
    }
    .block-container { max-width: 1180px; padding-top: 1rem; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #2a0e3c 0%, #24102f 60%, #170420 100%);
      color: white;
    }
    [data-testid="stSidebar"] * { color: white; }
    .hero {
      border-radius: 24px;
      padding: 42px;
      background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(255,255,255,.72));
      border: 1px solid rgba(42,14,60,.1);
      box-shadow: 0 24px 80px rgba(42,14,60,.12);
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 28px;
      align-items: center;
    }
    .hero h1 {
      font-size: 56px;
      line-height: 1.05;
      margin: 12px 0;
      color: #5e2b6d;
      font-weight: 900;
    }
    .eyebrow {
      display: inline-flex;
      padding: 9px 14px;
      border-radius: 999px;
      background: #ece9fb;
      border: 1px solid #d9d1ef;
      color: #145b62;
      font-weight: 800;
      font-size: 13px;
    }
    .panel, .card {
      border-radius: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 18px 50px rgba(42,14,60,.10);
      padding: 24px;
    }
    .dark-panel {
      background: #2a0e3c;
      color: white;
      border-radius: 22px;
      padding: 28px;
    }
    .dark-panel * { color: white; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 20px;
    }
    .metric {
      border-radius: 14px;
      padding: 18px;
      background: rgba(255,255,255,.12);
      border: 1px solid rgba(255,255,255,.2);
    }
    .metric span { font-size: 12px; text-transform: uppercase; font-weight: 800; opacity: .8; }
    .metric b { display: block; font-size: 24px; margin-top: 8px; }
    .feature-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      margin-top: 22px;
    }
    .card h3 { margin-top: 0; color: #5e2b6d; }
    .risk-low { color: #0f7b52; font-weight: 900; }
    .risk-medium { color: #ad7600; font-weight: 900; }
    .risk-high { color: #b42318; font-weight: 900; }
    @media (max-width: 900px) {
      .hero, .feature-grid { grid-template-columns: 1fr; }
      .hero h1 { font-size: 40px; }
      .metric-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


POSITIVE_WORDS = {
    "happy", "calm", "good", "great", "fine", "better", "hope", "peace",
    "smile", "relaxed", "positive", "strong", "love", "support",
}

RISK_WORDS = {
    "sad", "stress", "stressed", "anxiety", "anxious", "depressed", "alone",
    "cry", "tension", "panic", "hopeless", "tired", "fear", "angry", "low",
    "hurt", "broken", "empty", "pressure", "worry", "worried",
}


def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())).strip()


def analyze_text(text):
    tokens = clean_text(text).split()
    risk_hits = sum(1 for word in tokens if word in RISK_WORDS)
    positive_hits = sum(1 for word in tokens if word in POSITIVE_WORDS)
    score = 30 + risk_hits * 18 - positive_hits * 10
    if any(word in clean_text(text) for word in ["very sad", "too much", "give up", "no hope"]):
        score += 18
    score = max(5, min(95, score))
    if score >= 65:
        return score, "High Risk", "risk-high", "Strong emotional distress signal detected."
    if score >= 40:
        return score, "Medium Risk", "risk-medium", "Some stress or low mood pattern detected."
    return score, "Low Risk", "risk-low", "Text looks mostly stable or positive."


st.sidebar.markdown("## MindCare AI")
st.sidebar.caption("AI assisted mental health awareness project")
page = st.sidebar.radio(
    "Menu",
    ["Home", "Text Analysis", "Results Dashboard", "Project Details", "Privacy"],
)

if "history" not in st.session_state:
    st.session_state.history = []


if page == "Home":
    latest = st.session_state.history[-1] if st.session_state.history else None
    latest_score = f"{latest['score']}%" if latest else "--"
    latest_risk = latest["risk"] if latest else "Waiting"
    latest_note = latest["note"] if latest else "Run an analysis to create your first signal"

    st.markdown(
        f"""
        <section class="hero">
          <div>
            <div class="eyebrow">AI assisted wellbeing intelligence</div>
            <h1>MindCare AI for clearer emotional signals.</h1>
            <p>
              Analyze social media-style text and mood notes to estimate low,
              medium, or high emotional risk patterns for awareness and review.
            </p>
          </div>
          <div class="dark-panel">
            <h3>No diagnosis, only awareness</h3>
            <p>{latest_note}</p>
            <div class="metric-grid">
              <div class="metric"><span>Latest score</span><b>{latest_score}</b></div>
              <div class="metric"><span>Status</span><b>{latest_risk}</b></div>
              <div class="metric"><span>Mode</span><b>Private</b></div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
          <div class="card"><h3>Text Analysis</h3><p>Enter a post, caption, or reflection and get an instant risk score.</p></div>
          <div class="card"><h3>Dashboard</h3><p>Recent checks are summarized with score and detected signal.</p></div>
          <div class="card"><h3>Project Purpose</h3><p>Built for educational mental health awareness, not medical diagnosis.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Text Analysis":
    st.markdown("## Text Analysis")
    st.write("Paste social media-style text, a caption, or a short mood note.")
    text = st.text_area(
        "Text to analyze",
        height=160,
        placeholder="Example: I feel stressed and alone today, too much pressure...",
    )
    if st.button("Analyze Text", use_container_width=True):
        if not text.strip():
            st.warning("Please enter text first.")
        else:
            score, risk, css_class, note = analyze_text(text)
            st.session_state.history.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "text": text.strip(),
                    "score": score,
                    "risk": risk,
                    "note": note,
                }
            )
            st.markdown(
                f"""
                <div class="panel">
                  <h3 class="{css_class}">{risk}</h3>
                  <h1>{score}%</h1>
                  <p>{note}</p>
                  <p><b>Suggestion:</b> If distress feels strong or unsafe, talk to a trusted person or mental health professional.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif page == "Results Dashboard":
    st.markdown("## Results Dashboard")
    if not st.session_state.history:
        st.info("No saved analysis yet. Go to Text Analysis and run one check.")
    else:
        rows = list(reversed(st.session_state.history))
        avg = sum(item["score"] for item in rows) / len(rows)
        high = sum(1 for item in rows if item["risk"] == "High Risk")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total checks", len(rows))
        c2.metric("Average score", f"{avg:.1f}%")
        c3.metric("High risk signals", high)
        st.dataframe(rows, use_container_width=True)

elif page == "Project Details":
    st.markdown("## Project Details")
    st.markdown(
        """
        <div class="panel">
          <h3>MindCare AI</h3>
          <p>This project demonstrates how AI-style text processing can support mental health awareness by identifying emotional patterns in user-written text.</p>
          <p><b>Modules:</b> text analysis, risk scoring, dashboard, privacy note, and project explanation.</p>
          <p><b>Important:</b> This is not a medical diagnosis system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Privacy":
    st.markdown("## Privacy")
    st.markdown(
        """
        <div class="panel">
          <p>This demo keeps analysis history only in the current browser session. Refreshing or redeploying can clear it.</p>
          <p>The project is intended for academic demonstration and awareness only.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
