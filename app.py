import streamlit as st
from pathlib import Path
from model import load_artifacts, analyze_news

# ---------- Page setup ----------
st.set_page_config(page_title="TruthGuard | Fake News Detector", page_icon="🛡️", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --tg-bg:#0a1020; --tg-panel:#111a2c; --tg-border:#26334a; --tg-text:#eaf0ff; --tg-muted:#9aa9c4; --tg-cyan:#62d6f5; }
.stApp { background: radial-gradient(ellipse at 10% 0%, #142743 0%, #0a1020 42%, #080d18 100%); color:var(--tg-text); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; }
.block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #0d1525; border-right:1px solid #243148; }
[data-testid="stSidebar"] * { color:#dce7fb; }
.tg-hero { border:1px solid #283950; background:linear-gradient(135deg,rgba(23,42,68,.94),rgba(14,23,39,.93)); padding:28px 30px; border-radius:22px; box-shadow:0 18px 50px rgba(0,0,0,.18); margin-bottom:20px; }
.tg-eyebrow { color:#71ddf5; letter-spacing:2.1px; text-transform:uppercase; font-size:11px; font-weight:700; }
.tg-title { font-family:'Space Grotesk',sans-serif; font-size:40px; font-weight:700; letter-spacing:-1px; margin:7px 0 4px; color:#f2f6ff; }
.tg-sub { color:#aab8d0; font-size:16px; margin:0; }
.tg-card { border:1px solid #26354d; background:rgba(16,26,44,.88); border-radius:18px; padding:18px 20px; min-height:112px; }
.tg-card-title { color:#a8b6ce; font-size:12px; letter-spacing:.7px; text-transform:uppercase; margin-bottom:8px; }
.tg-card-value { color:#eef4ff; font-size:24px; font-weight:700; font-family:'Space Grotesk',sans-serif; }
.tg-small { color:#93a5c3; font-size:13px; }
.tg-result { padding:18px 20px; border-radius:18px; border:1px solid; margin:8px 0; }
.tg-real { background:rgba(24,80,62,.26); border-color:#286e55; }
.tg-fake { background:rgba(110,36,51,.24); border-color:#7a3548; }
.tg-neutral { background:rgba(28,42,64,.4); border-color:#34465f; }
.tg-result-label { font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; margin-bottom:3px; }
div.stButton > button { border-radius:12px; border:1px solid #2c819b; background:linear-gradient(135deg,#1d7994,#18556e); color:#fff; font-weight:700; min-height:44px; }
div.stButton > button:hover { border-color:#6de1fa; color:#fff; background:linear-gradient(135deg,#2495b2,#1c6683); }
textarea { border-radius:12px !important; }
hr { border-color:#243148 !important; }
.stTextArea textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🛡️ TruthGuard")
    st.caption("ML-powered news text classifier")
    st.markdown("---")
    page = st.radio("Navigate", ["News Analyzer", "How It Works", "About Project"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model suite**")
    st.markdown("• Logistic Regression\n• Decision Tree\n• Gradient Boosting\n• Random Forest")
    st.caption("Predictions are model outputs, not a substitute for independent fact-checking.")

# ---------- Header ----------
st.markdown("""
<div class="tg-hero">
  <div class="tg-eyebrow">Machine learning · Text classification</div>
  <div class="tg-title">TruthGuard <span style="color:#62d6f5"></span></div>
  <p class="tg-sub">Analyze news text with four machine-learning models and compare their predictions.</p>
</div>
""", unsafe_allow_html=True)

if page == "News Analyzer":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="tg-card"><div class="tg-card-title">Model ensemble</div><div class="tg-card-value">4 Models</div><div class="tg-small">Independent predictions</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="tg-card"><div class="tg-card-title">Text features</div><div class="tg-card-value">TF-IDF</div><div class="tg-small">Text preprocessing + vectorization</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="tg-card"><div class="tg-card-title">Output</div><div class="tg-card-value">Real / Fake</div><div class="tg-small">Compare model agreement</div></div>', unsafe_allow_html=True)

    st.markdown("## News Analysis")
    st.markdown("Paste the article text below. For better results, provide the full article body rather than only a headline.")
    example_texts = {
        "Select an example (optional)": "",
        "Example — news-style article": "Officials announced a new public transport project today, saying the proposal will be reviewed by the city council before construction begins.",
        "Example — short headline": "A new report claims a major policy change will take effect next week.",
    }
    chosen = st.selectbox("Quick sample", list(example_texts.keys()))
    if chosen != "Select an example (optional)" and not st.session_state.get("example_loaded", False):
        st.session_state.news_input = example_texts[chosen]
    elif chosen == "Select an example (optional)":
        st.session_state.example_loaded = False

    
    news_input = st.text_area(
        "News text",
        height=230,
        placeholder="Paste or type the news article here…",
        key="news_input",
    )
    def clear_form():
        st.session_state.news_input = ""
        st.session_state.last_analysis = None
        st.session_state.example_loaded = False

    b1, b2 = st.columns([1,1])
    with b1:
        analyze_clicked = st.button("🔎  Analyze News", use_container_width=True)
    with b2:
        st.button("↻  Clear text", use_container_width=True, on_click=clear_form)

    if analyze_clicked:
        if not news_input or not news_input.strip():
            st.warning("Please enter some news text before analyzing.")
        elif len(news_input.strip()) < 40:
            st.warning("This text is very short. Try providing more context for a more meaningful model prediction.")
        else:
            try:
                artifacts = load_artifacts(MODEL_DIR)
                results = analyze_news(news_input, artifacts)
                st.session_state.last_analysis = results
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    results = st.session_state.get("last_analysis")
    if results:
        st.markdown("## Analysis Results")
        fake_votes = sum(1 for r in results["models"].values() if r["class"] == 0)
        real_votes = len(results["models"]) - fake_votes
        overall_class = 1 if real_votes > fake_votes else 0 if fake_votes > real_votes else None
        if overall_class == 1:
            label, css, icon = "REAL NEWS — majority model vote", "tg-real", "🟢"
        elif overall_class == 0:
            label, css, icon = "FAKE NEWS — majority model vote", "tg-fake", "🔴"
        else:
            label, css, icon = "NO MAJORITY — models disagree", "tg-neutral", "🟡"
        st.markdown(f'<div class="tg-result {css}"><div class="tg-result-label">{icon} {label}</div><div>{real_votes} Real vote(s) · {fake_votes} Fake vote(s)</div></div>', unsafe_allow_html=True)

        model_cols = st.columns(4)
        names = [
            ("Logistic Regression", "LR"),
            ("Decision Tree", "DT"),
            ("Gradient Boosting", "GB"),
            ("Random Forest", "RF"),
        ]
        for col, (display, key) in zip(model_cols, names):
            pred = results["models"][key]
            is_real = pred["class"] == 1
            color = "#67d9a1" if is_real else "#ff8b96"
            text = "REAL NEWS" if is_real else "FAKE NEWS"
            with col:
                st.markdown(f"""
                <div class="tg-card">
                  <div class="tg-card-title">{display}</div>
                  <div class="tg-card-value" style="color:{color};font-size:20px">{text}</div>
                  <div class="tg-small">Prediction: {pred['label']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption("The majority verdict is a simple vote across these four models. Model predictions can be wrong; verify claims using reliable sources.")

elif page == "How It Works":
    st.markdown("## How TruthGuard works")
    st.markdown("Your existing notebook uses the same pipeline below. The app loads the saved TF-IDF vectorizer and trained classifiers.")
    steps = [
        ("01", "News input", "User pastes the article or news text."),
        ("02", "Preprocessing", "The notebook's wordopt() cleaning function lowercases text and removes selected noise."),
        ("03", "TF-IDF", "The saved vectorizer converts the cleaned text into numerical features."),
        ("04", "Classification", "Four trained classifiers independently predict the label."),
        ("05", "Model comparison", "TruthGuard shows each model's prediction and a simple majority vote."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="tg-card" style="margin:12px 0;min-height:unset">
          <div class="tg-eyebrow">{num}</div>
          <div style="font-size:19px;font-weight:700;margin:5px 0">{title}</div>
          <div class="tg-small">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "About Project":
    st.markdown("## About TruthGuard")
    st.markdown("""
    **TruthGuard – Fake News Detection Model using Machine Learning** is an academic project
    that applies text classification techniques to news content.

    It uses the preprocessing and TF-IDF workflow from the existing Colab notebook and compares
    four trained classifiers: Logistic Regression, Decision Tree, Gradient Boosting and Random Forest.

    The app is intended for educational demonstration. It does not check news against live sources
    and should not be treated as a definitive fact-checking service.
    """)
    st.info("A model can learn patterns from its training data, but its output is not proof that a news story is true or false.")

st.markdown("---")
st.markdown('<div class="tg-small" style="text-align:center">TruthGuard · MCA Minor Project · Machine Learning based news classification</div>', unsafe_allow_html=True)
