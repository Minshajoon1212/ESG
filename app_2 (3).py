"""
ESG Risk Intelligence Platform
================================
Professional ESG Risk Prediction and Decision Support System.
Models: Logistic Regression | Random Forest
Features: revenue_growth, debt_to_equity, return_on_assets,
          current_ratio, market_volatility, stock_return, esg_score
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, os, io, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Risk Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — White canvas | Dark navy sidebar | Full contrast
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
* { font-family: 'Times New Roman', Times, serif !important; }

/* ── White main canvas ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
section[data-testid="stMain"] > div {
    background-color: #ffffff !important;
}

/* ── All text dark in main area ── */
.stApp p, .stApp span, .stApp div, .stApp label,
.stApp li, .stApp td, .stApp caption,
.stMarkdown, .element-container,
[data-testid="stMarkdownContainer"] > * {
    color: #111111 !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #0d1b2a !important;
}

/* ── SIDEBAR — light grey ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #F5F5F5 !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #111111 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #0B3C5D !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #111111 !important;
    background-color: transparent !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #0B3C5D !important; }

/* ── Number / text inputs in MAIN area ── */
input[type="number"], input[type="text"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #b0a080 !important;
    border-radius: 2px !important;
}
.stNumberInput label, .stTextInput label {
    color: #111111 !important;
    font-weight: 700 !important;
    font-size: 0.87rem !important;
    letter-spacing: 0.03em !important;
}

/* ── Selectbox main area ── */
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #b0a080 !important;
    border-radius: 2px !important;
}
[data-baseweb="select"] span { color: #111111 !important; }
[data-baseweb="select"] svg { fill: #444444 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #f5f3ee !important;
    border: 1px solid #d0c8ba !important;
    border-left: 3px solid #c8a96e !important;
    padding: 0.9rem 1.1rem !important;
    border-radius: 0 !important;
}
[data-testid="stMetricValue"] {
    color: #0d1b2a !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #444444 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── PRIMARY BUTTON — Run Assessment ── */
.stButton > button {
    background-color: #0B3C5D !important;
    color: #ffffff !important;
    border: 1.5px solid #0B3C5D !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #1e5a8a !important;
    color: #ffffff !important;
    border-color: #1e5a8a !important;
}
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #ffffff !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background-color: #0B3C5D !important;
    color: #ffffff !important;
    border: 1.5px solid #0B3C5D !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.5rem !important;
}
.stDownloadButton > button:hover { background-color: #1e5a8a !important; }
.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #ffffff !important;
}

/* ── Form container ── */
[data-testid="stForm"] {
    background-color: #fafaf8 !important;
    border: 1px solid #d8d0c4 !important;
    padding: 1.5rem !important;
    border-radius: 0 !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid #d0c8ba !important;
}
.stDataFrame thead tr th {
    background-color: #0d1b2a !important;
    color: #e8dcc8 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}
.stDataFrame tbody tr:nth-child(even) td {
    background-color: #f5f3ee !important;
}

/* ── st.table native ── */
table {
    border-collapse: collapse !important;
    width: 100% !important;
}
table th {
    background-color: #0d1b2a !important;
    color: #e8dcc8 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 0.9rem !important;
    border: 1px solid #1e3a5a !important;
}
table td {
    color: #111111 !important;
    padding: 0.5rem 0.9rem !important;
    border: 1px solid #d8d0c4 !important;
    font-size: 0.9rem !important;
}
table tr:nth-child(even) td {
    background-color: #f5f3ee !important;
}

/* ── Divider ── */
hr { border-color: #d0c8ba !important; }

/* ── Alert / info boxes ── */
.stAlert { border-radius: 0 !important; }
[data-testid="stAlert"] { border-radius: 0 !important; }

/* ── Warning box — keep readable ── */
[data-testid="stAlert"][data-type="warning"] * { color: #5a3a00 !important; }
[data-testid="stAlert"][data-type="error"] *   { color: #5a0000 !important; }
[data-testid="stAlert"][data-type="success"] * { color: #0a3a10 !important; }
[data-testid="stAlert"][data-type="info"] *    { color: #0a2a4a !important; }

/* ── Spinner text ── */
.stSpinner > div { color: #0d1b2a !important; }

/* ── Bar chart labels ── */
[data-testid="stVegaLiteChart"] text { fill: #111111 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "revenue_growth", "debt_to_equity", "return_on_assets",
    "current_ratio", "market_volatility", "stock_return", "esg_score",
]
FEATURE_LABELS = {
    "revenue_growth":    "Revenue Growth (%)",
    "debt_to_equity":    "Debt-to-Equity Ratio",
    "return_on_assets":  "Return on Assets — ROA (%)",
    "current_ratio":     "Current Ratio",
    "market_volatility": "Market Volatility",
    "stock_return":      "Stock Return (%)",
    "esg_score":         "ESG Score (0 – 100)",
}
BENCHMARKS = {
    "debt_to_equity":   {"label": "Debt-to-Equity",  "target": "< 1.5",    "check": lambda v: v < 1.5},
    "return_on_assets": {"label": "Return on Assets", "target": "> 5%",     "check": lambda v: v > 5},
    "current_ratio":    {"label": "Current Ratio",    "target": "1.5 – 2.5","check": lambda v: 1.5 <= v <= 2.5},
    "esg_score":        {"label": "ESG Score",        "target": "> 60",     "check": lambda v: v > 60},
}
DEFAULTS = {
    "revenue_growth": 5.0, "debt_to_equity": 1.8, "return_on_assets": 3.5,
    "current_ratio": 1.2,  "market_volatility": 0.25, "stock_return": 4.0, "esg_score": 50.0,
}

PKL_SEARCH_PATHS = [
    ".",
    "/mnt/user-data/uploads",
    os.path.expanduser("~"),
]


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models, errors = {}, []
    artifacts = {
        "logistic":      "logistic_model.pkl",
        "random_forest": "random_forest_model.pkl",
        "scaler":        "scaler.pkl",
        "label_encoder": "label_encoder.pkl",
    }
    for key, filename in artifacts.items():
        found = False
        for search_dir in PKL_SEARCH_PATHS:
            full_path = os.path.join(search_dir, filename)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "rb") as f:
                        models[key] = pickle.load(f)
                    found = True
                    break
                except Exception as e:
                    errors.append(f"Error loading {filename}: {e}")
                    break
        if not found and key not in models:
            errors.append(f"File not found: {filename}")
    return models, errors


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────
def predict(models: dict, inputs: np.ndarray, model_choice: str):
    le = models.get("label_encoder")
    if model_choice == "Logistic Regression":
        model  = models.get("logistic")
        scaler = models.get("scaler")
        if model is None:
            return None, None, None
        X = scaler.transform(inputs) if scaler else inputs
    else:
        model = models.get("random_forest")
        if model is None:
            return None, None, None
        X = inputs

    raw_pred   = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    label      = le.inverse_transform([raw_pred])[0] if le else str(raw_pred)
    confidence = float(np.max(proba))
    classes    = le.classes_ if le else [str(c) for c in model.classes_]
    return label, confidence, dict(zip(classes, proba))


# ─────────────────────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
def get_recommendations(risk_level: str, u: dict) -> list:
    r = risk_level.strip().lower()
    if r == "high":
        return [
            f"Restructure debt obligations to reduce the Debt-to-Equity ratio from its current level "
            f"of {u['debt_to_equity']:.2f} to below 1.5. Consider equity issuance, asset divestitures, "
            f"or refinancing high-cost liabilities.",

            f"Improve Return on Assets from {u['return_on_assets']:.2f}% to above 5% by streamlining "
            f"operational costs, optimising asset utilisation, and eliminating non-performing assets "
            f"from the balance sheet.",

            f"Strengthen the Current Ratio from {u['current_ratio']:.2f} to above 1.5 by accelerating "
            f"receivables collection, renegotiating payable terms, and securing working capital credit "
            f"facilities to buffer short-term obligations.",

            f"Elevate the ESG Score from {u['esg_score']:.1f} to above 60 by adopting formal sustainability "
            f"reporting frameworks (GRI or SASB), setting science-based emissions reduction targets, "
            f"and strengthening board-level ESG governance.",

            f"Reduce market volatility exposure (currently {u['market_volatility']:.3f}) through derivatives "
            f"hedging strategies, sector diversification, and active portfolio duration management to "
            f"insulate against systemic market shocks.",

            f"Accelerate revenue growth from {u['revenue_growth']:.2f}% toward 8%+ through geographic "
            f"market expansion, product portfolio diversification, and strategic partnerships or "
            f"M&A activity in high-growth segments.",
        ]
    elif r == "medium":
        return [
            f"Gradually reduce financial leverage, targeting a Debt-to-Equity ratio below 1.2 "
            f"from the current {u['debt_to_equity']:.2f} over a 12 to 18 month period through "
            f"disciplined debt amortisation and retained earnings reinvestment.",

            f"Advance the ESG Score from {u['esg_score']:.1f} toward 70 by formalising ESG disclosures, "
            f"engaging independent third-party verification, and integrating sustainability KPIs "
            f"into executive performance frameworks.",

            f"Stabilise core financial performance indicators — ROA at {u['return_on_assets']:.2f}%, "
            f"Current Ratio at {u['current_ratio']:.2f} — through rigorous capital allocation "
            f"discipline and working capital cycle optimisation.",

            f"Strengthen investor-facing ESG communications and governance disclosures to reduce "
            f"perceived information asymmetry and improve eligibility for ESG-linked financing instruments "
            f"such as green bonds and sustainability-linked loans.",
        ]
    else:  # Low
        return [
            f"The company's ESG profile — with a score of {u['esg_score']:.1f} and a strong financial "
            f"foundation — is well-suited to attract institutional investors with ESG mandates, "
            f"potentially broadening the shareholder base and supporting premium valuation multiples.",

            f"A Low ESG Risk classification typically correlates with a reduced cost of capital across "
            f"both debt and equity, reflecting lower regulatory, reputational, and operational risk "
            f"premia demanded by lenders and investors.",

            f"The organisation is strategically positioned to access ESG-linked financing instruments "
            f"including green bonds, social bonds, sustainability-linked credit facilities, and "
            f"impact-focused investment capital at competitive terms.",

            f"Sustain the current ESG positioning by embedding continuous improvement mechanisms — "
            f"annual ESG audits, science-based target refreshes, and supply chain sustainability "
            f"assessments — to defend and extend the low-risk classification over time.",
        ]


# ─────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
def get_feature_importance(models: dict, model_choice: str) -> pd.DataFrame:
    if model_choice == "Logistic Regression":
        model = models.get("logistic")
        if model is None or not hasattr(model, "coef_"):
            return pd.DataFrame()
        coef = model.coef_
        importance = np.mean(np.abs(coef), axis=0)
        direction  = ["Increases Risk" if coef[0][i] > 0 else "Reduces Risk" for i in range(len(FEATURES))]
        df = pd.DataFrame({
            "Feature":               [FEATURE_LABELS[f] for f in FEATURES],
            "Coefficient Magnitude": importance,
            "Effect on Risk":        direction,
        }).sort_values("Coefficient Magnitude", ascending=False).reset_index(drop=True)
        df.index += 1
        return df
    else:
        model = models.get("random_forest")
        if model is None or not hasattr(model, "feature_importances_"):
            return pd.DataFrame()
        df = pd.DataFrame({
            "Feature":          [FEATURE_LABELS[f] for f in FEATURES],
            "Importance Score": model.feature_importances_,
        }).sort_values("Importance Score", ascending=False).reset_index(drop=True)
        df["Importance Score (%)"] = (df["Importance Score"] * 100).round(2).astype(str) + "%"
        df.index += 1
        return df


# ─────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────
def generate_txt_report(user_inputs, model_choice, risk_label, confidence, probas, recs, bench_rows):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 72
    sub = "-" * 44
    lines = [
        sep,
        "   ESG RISK ASSESSMENT REPORT",
        f"   Generated : {ts}",
        f"   Model     : {model_choice}",
        sep, "",
        "SECTION 1 — INPUT PARAMETERS", sub,
    ]
    for k in FEATURES:
        lines.append(f"  {FEATURE_LABELS[k]:<40}  {user_inputs[k]:.4f}")
    lines += ["", "SECTION 2 — PREDICTION RESULTS", sub,
              f"  Predicted Risk Level  :  {risk_label.upper()}",
              f"  Confidence Score      :  {confidence*100:.2f}%", "",
              "  Probability Distribution:"]
    for cls, prob in probas.items():
        lines.append(f"    {cls:<10}  {prob*100:.2f}%")
    lines += ["", "SECTION 3 — BENCHMARK COMPARISON", sub]
    for row in bench_rows:
        lines.append(f"  {row['Metric']:<22}  Value: {row['Actual Value']:<10}  "
                     f"Target: {row['Benchmark']:<12}  [{row['Status'].upper()}]")
    lines += ["", "SECTION 4 — RECOMMENDATIONS", sub]
    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")
        lines.append("")
    lines += [sep,
              "  DISCLAIMER: Generated by a machine learning model for analytical purposes only.",
              "  This report does not constitute financial, legal, or investment advice.",
              sep]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# SECTION HEADING HELPER  (native st.markdown, no HTML tables)
# ─────────────────────────────────────────────────────────────
def section_heading(title: str):
    st.markdown(
        f"<div style='font-size:0.95rem; font-weight:700; text-transform:uppercase; "
        f"letter-spacing:0.13em; color:#0d1b2a; border-bottom:2px solid #c8a96e; "
        f"padding-bottom:0.35rem; margin:2rem 0 1.1rem 0;'>{title}</div>",
        unsafe_allow_html=True,
    )

def sidebar_heading(title: str) -> str:
    return (
        f"<div style='font-size:0.82rem; font-weight:700; text-transform:uppercase; "
        f"letter-spacing:0.13em; color:#0B3C5D !important; border-bottom:1px solid #0B3C5D; "
        f"padding-bottom:0.3rem; margin:1.4rem 0 0.7rem 0;'>{title}</div>"
    )


# ═════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════

# ── 1. Header Banner ─────────────────────────────────────────
st.markdown("""
<div style="background-color:#0d1b2a; color:#e8dcc8;
            padding:2.2rem 2.8rem 1.8rem 2.8rem;
            border-bottom:3px solid #c8a96e; margin-bottom:1.8rem;">
    <div style="font-size:2rem; font-weight:700; letter-spacing:0.05em;
                color:#e8dcc8; margin-bottom:0.3rem;">
        ESG Risk Intelligence Platform
    </div>
    <div style="font-size:0.85rem; color:#a89070; letter-spacing:0.12em;
                text-transform:uppercase;">
        Quantitative ESG Risk Assessment &amp; Decision Support System
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load Models ───────────────────────────────────────────────
with st.spinner("Loading model artifacts..."):
    models, load_errors = load_models()

models_loaded = ("logistic" in models or "random_forest" in models)

if load_errors and not models_loaded:
    st.error("Model files could not be located. Please ensure the four .pkl files are in the "
             "same directory as app.py, or in /mnt/user-data/uploads/.")
    for e in load_errors:
        st.caption(f"   {e}")
elif load_errors:
    missing = [e for e in load_errors if "not found" in e.lower() or "error" in e.lower()]
    if missing:
        st.info("Running with available model files. " + " | ".join(missing))


# ── 2. Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(sidebar_heading("Model Configuration"), unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Prediction Model",
        ["Logistic Regression", "Random Forest"],
        index=0,
    )

    st.markdown(sidebar_heading("Model File Status"), unsafe_allow_html=True)
    status_items = [
        ("logistic",      "Logistic Regression"),
        ("random_forest", "Random Forest"),
        ("scaler",        "Feature Scaler"),
        ("label_encoder", "Label Encoder"),
    ]
    for key, lbl in status_items:
        ok     = key in models
        colour = "#1a6a2a" if ok else "#8b1c1c"
        symbol = "OK" if ok else "--"
        st.markdown(
            f'<div style="font-size:0.82rem; color:{colour} !important; '
            f'margin-bottom:5px;">[{symbol}]  {lbl}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(sidebar_heading("About"), unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; line-height:1.75; color:#333333 !important;">
    This platform applies pre-trained ML models to assess corporate
    ESG risk based on financial and sustainability indicators.
    <br><br>
    <strong style="color:#0d1b2a !important;">Risk Levels:</strong><br>
    <span style="color:#8b1c1c !important;">High</span> &mdash;
    <span style="color:#7a5500 !important;">Medium</span> &mdash;
    <span style="color:#1a6a2a !important;">Low</span>
    <br><br>
    <strong style="color:#0d1b2a !important;">Data Source:</strong><br>
    Bloomberg Terminal financial and ESG indicators.
    </div>
    """, unsafe_allow_html=True)


# ── 3. Input Form ─────────────────────────────────────────────
section_heading("Financial and ESG Input Parameters")

with st.form("esg_form"):
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown(
            '<div style="font-weight:700; font-size:0.9rem; color:#0d1b2a; '
            'margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.07em;">'
            'Financial Indicators</div>',
            unsafe_allow_html=True,
        )
        revenue_growth   = st.number_input(FEATURE_LABELS["revenue_growth"],
                            value=DEFAULTS["revenue_growth"], step=0.1, format="%.2f",
                            help="Year-over-year revenue growth (%)")
        debt_to_equity   = st.number_input(FEATURE_LABELS["debt_to_equity"],
                            value=DEFAULTS["debt_to_equity"], min_value=0.0, step=0.05, format="%.2f",
                            help="Total debt / total shareholder equity")
        return_on_assets = st.number_input(FEATURE_LABELS["return_on_assets"],
                            value=DEFAULTS["return_on_assets"], step=0.1, format="%.2f",
                            help="Net income as a percentage of total assets")
        current_ratio    = st.number_input(FEATURE_LABELS["current_ratio"],
                            value=DEFAULTS["current_ratio"], min_value=0.0, step=0.05, format="%.2f",
                            help="Current assets / current liabilities")

    with col_r:
        st.markdown(
            '<div style="font-weight:700; font-size:0.9rem; color:#0d1b2a; '
            'margin-bottom:0.6rem; text-transform:uppercase; letter-spacing:0.07em;">'
            'Market and Sustainability Indicators</div>',
            unsafe_allow_html=True,
        )
        market_volatility = st.number_input(FEATURE_LABELS["market_volatility"],
                             value=DEFAULTS["market_volatility"], min_value=0.0, step=0.01, format="%.3f",
                             help="Annualised standard deviation of market returns")
        stock_return      = st.number_input(FEATURE_LABELS["stock_return"],
                             value=DEFAULTS["stock_return"], step=0.1, format="%.2f",
                             help="Annualised stock return (%)")
        esg_score         = st.number_input(FEATURE_LABELS["esg_score"],
                             value=DEFAULTS["esg_score"], min_value=0.0, max_value=100.0,
                             step=0.5, format="%.1f", help="Composite ESG score (0 – 100)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4. Run Assessment Button ───────────────────────────────
    submitted = st.form_submit_button(
        "Run ESG Risk Assessment",
        use_container_width=False,
    )


# ─────────────────────────────────────────────────────────────
# Assemble inputs
# ─────────────────────────────────────────────────────────────
user_inputs = {
    "revenue_growth":    revenue_growth,
    "debt_to_equity":    debt_to_equity,
    "return_on_assets":  return_on_assets,
    "current_ratio":     current_ratio,
    "market_volatility": market_volatility,
    "stock_return":      stock_return,
    "esg_score":         esg_score,
}
input_array = np.array([[user_inputs[f] for f in FEATURES]])


# ═════════════════════════════════════════════════════════════
# RESULTS  (only shown after form submission)
# ═════════════════════════════════════════════════════════════
if submitted:

    # ── Prediction ────────────────────────────────────────────
    if models_loaded:
        risk_label, confidence, probas = predict(models, input_array, model_choice)
        if risk_label is None:
            st.error("Prediction failed. The selected model file may be corrupted or incompatible.")
            st.stop()
    else:
        # Rule-based demo mode (no model files required)
        if esg_score > 65 and debt_to_equity < 1.5:
            risk_label, confidence = "Low",    0.78
            probas = {"High": 0.05, "Low": 0.78, "Medium": 0.17}
        elif esg_score > 50:
            risk_label, confidence = "Medium", 0.64
            probas = {"High": 0.15, "Low": 0.21, "Medium": 0.64}
        else:
            risk_label, confidence = "High",   0.72
            probas = {"High": 0.72, "Low": 0.08, "Medium": 0.20}

    risk_lower = risk_label.strip().lower()
    RISK_BG    = {"high": "#8b1c1c", "medium": "#7a5500", "low": "#0d4a20"}
    RISK_TEXT  = {"high": "#fff0f0", "medium": "#fff8e0", "low": "#e8fff0"}
    RISK_BORDER = {"high": "#ff6060", "medium": "#ffc000", "low": "#40c060"}
    risk_bg  = RISK_BG.get(risk_lower, "#1a1a1a")
    risk_fg  = RISK_TEXT.get(risk_lower, "#ffffff")
    risk_bdr = RISK_BORDER.get(risk_lower, "#888888")

    st.markdown("---")

    # ── 5. Prediction Summary ─────────────────────────────────
    with st.container():
        section_heading("Prediction Summary")

        pc1, pc2, pc3 = st.columns([2.5, 1.2, 1.2])
        with pc1:
            st.markdown(
                f"<div style='background:{risk_bg}; color:{risk_fg}; "
                f"padding:0.8rem 1.8rem; display:inline-block; "
                f"border:2px solid {risk_bdr}; margin-bottom:0.5rem;'>"
                f"<span style='font-size:1.5rem; font-weight:700; letter-spacing:0.12em; "
                f"color:{risk_fg} !important;'>{risk_label.upper()} ESG RISK</span>"
                f"</div>"
                f"<div style='font-size:0.88rem; color:#555555; margin-top:0.5rem;'>"
                f"Prediction by <strong>{model_choice}</strong></div>",
                unsafe_allow_html=True,
            )
        with pc2:
            st.metric("Confidence Score", f"{confidence * 100:.1f}%")
        with pc3:
            st.metric("Model Used", model_choice.split()[0])

        # Probability distribution — native DataFrame
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.85rem; font-weight:700; color:#0d1b2a; "
            "margin-bottom:0.4rem;'>Probability Distribution</div>",
            unsafe_allow_html=True,
        )
        prob_df = pd.DataFrame([
            {"Risk Class": cls, "Probability (%)": f"{prob * 100:.2f}%"}
            for cls, prob in sorted(probas.items())
        ])
        st.dataframe(prob_df, hide_index=True, use_container_width=False)

    st.markdown("---")

    # ── 6. Benchmark Comparison (native st.dataframe — NO HTML) ──
    with st.container():
        section_heading("Benchmark Comparison")

        bench_rows = []
        for feat, meta in BENCHMARKS.items():
            val    = user_inputs[feat]
            passed = meta["check"](val)
            bench_rows.append({
                "Metric":       meta["label"],
                "Actual Value": f"{val:.4f}",
                "Benchmark":    meta["target"],
                "Status":       "Pass" if passed else "Fail",
                "_pass":        passed,
            })

        # Build clean DataFrame for display (drop internal _pass column)
        bench_display = pd.DataFrame([
            {
                "Metric":       r["Metric"],
                "Actual Value": r["Actual Value"],
                "Benchmark":    r["Benchmark"],
                "Status":       r["Status"],
            }
            for r in bench_rows
        ])

        st.dataframe(
            bench_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Metric":       st.column_config.TextColumn("Metric",       width="medium"),
                "Actual Value": st.column_config.TextColumn("Actual Value", width="small"),
                "Benchmark":    st.column_config.TextColumn("Benchmark",    width="small"),
                "Status":       st.column_config.TextColumn("Status",       width="small"),
            },
        )

        # Pass / Fail summary counts below the table
        pass_count = sum(1 for r in bench_rows if r["_pass"])
        fail_count = len(bench_rows) - pass_count
        sc1, sc2, _, _ = st.columns(4)
        with sc1:
            st.metric("Benchmarks Passed", f"{pass_count} / {len(bench_rows)}")
        with sc2:
            st.metric("Benchmarks Failed", str(fail_count))

    st.markdown("---")

    # ── 7. Decision Support Recommendations ───────────────────
    with st.container():
        recs = get_recommendations(risk_label, user_inputs)

        section_heading("Decision Support Recommendations")

        rec_label_map = {
            "high":   ("HIGH RISK — Immediate Corrective Action Required", "#8b1c1c", "#fff8f8"),
            "medium": ("MEDIUM RISK — Improvement Actions Recommended",    "#7a5500", "#fffbf0"),
            "low":    ("LOW RISK — Strategic Advantages Available",        "#0d4a20", "#f0fff4"),
        }
        rec_title, rec_title_color, rec_bg = rec_label_map.get(
            risk_lower, ("ASSESSMENT", "#1a1a1a", "#fafafa")
        )

        st.markdown(
            f"<div style='background:{rec_bg}; border:1px solid #d0c8ba; "
            f"border-left:5px solid {risk_bdr}; padding:0.6rem 1.4rem 0.2rem 1.4rem; "
            f"margin-bottom:0.5rem;'>"
            f"<div style='font-size:0.8rem; font-weight:700; text-transform:uppercase; "
            f"letter-spacing:0.1em; color:{rec_title_color} !important; "
            f"margin-bottom:0.6rem; border-bottom:1px solid #d0c8ba; padding-bottom:0.4rem;'>"
            f"{rec_title}</div></div>",
            unsafe_allow_html=True,
        )

        for i, rec in enumerate(recs, 1):
            st.markdown(f"**{i}.** {rec}")

    st.markdown("---")

    # ── 8. Feature Importance ─────────────────────────────────
    with st.container():
        section_heading("Feature Importance and Model Explanation")

        fi_df = get_feature_importance(models, model_choice)
        if not fi_df.empty:
            score_col = ("Coefficient Magnitude" if model_choice == "Logistic Regression"
                         else "Importance Score")
            exp_text = (
                "Coefficient magnitudes (mean absolute value across classes) indicate each feature's "
                "influence on the predicted risk level. A higher magnitude implies a stronger contribution."
                if model_choice == "Logistic Regression" else
                "Feature importance scores represent each variable's relative predictive contribution "
                "within the Random Forest ensemble. Scores sum to 1.0 across all features."
            )
            st.caption(exp_text)

            disp_cols = (
                ["Feature", "Coefficient Magnitude", "Effect on Risk"]
                if model_choice == "Logistic Regression"
                else ["Feature", "Importance Score", "Importance Score (%)"]
            )
            fi_show = fi_df[disp_cols].copy()
            if "Coefficient Magnitude" in fi_show.columns:
                fi_show["Coefficient Magnitude"] = fi_show["Coefficient Magnitude"].map("{:.6f}".format)
            if "Importance Score" in fi_show.columns:
                fi_show["Importance Score"] = fi_show["Importance Score"].map("{:.6f}".format)
            st.dataframe(fi_show, use_container_width=True)

            # Matplotlib horizontal bar chart — explicit white bg + black labels
            chart_data = fi_df.set_index("Feature")[[score_col]].rename(columns={score_col: "Score"})
            feat_names = list(chart_data.index)
            feat_vals  = list(chart_data["Score"])
            fig, ax = plt.subplots(figsize=(6, max(3, len(feat_names) * 0.55)),
                                   facecolor="white")
            ax.barh(feat_names, feat_vals, color="#0B3C5D")
            ax.set_title(
                "Coefficient Magnitude" if model_choice == "Logistic Regression"
                else "Feature Importance Score",
                color="black", fontsize=11,
            )
            ax.set_xlabel("Score", color="black")
            ax.set_ylabel("Feature", color="black")
            ax.tick_params(colors="black")
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Feature importance data is unavailable for the selected model configuration.")

    st.markdown("---")

    # ── 9. Report Generation ──────────────────────────────────
    with st.container():
        section_heading("Report Generation")

        st.caption(
            "Download a structured ESG Risk Assessment Report containing all inputs, "
            "prediction results, benchmark comparison, and recommendations."
        )

        txt_report = generate_txt_report(
            user_inputs, model_choice, risk_label, confidence, probas, recs, bench_rows
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="Download Report (.txt)",
                data=txt_report,
                file_name=f"ESG_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

        with dl2:
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import cm
                from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                                Spacer, Table, TableStyle)
                from reportlab.lib import colors as rl_colors

                def build_pdf():
                    buf = io.BytesIO()
                    doc = SimpleDocTemplate(buf, pagesize=A4,
                                            leftMargin=2.2*cm, rightMargin=2.2*cm,
                                            topMargin=2.5*cm, bottomMargin=2.5*cm)
                    styles = getSampleStyleSheet()
                    S = lambda name, **kw: ParagraphStyle(name, parent=styles["Normal"], **kw)
                    title_s   = S("T", fontName="Times-Bold",   fontSize=17,
                                   textColor=rl_colors.HexColor("#0d1b2a"), spaceAfter=4)
                    sub_s     = S("S", fontName="Times-Italic", fontSize=9,
                                   textColor=rl_colors.HexColor("#888888"), spaceAfter=12)
                    heading_s = S("H", fontName="Times-Bold",   fontSize=11,
                                   textColor=rl_colors.HexColor("#0d1b2a"), spaceBefore=12, spaceAfter=4)
                    body_s    = S("B", fontName="Times-Roman",  fontSize=10,
                                   leading=15, spaceAfter=3)
                    rec_s     = S("R", fontName="Times-Roman",  fontSize=9.5,
                                   leading=14, leftIndent=12, spaceAfter=5)
                    disc_s    = S("D", fontName="Times-Italic", fontSize=8,
                                   textColor=rl_colors.grey, spaceBefore=16)
                    story = [
                        Paragraph("ESG Risk Assessment Report", title_s),
                        Paragraph(
                            f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')} "
                            f"| Model: {model_choice}", sub_s
                        ),
                        Paragraph("1. Input Parameters", heading_s),
                    ]
                    for k in FEATURES:
                        story.append(Paragraph(f"{FEATURE_LABELS[k]}:  {user_inputs[k]:.4f}", body_s))
                    story += [
                        Paragraph("2. Prediction Results", heading_s),
                        Paragraph(f"Predicted Risk Level:  <b>{risk_label.upper()}</b>", body_s),
                        Paragraph(f"Confidence Score:  {confidence*100:.2f}%", body_s),
                        Paragraph("Probability Distribution:", body_s),
                    ]
                    for cls, prob in sorted(probas.items()):
                        story.append(Paragraph(f"   {cls}:  {prob*100:.2f}%", body_s))
                    story.append(Paragraph("3. Benchmark Comparison", heading_s))
                    t_data = [["Metric", "Actual Value", "Benchmark", "Status"]]
                    for row in bench_rows:
                        t_data.append([row["Metric"], row["Actual Value"],
                                        row["Benchmark"], row["Status"]])
                    t = Table(t_data, colWidths=[5.5*cm, 2.8*cm, 3.2*cm, 2*cm])
                    t.setStyle(TableStyle([
                        ("BACKGROUND",     (0,0), (-1,0), rl_colors.HexColor("#0d1b2a")),
                        ("TEXTCOLOR",      (0,0), (-1,0), rl_colors.HexColor("#e8dcc8")),
                        ("FONTNAME",       (0,0), (-1,-1), "Times-Roman"),
                        ("FONTSIZE",       (0,0), (-1,-1), 9),
                        ("GRID",           (0,0), (-1,-1), 0.4, rl_colors.HexColor("#cccccc")),
                        ("ROWBACKGROUNDS", (0,1), (-1,-1),
                         [rl_colors.white, rl_colors.HexColor("#f5f3ee")]),
                    ]))
                    story += [t, Paragraph("4. Recommendations", heading_s)]
                    for i, rec in enumerate(recs, 1):
                        story.append(Paragraph(f"{i}.  {rec}", rec_s))
                    story.append(Paragraph(
                        "Disclaimer: This report is produced by a machine learning model and is "
                        "intended for analytical purposes only. It does not constitute financial, "
                        "legal, or investment advice.", disc_s))
                    doc.build(story)
                    buf.seek(0)
                    return buf.read()

                pdf_bytes = build_pdf()
                st.download_button(
                    label="Download Report (.pdf)",
                    data=pdf_bytes,
                    file_name=f"ESG_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            except ImportError:
                st.caption("PDF export requires ReportLab: pip install reportlab")

    st.markdown("---")

    # ── Footer ────────────────────────────────────────────────
    st.caption(
        "ESG Risk Intelligence Platform — For analytical use only. "
        "Predictions do not constitute financial, legal, or investment advice."
    )


# ═════════════════════════════════════════════════════════════
# IDLE STATE  (before any submission)
# ═════════════════════════════════════════════════════════════
else:
    st.info(
        "**How to use this platform:** "
        "Enter the financial and sustainability indicators in the form above. "
        "Select a prediction model from the left sidebar. "
        "Click *Run ESG Risk Assessment* to generate a full analysis including "
        "risk prediction, benchmark comparison, feature importance ranking, "
        "personalised recommendations, and a downloadable report."
    )

    st.markdown("---")
    section_heading("Input Reference Guide")

    ref_data = [
        {"Indicator": "Revenue Growth (%)",   "Description": "Year-over-year revenue growth rate",              "Benchmark": "Above 8% is strong"},
        {"Indicator": "Debt-to-Equity Ratio", "Description": "Total debt divided by shareholder equity",        "Benchmark": "Below 1.5 is preferred"},
        {"Indicator": "Return on Assets (%)", "Description": "Net income as a percentage of total assets",      "Benchmark": "Above 5% is target"},
        {"Indicator": "Current Ratio",        "Description": "Current assets divided by current liabilities",   "Benchmark": "Between 1.5 and 2.5"},
        {"Indicator": "Market Volatility",    "Description": "Annualised standard deviation of market returns", "Benchmark": "Lower is better"},
        {"Indicator": "Stock Return (%)",     "Description": "Annualised stock price return",                   "Benchmark": "Positive and stable"},
        {"Indicator": "ESG Score (0 – 100)",  "Description": "Composite environmental, social, governance score","Benchmark": "Above 60 is preferred"},
    ]
    st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)
