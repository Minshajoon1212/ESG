"""
ESG Risk Prediction Application
Professional analytics tool for ESG risk assessment using pre-trained ML models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import io
from datetime import datetime

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Risk Intelligence Platform",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Professional consulting aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Times New Roman', Times, serif !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #f5f3ee !important;
    border-right: 1px solid #d0c8ba !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Times New Roman', Times, serif !important;
    color: #1a1a1a !important;
}

/* ── Main header banner ── */
.header-banner {
    background-color: #0d1b2a;
    color: #e8dcc8;
    padding: 2.5rem 3rem 2rem 3rem;
    margin-bottom: 2rem;
    border-bottom: 3px solid #c8a96e;
}
.header-banner h1 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #e8dcc8;
    margin: 0;
}
.header-banner p {
    font-family: 'Times New Roman', Times, serif;
    font-size: 0.95rem;
    color: #a89880;
    margin: 0.4rem 0 0 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Times New Roman', Times, serif;
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #0d1b2a;
    border-bottom: 1.5px solid #c8a96e;
    padding-bottom: 0.4rem;
    margin: 2rem 0 1.2rem 0;
}

/* ── Cards ── */
.card {
    background: #fafaf8;
    border: 1px solid #e0d8cc;
    border-left: 4px solid #c8a96e;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.card-dark {
    background: #0d1b2a;
    border: 1px solid #2a3a4a;
    border-left: 4px solid #c8a96e;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}

/* ── Risk badges ── */
.risk-high {
    background-color: #7b1c1c;
    color: #fff8f0;
    padding: 0.6rem 1.6rem;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    display: inline-block;
    text-transform: uppercase;
}
.risk-medium {
    background-color: #5a4000;
    color: #fff8e0;
    padding: 0.6rem 1.6rem;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    display: inline-block;
    text-transform: uppercase;
}
.risk-low {
    background-color: #0d3320;
    color: #e0ffe8;
    padding: 0.6rem 1.6rem;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    display: inline-block;
    text-transform: uppercase;
}

/* ── Recommendation blocks ── */
.rec-item {
    padding: 0.55rem 0;
    border-bottom: 1px dotted #d0c8ba;
    font-size: 0.95rem;
    line-height: 1.65;
}
.rec-item:last-child { border-bottom: none; }

/* ── Benchmark table ── */
.bench-pass {
    color: #1a5c2a;
    font-weight: 700;
}
.bench-fail {
    color: #7b1c1c;
    font-weight: 700;
}

/* ── Streamlit overrides ── */
.stButton > button {
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    background-color: #0d1b2a !important;
    color: #e8dcc8 !important;
    border: 1.5px solid #c8a96e !important;
    padding: 0.5rem 1.6rem !important;
    border-radius: 0 !important;
    transition: background-color 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #1e3a5a !important;
}
.stDownloadButton > button {
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 0.9rem !important;
    background-color: #2a4a2a !important;
    color: #e0ffe8 !important;
    border: 1.5px solid #4a8a4a !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.4rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-family: 'Times New Roman', Times, serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    color: #2a2a2a !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #0d1b2a !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
.stAlert {
    font-family: 'Times New Roman', Times, serif !important;
    border-radius: 0 !important;
}
table {
    font-family: 'Times New Roman', Times, serif !important;
    width: 100% !important;
}
thead tr th {
    background-color: #0d1b2a !important;
    color: #e8dcc8 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 0.8rem !important;
}
tbody tr:nth-child(even) { background-color: #f5f3ee !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all pre-trained model artifacts from disk."""
    models = {}
    errors = []

    artifacts = {
        "logistic": "logistic_model.pkl",
        "random_forest": "random_forest_model.pkl",
        "scaler": "scaler.pkl",
        "label_encoder": "label_encoder.pkl",
    }

    for key, filename in artifacts.items():
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    models[key] = pickle.load(f)
            except Exception as e:
                errors.append(f"Failed to load {filename}: {e}")
        else:
            errors.append(f"File not found: {filename}")

    return models, errors


# ─────────────────────────────────────────────
# Feature Definitions
# ─────────────────────────────────────────────
FEATURES = [
    "revenue_growth",
    "debt_to_equity",
    "return_on_assets",
    "current_ratio",
    "market_volatility",
    "stock_return",
    "esg_score",
]

FEATURE_LABELS = {
    "revenue_growth": "Revenue Growth (%)",
    "debt_to_equity": "Debt-to-Equity Ratio",
    "return_on_assets": "Return on Assets — ROA (%)",
    "current_ratio": "Current Ratio",
    "market_volatility": "Market Volatility",
    "stock_return": "Stock Return (%)",
    "esg_score": "ESG Score (0–100)",
}

BENCHMARKS = {
    "debt_to_equity":   {"label": "Debt-to-Equity",   "target": "< 1.5",   "check": lambda v: v < 1.5},
    "return_on_assets": {"label": "Return on Assets",  "target": "> 5%",    "check": lambda v: v > 5},
    "current_ratio":    {"label": "Current Ratio",     "target": "1.5 – 2.5","check": lambda v: 1.5 <= v <= 2.5},
    "esg_score":        {"label": "ESG Score",         "target": "> 60",    "check": lambda v: v > 60},
}

DEFAULTS = {
    "revenue_growth": 5.0,
    "debt_to_equity": 1.8,
    "return_on_assets": 3.5,
    "current_ratio": 1.2,
    "market_volatility": 0.25,
    "stock_return": 4.0,
    "esg_score": 50.0,
}


# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
def predict(models: dict, inputs: np.ndarray, model_choice: str):
    """Run prediction and return (label, confidence, probabilities)."""
    if model_choice == "Logistic Regression":
        model = models.get("logistic")
        scaler = models.get("scaler")
        if model is None:
            return None, None, None
        X = scaler.transform(inputs) if scaler else inputs
    else:
        model = models.get("random_forest")
        if model is None:
            return None, None, None
        X = inputs

    le = models.get("label_encoder")
    raw_pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = le.inverse_transform([raw_pred])[0] if le else str(raw_pred)
    confidence = float(np.max(proba))
    classes = le.classes_ if le else model.classes_

    return label, confidence, dict(zip(classes, proba))


# ─────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────
def get_recommendations(risk_level: str, user_inputs: dict) -> list:
    risk = risk_level.strip().lower()

    if risk == "high":
        return [
            "Reduce Debt-to-Equity ratio to below 1.5 through debt restructuring, equity issuance, "
            f"or asset divestitures. Current value: {user_inputs['debt_to_equity']:.2f}.",
            "Improve Return on Assets above 5% by enhancing operational margins, reducing cost of goods, "
            f"and optimising asset utilisation. Current ROA: {user_inputs['return_on_assets']:.2f}%.",
            "Strengthen the Current Ratio to above 1.5 by accelerating receivables collection, "
            f"renegotiating payables, or securing short-term credit facilities. Current ratio: {user_inputs['current_ratio']:.2f}.",
            "Raise the ESG Score above 60 by initiating formal sustainability disclosures, "
            "establishing science-based emissions targets, and implementing governance reforms. "
            f"Current score: {user_inputs['esg_score']:.1f}.",
            "Mitigate market volatility exposure through derivatives hedging, portfolio diversification, "
            f"and active duration management. Current volatility: {user_inputs['market_volatility']:.3f}.",
            "Accelerate revenue growth above 8% through geographic expansion, product line extension, "
            f"and strategic partnerships. Current growth: {user_inputs['revenue_growth']:.2f}%.",
        ]
    elif risk == "medium":
        return [
            "Gradually reduce financial leverage toward a Debt-to-Equity ratio below 1.2 over the "
            f"next 12–18 months. Current value: {user_inputs['debt_to_equity']:.2f}.",
            "Target an ESG Score above 70 by formalising ESG reporting frameworks (GRI/SASB) "
            f"and engaging third-party verification. Current score: {user_inputs['esg_score']:.1f}.",
            "Stabilise core performance indicators — ROA, Current Ratio, and Stock Return — "
            "through disciplined capital allocation and working capital optimisation.",
            "Strengthen ESG disclosures and investor communications to reduce perceived governance risk "
            "and improve access to ESG-linked capital instruments.",
        ]
    else:  # low
        return [
            "The company's ESG profile is attractive to institutional investors, particularly ESG-mandate "
            "funds, which may broaden the shareholder base and support premium valuation multiples.",
            "A low-risk ESG rating typically correlates with lower cost of capital — both debt and equity — "
            "reflecting reduced regulatory and reputational risk premia.",
            "The organisation is well-positioned to access ESG-linked financing instruments such as "
            "green bonds, sustainability-linked loans, and impact credit facilities.",
            "Maintain current ESG positioning through continuous improvement in sustainability practices, "
            "supporting long-term competitive differentiation and regulatory resilience.",
        ]


# ─────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────
def get_feature_importance(models: dict, model_choice: str) -> pd.DataFrame:
    if model_choice == "Logistic Regression":
        model = models.get("logistic")
        if model is None or not hasattr(model, "coef_"):
            return pd.DataFrame()
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.abs(coef)
        df = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Coefficient Magnitude": importance,
            "Direction": ["Positive" if c > 0 else "Negative"
                          for c in (coef[0] if coef.ndim > 1 else coef)],
        }).sort_values("Coefficient Magnitude", ascending=False).reset_index(drop=True)
        df.index += 1
        return df
    else:
        model = models.get("random_forest")
        if model is None or not hasattr(model, "feature_importances_"):
            return pd.DataFrame()
        df = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Importance Score": model.feature_importances_,
        }).sort_values("Importance Score", ascending=False).reset_index(drop=True)
        df.index += 1
        return df


# ─────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────
def generate_txt_report(user_inputs, model_choice, risk_label, confidence, probas, recs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "ESG RISK ASSESSMENT REPORT",
        f"Generated: {ts}",
        "=" * 70,
        "",
        "SECTION 1: INPUT PARAMETERS",
        "-" * 40,
    ]
    for k, v in user_inputs.items():
        lines.append(f"  {FEATURE_LABELS[k]:<40} {v:.4f}")
    lines += [
        "",
        "SECTION 2: MODEL & PREDICTION",
        "-" * 40,
        f"  Model Selected        : {model_choice}",
        f"  Predicted Risk Level  : {risk_label.upper()}",
        f"  Confidence Score      : {confidence*100:.2f}%",
        "",
        "  Probability Distribution:",
    ]
    for cls, prob in probas.items():
        lines.append(f"    {cls:<12} {prob*100:.2f}%")
    lines += [
        "",
        "SECTION 3: BENCHMARK COMPARISON",
        "-" * 40,
    ]
    for feat, meta in BENCHMARKS.items():
        val = user_inputs[feat]
        status = "PASS" if meta["check"](val) else "FAIL"
        lines.append(f"  {meta['label']:<22} Value: {val:.4f}   Target: {meta['target']:<12} [{status}]")
    lines += [
        "",
        "SECTION 4: RECOMMENDATIONS",
        "-" * 40,
    ]
    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")
        lines.append("")
    lines += [
        "=" * 70,
        "DISCLAIMER: This report is generated by a machine learning model and",
        "should be used for analytical purposes only. It does not constitute",
        "financial, legal, or investment advice.",
        "=" * 70,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>ESG Risk Intelligence Platform</h1>
    <p>Quantitative ESG Risk Assessment and Decision Support System</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────
with st.spinner("Loading model artifacts..."):
    models, load_errors = load_models()

if load_errors:
    st.warning("One or more model files could not be loaded. Running in demo mode with simulated predictions.")
    for err in load_errors:
        st.caption(f"  - {err}")

models_loaded = bool(models)


# ─────────────────────────────────────────────
# Sidebar — Model Selection
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-heading">Model Configuration</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Prediction Model",
        ["Logistic Regression", "Random Forest"],
        index=0,
        help="Logistic Regression applies feature scaling via the pre-trained scaler. Random Forest operates on raw inputs.",
    )

    st.markdown('<div class="section-heading">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; line-height:1.7; color:#3a3a3a;">
    This platform applies pre-trained machine learning models to assess corporate ESG risk levels
    based on financial and sustainability indicators.<br><br>
    <strong>Models:</strong><br>
    &mdash; Logistic Regression (with scaling)<br>
    &mdash; Random Forest<br><br>
    <strong>Risk Levels:</strong><br>
    &mdash; Low &mdash; Medium &mdash; High
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Model Status</div>', unsafe_allow_html=True)
    for key, label in [("logistic", "Logistic Regression"), ("random_forest", "Random Forest"),
                        ("scaler", "Feature Scaler"), ("label_encoder", "Label Encoder")]:
        status = "Loaded" if key in models else "Not Found"
        colour = "#1a5c2a" if key in models else "#7b1c1c"
        st.markdown(f'<div style="font-size:0.82rem; color:{colour}; margin-bottom:4px;">'
                    f'{"[OK]" if key in models else "[--]"} {label}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Layout: Input Form
# ─────────────────────────────────────────────
st.markdown('<div class="section-heading">Financial and ESG Input Parameters</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**Financial Indicators**")
        revenue_growth = st.number_input(
            FEATURE_LABELS["revenue_growth"], value=DEFAULTS["revenue_growth"],
            step=0.1, format="%.2f", help="Year-over-year revenue growth percentage")
        debt_to_equity = st.number_input(
            FEATURE_LABELS["debt_to_equity"], value=DEFAULTS["debt_to_equity"],
            min_value=0.0, step=0.05, format="%.2f", help="Total debt divided by total shareholder equity")
        return_on_assets = st.number_input(
            FEATURE_LABELS["return_on_assets"], value=DEFAULTS["return_on_assets"],
            step=0.1, format="%.2f", help="Net income as a percentage of total assets")
        current_ratio = st.number_input(
            FEATURE_LABELS["current_ratio"], value=DEFAULTS["current_ratio"],
            min_value=0.0, step=0.05, format="%.2f", help="Current assets divided by current liabilities")

    with col2:
        st.markdown("**Market and Sustainability Indicators**")
        market_volatility = st.number_input(
            FEATURE_LABELS["market_volatility"], value=DEFAULTS["market_volatility"],
            min_value=0.0, step=0.01, format="%.3f", help="Standard deviation of market returns (annualised)")
        stock_return = st.number_input(
            FEATURE_LABELS["stock_return"], value=DEFAULTS["stock_return"],
            step=0.1, format="%.2f", help="Annualised stock return percentage")
        esg_score = st.number_input(
            FEATURE_LABELS["esg_score"], value=DEFAULTS["esg_score"],
            min_value=0.0, max_value=100.0, step=0.5, format="%.1f",
            help="Composite ESG score on a scale of 0 to 100")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Run ESG Risk Assessment", use_container_width=False)


# ─────────────────────────────────────────────
# Collect Inputs
# ─────────────────────────────────────────────
user_inputs = {
    "revenue_growth": revenue_growth,
    "debt_to_equity": debt_to_equity,
    "return_on_assets": return_on_assets,
    "current_ratio": current_ratio,
    "market_volatility": market_volatility,
    "stock_return": stock_return,
    "esg_score": esg_score,
}
input_array = np.array([[user_inputs[f] for f in FEATURES]])


# ─────────────────────────────────────────────
# Results Section
# ─────────────────────────────────────────────
if submitted:

    # ── Prediction ──────────────────────────
    if models_loaded:
        risk_label, confidence, probas = predict(models, input_array, model_choice)
    else:
        # Demo mode: deterministic simulation
        score = (esg_score / 100) * 0.4 + (min(return_on_assets, 10) / 10) * 0.3 + \
                (max(0, 10 - debt_to_equity * 5) / 10) * 0.3
        if score > 0.6:
            risk_label, confidence = "Low", 0.78
            probas = {"Low": 0.78, "Medium": 0.17, "High": 0.05}
        elif score > 0.35:
            risk_label, confidence = "Medium", 0.65
            probas = {"Low": 0.20, "Medium": 0.65, "High": 0.15}
        else:
            risk_label, confidence = "High", 0.72
            probas = {"Low": 0.08, "Medium": 0.20, "High": 0.72}

    if risk_label is None:
        st.error("Prediction failed. Please verify that all model files are present and correctly formatted.")
        st.stop()

    risk_lower = risk_label.strip().lower()
    badge_class = {"high": "risk-high", "medium": "risk-medium", "low": "risk-low"}.get(risk_lower, "risk-low")

    # ── Prediction Summary ───────────────────
    st.markdown('<div class="section-heading">Prediction Summary</div>', unsafe_allow_html=True)

    pcol1, pcol2, pcol3 = st.columns([2, 1, 1])
    with pcol1:
        st.markdown(f'<span class="{badge_class}">{risk_label.upper()} RISK</span>', unsafe_allow_html=True)
        st.markdown(f"<p style='margin-top:0.8rem; font-size:0.9rem; color:#555;'>"
                    f"Prediction generated by <strong>{model_choice}</strong></p>", unsafe_allow_html=True)
    with pcol2:
        st.metric("Confidence Score", f"{confidence*100:.1f}%")
    with pcol3:
        st.metric("Model", model_choice.split()[0])

    # Probability breakdown
    st.markdown("**Probability Distribution**")
    prob_df = pd.DataFrame({
        "Risk Level": list(probas.keys()),
        "Probability (%)": [f"{v*100:.2f}%" for v in probas.values()],
        "Probability": [round(v, 4) for v in probas.values()],
    })
    st.dataframe(prob_df[["Risk Level", "Probability (%)"]], hide_index=True, use_container_width=False)

    st.divider()

    # ── Recommendations ──────────────────────
    recs = get_recommendations(risk_label, user_inputs)
    st.markdown('<div class="section-heading">Decision Support Recommendations</div>', unsafe_allow_html=True)

    rec_html = ""
    for i, rec in enumerate(recs, 1):
        rec_html += f'<div class="rec-item"><strong>{i}.</strong>&nbsp; {rec}</div>'

    risk_colors = {"high": "#7b1c1c", "medium": "#5a4000", "low": "#0d3320"}
    risk_color = risk_colors.get(risk_lower, "#0d1b2a")

    st.markdown(f"""
    <div style="border-left: 4px solid {risk_color}; padding: 1rem 1.4rem;
                background: #fafaf8; border: 1px solid #e0d8cc; border-left: 4px solid {risk_color};">
    {rec_html}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Benchmark Comparison ─────────────────
    st.markdown('<div class="section-heading">Benchmark Comparison</div>', unsafe_allow_html=True)

    bench_rows = []
    for feat, meta in BENCHMARKS.items():
        val = user_inputs[feat]
        passed = meta["check"](val)
        bench_rows.append({
            "Indicator": meta["label"],
            "Your Value": f"{val:.4f}",
            "Target Benchmark": meta["target"],
            "Status": "Pass" if passed else "Fail",
        })
    bench_df = pd.DataFrame(bench_rows)

    def highlight_status(row):
        color = "#d4edda" if row["Status"] == "Pass" else "#f8d7da"
        return [""] * (len(row) - 1) + [f"background-color: {color}; font-weight: bold;"]

    st.dataframe(
        bench_df.style.apply(highlight_status, axis=1),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    # ── Feature Importance ───────────────────
    st.markdown('<div class="section-heading">Feature Importance and Model Explanation</div>', unsafe_allow_html=True)

    fi_df = get_feature_importance(models, model_choice)
    if not fi_df.empty:
        if model_choice == "Logistic Regression":
            st.markdown("Coefficient magnitudes indicate each feature's influence on the predicted risk level. "
                        "Larger magnitude implies stronger contribution.")
            score_col = "Coefficient Magnitude"
        else:
            st.markdown("Feature importance scores from the Random Forest ensemble indicate the relative "
                        "predictive contribution of each input variable.")
            score_col = "Importance Score"

        fi_display = fi_df.copy()
        fi_display[score_col] = fi_display[score_col].map("{:.6f}".format)
        st.dataframe(fi_display, use_container_width=True)

        # Bar chart (using st.bar_chart with renamed column)
        chart_df = fi_df.set_index("Feature")[[score_col]].rename(columns={score_col: "Score"})
        st.bar_chart(chart_df)
    else:
        st.info("Feature importance data is unavailable for the selected model configuration.")

    st.divider()

    # ── Report Generation ────────────────────
    st.markdown('<div class="section-heading">Report Generation</div>', unsafe_allow_html=True)
    st.markdown("Generate a structured ESG Risk Assessment Report for download.")

    rcol1, rcol2 = st.columns(2)

    txt_report = generate_txt_report(user_inputs, model_choice, risk_label, confidence, probas, recs)

    with rcol1:
        st.download_button(
            label="Download Report (.txt)",
            data=txt_report,
            file_name=f"ESG_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    with rcol2:
        # PDF generation via reportlab if available, else skip
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib import colors

            def generate_pdf_report():
                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4,
                                        leftMargin=2.5*cm, rightMargin=2.5*cm,
                                        topMargin=2.5*cm, bottomMargin=2.5*cm)
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                             fontName="Times-Bold", fontSize=16,
                                             textColor=colors.HexColor("#0d1b2a"), spaceAfter=6)
                heading_style = ParagraphStyle("heading", parent=styles["Heading2"],
                                               fontName="Times-Bold", fontSize=11,
                                               textColor=colors.HexColor("#0d1b2a"), spaceAfter=4)
                body_style = ParagraphStyle("body", parent=styles["Normal"],
                                            fontName="Times-Roman", fontSize=10,
                                            leading=15, spaceAfter=4)
                story = []
                story.append(Paragraph("ESG Risk Assessment Report", title_style))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
                story.append(Spacer(1, 0.4*cm))
                story.append(Paragraph("1. Input Parameters", heading_style))
                for k, v in user_inputs.items():
                    story.append(Paragraph(f"{FEATURE_LABELS[k]}: {v:.4f}", body_style))
                story.append(Spacer(1, 0.3*cm))
                story.append(Paragraph("2. Prediction Results", heading_style))
                story.append(Paragraph(f"Model: {model_choice}", body_style))
                story.append(Paragraph(f"Predicted Risk Level: {risk_label.upper()}", body_style))
                story.append(Paragraph(f"Confidence Score: {confidence*100:.2f}%", body_style))
                story.append(Spacer(1, 0.3*cm))
                story.append(Paragraph("3. Recommendations", heading_style))
                for i, rec in enumerate(recs, 1):
                    story.append(Paragraph(f"{i}. {rec}", body_style))
                story.append(Spacer(1, 0.3*cm))
                story.append(Paragraph("4. Benchmark Comparison", heading_style))
                t_data = [["Indicator", "Your Value", "Target", "Status"]]
                for feat, meta in BENCHMARKS.items():
                    val = user_inputs[feat]
                    status = "Pass" if meta["check"](val) else "Fail"
                    t_data.append([meta["label"], f"{val:.4f}", meta["target"], status])
                t = Table(t_data, colWidths=[5*cm, 3*cm, 3*cm, 2*cm])
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d1b2a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#e8dcc8")),
                    ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ee")]),
                ]))
                story.append(t)
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph(
                    "Disclaimer: This report is generated by a machine learning model and is intended "
                    "for analytical purposes only. It does not constitute financial or investment advice.",
                    ParagraphStyle("disclaimer", parent=styles["Normal"], fontName="Times-Italic",
                                   fontSize=8, textColor=colors.grey)))
                doc.build(story)
                buf.seek(0)
                return buf.read()

            pdf_bytes = generate_pdf_report()
            st.download_button(
                label="Download Report (.pdf)",
                data=pdf_bytes,
                file_name=f"ESG_Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.caption("PDF export requires ReportLab. Install via: pip install reportlab")

    st.divider()

    # ── Footer ───────────────────────────────
    st.markdown("""
    <div style="font-size:0.8rem; color:#888; margin-top:2rem; padding-top:1rem;
                border-top: 1px solid #e0d8cc; text-align: center; font-style: italic;">
    ESG Risk Intelligence Platform &mdash; For analytical use only.
    Predictions do not constitute financial, legal, or investment advice.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Idle State ───────────────────────────
    st.markdown("""
    <div style="background:#f5f3ee; border:1px solid #d0c8ba; border-left:4px solid #c8a96e;
                padding: 1.4rem 1.8rem; margin-top:1rem;">
    <strong>Instructions:</strong> Enter the financial and ESG indicators in the form above,
    select a prediction model from the sidebar, and click <em>Run ESG Risk Assessment</em> to
    generate the full analysis including risk prediction, benchmark comparison, feature importance,
    and downloadable report.
    </div>
    """, unsafe_allow_html=True)
