"""
generate_demo_models.py
───────────────────────
Run this script ONCE to create demo .pkl files for local testing.
These models are trained on synthetic ESG data.

Usage:
    python generate_demo_models.py
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_STATE = 42
N_SAMPLES = 1200

np.random.seed(RANDOM_STATE)

# ── Synthetic dataset ────────────────────────
X = np.column_stack([
    np.random.normal(5,  4,   N_SAMPLES),   # revenue_growth
    np.random.uniform(0.5, 4, N_SAMPLES),   # debt_to_equity
    np.random.normal(4,  3,   N_SAMPLES),   # return_on_assets
    np.random.uniform(0.5, 3, N_SAMPLES),   # current_ratio
    np.random.uniform(0.1, 0.6, N_SAMPLES), # market_volatility
    np.random.normal(5,  8,   N_SAMPLES),   # stock_return
    np.random.uniform(20, 90, N_SAMPLES),   # esg_score
])

# Risk label logic (0=High, 1=Low, 2=Medium)
scores = (
    (X[:, 6] / 100) * 0.35 +                   # esg_score
    np.clip(X[:, 2], 0, 10) / 10 * 0.25 +       # ROA
    (np.clip(2 - X[:, 1], 0, 2) / 2) * 0.20 +  # debt_to_equity (lower is better)
    np.clip(X[:, 3] - 0.5, 0, 2) / 2 * 0.20    # current_ratio
)
noise = np.random.normal(0, 0.05, N_SAMPLES)
scores += noise

y_raw = np.where(scores > 0.62, 1,           # Low
         np.where(scores > 0.38, 2, 0))       # Medium / High

le = LabelEncoder()
le.fit(["High", "Low", "Medium"])
y = le.transform(["High", "Low", "Medium"])[y_raw]

# ── Scaler ───────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Logistic Regression ──────────────────────
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, multi_class="auto")
lr.fit(X_scaled, y)

# ── Random Forest ────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
rf.fit(X, y)

# ── Save artifacts ───────────────────────────
for obj, fname in [
    (lr,     "logistic_model.pkl"),
    (rf,     "random_forest_model.pkl"),
    (scaler, "scaler.pkl"),
    (le,     "label_encoder.pkl"),
]:
    with open(fname, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {fname}")

print("\nAll model artifacts created. Run: streamlit run app.py")
