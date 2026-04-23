# ESG Risk Intelligence Platform

A professional Streamlit application for ESG risk prediction using pre-trained machine learning models.

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your Model Files

Place the following `.pkl` files in the **same directory** as `app.py`:

| File | Description |
|------|-------------|
| `logistic_model.pkl` | Pre-trained Logistic Regression model |
| `random_forest_model.pkl` | Pre-trained Random Forest model |
| `scaler.pkl` | StandardScaler (used only for Logistic Regression) |
| `label_encoder.pkl` | LabelEncoder for risk level labels |

> **No model files?** Run `python generate_demo_models.py` to create synthetic demo models for testing.

### 3. Launch the Application

```bash
streamlit run app.py
```

---

## Application Features

| Feature | Description |
|---------|-------------|
| Model Selection | Choose between Logistic Regression and Random Forest |
| Input Form | Structured inputs for 7 financial/ESG indicators |
| Prediction Output | Risk level (Low / Medium / High) + confidence score |
| Decision Support | Metric-based professional recommendations per risk tier |
| Feature Importance | Coefficient table (LR) or importance ranking (RF) with bar chart |
| Benchmark Comparison | User values vs. industry-standard thresholds |
| Report Download | Structured report as `.txt` or `.pdf` (requires ReportLab) |

---

## Input Variables

| Variable | Description |
|----------|-------------|
| `revenue_growth` | Year-over-year revenue growth (%) |
| `debt_to_equity` | Total debt / shareholder equity |
| `return_on_assets` | Net income / total assets (%) |
| `current_ratio` | Current assets / current liabilities |
| `market_volatility` | Annualised standard deviation of market returns |
| `stock_return` | Annualised stock return (%) |
| `esg_score` | Composite ESG score (0–100) |

---

## Risk Level Definitions

| Level | Meaning |
|-------|---------|
| **Low** | Strong financial and ESG fundamentals; attractive to institutional investors |
| **Medium** | Moderate risk; improvement recommended across select indicators |
| **High** | Elevated risk; immediate corrective action required across multiple dimensions |

---

## PDF Export

PDF report generation requires ReportLab:

```bash
pip install reportlab
```

---

## Project Structure

```
esg_risk_app/
├── app.py                    # Main Streamlit application
├── generate_demo_models.py   # Script to generate synthetic .pkl files
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── logistic_model.pkl        # (you provide)
├── random_forest_model.pkl   # (you provide)
├── scaler.pkl                # (you provide)
└── label_encoder.pkl         # (you provide)
```
