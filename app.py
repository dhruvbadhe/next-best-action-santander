import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA – Next Best Action",
    page_icon="🏦",
    layout="wide"
)

# ── Model definition (must match training exactly) ────────────────────────────
class NBAModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# ── Load artifacts (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("nba_feature_cols.json") as f:
        feature_cols = json.load(f)

    scaler = joblib.load("nba_scaler.pkl")

    device = torch.device("cpu")
    model = NBAModel(input_size=len(feature_cols), hidden_size=128, num_layers=2, output_size=10)
    state = torch.load("nba_model_final.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    target_cols = [
        "direct_debit", "credit_card", "current_account",
        "payroll_account", "e_account", "long_term_deposit",
        "taxes", "securities", "funds", "particular_account"
    ]

    return feature_cols, scaler, model, device, target_cols

feature_cols, scaler, model, device, target_cols = load_artifacts()

NUMERIC_COLS = ["age", "tenure_months", "gross_income"]

# ── Helper: build one feature row ─────────────────────────────────────────────
def build_feature_row(inputs: dict) -> np.ndarray:
    row = {col: 0.0 for col in feature_cols}

    # numeric (will be scaled later)
    row["age"]            = inputs["age"]
    row["tenure_months"]  = inputs["tenure_months"]
    row["gross_income"]   = inputs["gross_income"]

    # product flags
    for p in inputs["products_owned"]:
        if p in row:
            row[p] = 1.0

    # one-hot categoricals
    def set_ohe(prefix, val):
        key = f"{prefix}_{val}"
        if key in row:
            row[key] = 1.0

    set_ohe("employee_index",               inputs["employee_index"])
    set_ohe("country_residence",            inputs["country_residence"])
    set_ohe("gender",                       inputs["gender"])
    set_ohe("is_new_customer",              inputs["is_new_customer"])
    set_ohe("primary_customer_flag",        inputs["primary_customer_flag"])
    set_ohe("relationship_type_start_month",inputs["relationship_type"])
    set_ohe("customer_type_start_month",    inputs["customer_type"])
    set_ohe("is_foreigner",                 inputs["is_foreigner"])
    set_ohe("join_channel",                 inputs["join_channel"])
    set_ohe("is_deceased",                  "0")
    set_ohe("province_code",                inputs["province_code"])
    set_ohe("is_active_customer",           inputs["is_active"])
    set_ohe("customer_segment",             inputs["segment"])

    arr = np.array([row[c] for c in feature_cols], dtype=np.float32)
    return arr

# ── Helper: scale numeric cols in feature array ───────────────────────────────
def scale_row(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    idx = [feature_cols.index(c) for c in NUMERIC_COLS]
    numeric_vals = arr[idx].reshape(1, -1)
    arr[idx] = scaler.transform(numeric_vals)[0]
    return arr

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(monthly_rows: list) -> np.ndarray:
    """monthly_rows: list of 16 raw feature arrays (pre-scale)"""
    scaled = np.stack([scale_row(r) for r in monthly_rows])  # (16, 239)
    X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 16, 239)
    with torch.no_grad():
        logits = model(X)               # (1, 16, 10)
    probs = torch.sigmoid(logits)
    last_month_probs = probs[0, -1, :].numpy()  # predict for month 17
    return last_month_probs

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏦 Next Best Action — Santander Product Recommender")
st.markdown(
    "Enter 16 months of customer data. The LSTM predicts which products "
    "the customer is most likely to add in **month 17**."
)

st.divider()

# ── Sidebar: static customer profile ─────────────────────────────────────────
with st.sidebar:
    st.header("Customer Profile")
    age           = st.number_input("Age", 18, 100, 35)
    gross_income  = st.number_input("Gross Income (€)", 0, 500000, 80000, step=1000)
    gender        = st.selectbox("Gender", ["H", "V"], format_func=lambda x: "Male" if x=="H" else "Female")
    segment       = st.selectbox("Segment", [
        "01 - TOP", "02 - PARTICULARES", "03 - UNIVERSITARIO", "Unknown"
    ])
    employee_idx  = st.selectbox("Employee Index", ["N", "A", "B", "F"])
    country       = st.selectbox("Country Residence", ["ES", "Other"],
                                  format_func=lambda x: "Spain" if x=="ES" else "Other")
    country_val   = "ES" if country == "ES" else "OTHER"
    is_foreigner  = st.selectbox("Foreigner", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    join_channel  = st.selectbox("Join Channel", ["KAT", "KHE", "KFA", "KFC", "RED", "Unknown"])
    province_code = st.selectbox("Province Code", [str(float(i)) for i in range(1, 53)], index=27)
    is_active     = st.selectbox("Active Customer", ["1.0", "0.0"],
                                  format_func=lambda x: "Yes" if x=="1.0" else "No")
    rel_type      = st.selectbox("Relationship Type", ["A", "I", "P"])
    cust_type     = st.selectbox("Customer Type", ["1.0", "3.0"])
    is_new        = st.selectbox("New Customer", ["0.0", "1.0"],
                                  format_func=lambda x: "No" if x=="0.0" else "Yes")
    primary_flag  = st.selectbox("Primary Customer", ["1.0", "0.0"],
                                  format_func=lambda x: "Yes" if x=="1.0" else "No")

# ── Main: 16-month product ownership history ──────────────────────────────────
st.subheader("📅 Monthly Product Ownership (16 months)")
st.caption("Check which products the customer owned each month.")

ALL_PRODUCTS = [
    "prod_savings_account", "prod_guarantees", "prod_current_account",
    "prod_derivada_account", "prod_payroll_account", "prod_junior_account",
    "prod_mas_particular_account", "prod_particular_account",
    "prod_particular_plus_account", "prod_short_term_deposit",
    "prod_medium_term_deposit", "prod_long_term_deposit", "prod_e_account",
    "prod_funds", "prod_mortgage", "prod_pension_plan", "prod_loans",
    "prod_taxes", "prod_credit_card", "prod_securities", "prod_home_account",
    "prod_payroll", "prod_pension_nomina", "prod_direct_debit"
]

PRODUCT_LABELS = {p: p.replace("prod_", "").replace("_", " ").title() for p in ALL_PRODUCTS}

# Default: current account owned from month 1, rest added gradually
def default_owned(month_idx):
    base = ["prod_current_account"]
    if month_idx >= 3:  base.append("prod_direct_debit")
    if month_idx >= 6:  base.append("prod_savings_account")
    return base

monthly_product_data = []
tabs = st.tabs([f"Month {i+1}" for i in range(16)])

for i, tab in enumerate(tabs):
    with tab:
        defaults = default_owned(i)
        selected = st.multiselect(
            f"Products owned in Month {i+1}",
            options=ALL_PRODUCTS,
            default=defaults,
            format_func=lambda x: PRODUCT_LABELS[x],
            key=f"month_{i}"
        )
        tenure = st.slider(f"Tenure (months) — Month {i+1}", 0, 240, i+12, key=f"tenure_{i}")
        monthly_product_data.append({"products": selected, "tenure": tenure})

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Month 17 Products", type="primary", use_container_width=True):
    base_inputs = {
        "age": age,
        "gross_income": gross_income,
        "gender": gender,
        "segment": segment,
        "employee_index": employee_idx,
        "country_residence": "ES" if country == "ES" else "",
        "is_foreigner": str(is_foreigner),
        "join_channel": join_channel,
        "province_code": province_code,
        "is_active": is_active,
        "relationship_type": rel_type,
        "customer_type": cust_type,
        "is_new_customer": is_new,
        "primary_customer_flag": primary_flag,
    }

    monthly_rows = []
    for i, m in enumerate(monthly_product_data):
        inp = {**base_inputs, "tenure_months": m["tenure"], "products_owned": m["products"]}
        monthly_rows.append(build_feature_row(inp))

    probs = predict(monthly_rows)

    st.subheader("📊 Predicted Product Acquisitions — Month 17")
    st.caption("Probability that the customer will **newly add** each product next month.")

    results = sorted(zip(target_cols, probs), key=lambda x: x[1], reverse=True)

    col1, col2 = st.columns(2)
    for idx, (product, prob) in enumerate(results):
        col = col1 if idx < 5 else col2
        label = product.replace("_", " ").title()
        color = "🟢" if prob > 0.5 else "🟡" if prob > 0.25 else "🔴"
        col.metric(f"{color} {label}", f"{prob*100:.1f}%")

    st.divider()
    # Bar chart
    df_results = pd.DataFrame(results, columns=["Product", "Probability"])
    df_results["Product"] = df_results["Product"].str.replace("_", " ").str.title()
    df_results = df_results.sort_values("Probability")
    st.bar_chart(df_results.set_index("Product"), horizontal=True)

    # Top recommendation
    top_product, top_prob = results[0]
    st.success(f"✅ **Top Recommendation:** {top_product.replace('_', ' ').title()} ({top_prob*100:.1f}% probability)")
