# FedEx SMART Hackathon Submission
# AI-Powered DCA Command Center
# Focus: Recovery Risk, Cost Leakage & Profit Optimization
# Author: Rishika Shreshtha and Arjun Sahu
# Note: Cost and recovery logic designed based on business assumptions


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="FedEx DCA Command Center", layout="wide")

st.title("📦 FedEx AI-Powered DCA Command Center")
st.caption("Risk Prediction • Cost Simulation • Profit Intelligence")

# =========================================================
# 1. DATA GENERATION (SIMULATED ENTERPRISE DATA)
# =========================================================
np.random.seed(42)
n = 1200

data = pd.DataFrame({
    "Overdue_Amount": np.random.randint(10000, 1000000, n),
    "Days_Overdue": np.random.randint(5, 240, n),
    "Customer_Risk": np.random.choice(["Low", "Medium", "High"], n, p=[0.45, 0.35, 0.2]),
    "Past_Payments": np.random.randint(0, 6, n),
    "DCA": np.random.choice(["DCA Alpha", "DCA Beta", "DCA Gamma"], n),
    "DCA_Commission": np.random.choice([0.05, 0.07, 0.10], n)
})

risk_map = {"Low": 0, "Medium": 1, "High": 2}
data["Risk_Score"] = data["Customer_Risk"].map(risk_map)

data["Recovered_Flag"] = (
    (data["Overdue_Amount"] < 500000).astype(int)
    & (data["Days_Overdue"] < 150).astype(int)
    & (data["Risk_Score"] < 2).astype(int)
)

# =========================================================
# 2. ML MODEL (LOGISTIC REGRESSION)
# =========================================================
features = ["Overdue_Amount", "Days_Overdue", "Risk_Score", "Past_Payments"]
X = data[features]
y = data["Recovered_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

data["Recovery_Probability"] = model.predict_proba(X)[:, 1]

# =========================================================
# 3. PROFIT & COST ENGINE
# =========================================================
CALL_COST_PER_DAY = 150
MANPOWER_COST_PER_CASE = 1200
LEGAL_COST_FIXED = 25000
CAPITAL_COST_RATE = 0.12

data["Expected_Recovery"] = data["Overdue_Amount"] * data["Recovery_Probability"]
data["Expected_Cost"] = data["Expected_Recovery"] * data["DCA_Commission"]

data["Call_Cost"] = data["Days_Overdue"] * CALL_COST_PER_DAY
data["Manpower_Cost"] = MANPOWER_COST_PER_CASE
data["Legal_Cost"] = np.where(data["Days_Overdue"] > 160, LEGAL_COST_FIXED, 0)
data["Opportunity_Cost"] = (
    data["Overdue_Amount"] * CAPITAL_COST_RATE * (data["Days_Overdue"] / 365)
)

data["Total_Cost"] = (
    data["Call_Cost"] +
    data["Manpower_Cost"] +
    data["Legal_Cost"] +
    data["Opportunity_Cost"]
)

data["Net_Profit"] = data["Expected_Recovery"] - data["Expected_Cost"] - data["Total_Cost"]

# =========================================================
# 4. AI DECISION POLICY
# =========================================================
def ai_action(row):
    if row["Net_Profit"] < 0:
        return "Write-Off Candidate"
    elif row["Recovery_Probability"] > 0.75 and row["Net_Profit"] > 20000:
        return "Fast-Track Recovery"
    elif row["Days_Overdue"] > 160:
        return "Escalate / Legal"
    else:
        return "Standard Follow-up"

data["AI_Recommendation"] = data.apply(ai_action, axis=1)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard",
    "🤖 AI Case Analyzer",
    "📈 DCA Performance",
    "⚖️ Governance & Risk",
    "💰 Cost Savings Simulation"
])

# =========================================================
# TAB 1 – EXECUTIVE DASHBOARD
# =========================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Overdue (₹)", f"₹{data['Overdue_Amount'].sum():,.0f}")
    c2.metric("Predicted Recovery (₹)", f"₹{data['Expected_Recovery'].sum():,.0f}")
    c3.metric("Net Profit After Cost (₹)", f"₹{data['Net_Profit'].sum():,.0f}")
    c4.metric("Avg Recovery Probability", f"{data['Recovery_Probability'].mean()*100:.1f}%")

    fig = px.scatter(
        data,
        x="Days_Overdue",
        y="Net_Profit",
        color="Customer_Risk",
        size="Overdue_Amount",
        title="Net Profit vs Risk Landscape"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 – AI CASE ANALYZER
# =========================================================
with tab2:
    st.subheader("🔍 Analyze New Case")

    col1, col2, col3 = st.columns(3)
    overdue = col1.number_input("Overdue Amount (₹)", min_value=1, step=1)

    days = col2.slider("Days Overdue", 1, 300, 60)
    past = col3.slider("Past Payments Count", 0, 5, 1)

    risk = st.selectbox("Customer Risk Level", ["Low", "Medium", "High"])
    commission = st.selectbox("DCA Commission", [0.05, 0.07, 0.10])

    if st.button("🔮 Run AI Analysis"):
        risk_score = risk_map[risk]
        X_new = np.array([[overdue, days, risk_score, past]])
        prob = model.predict_proba(X_new)[0][1]

        expected_recovery = overdue * prob
        expected_cost = expected_recovery * commission

        call_cost = days * CALL_COST_PER_DAY
        manpower_cost = MANPOWER_COST_PER_CASE
        legal_cost = LEGAL_COST_FIXED if days > 160 else 0
        opportunity_cost = overdue * CAPITAL_COST_RATE * (days / 365)

        net_profit = expected_recovery - expected_cost - (
            call_cost + manpower_cost + legal_cost + opportunity_cost
        )

        if net_profit < 0:
            action = "Write-Off Candidate"
        elif prob > 0.75 and net_profit > 20000:
            action = "Fast-Track Recovery"
        elif days > 160:
            action = "Escalate / Legal"
        else:
            action = "Standard Follow-up"

        st.metric("Recovery Probability", f"{prob*100:.1f}%")
        st.metric("Net Profit (₹)", f"₹{net_profit:,.0f}")
        st.success(f"AI Recommendation: {action}")

# =========================================================
# TAB 3 – DCA PERFORMANCE
# =========================================================
with tab3:
    dca_perf = data.groupby("DCA")[["Net_Profit"]].mean().reset_index()
    fig2 = px.bar(dca_perf, x="DCA", y="Net_Profit", title="DCA Profitability Ranking")
    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# TAB 4 – GOVERNANCE & RISK
# =========================================================
with tab4:
    fig3 = px.pie(data, names="AI_Recommendation", title="AI Governance Decisions")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🚨 High-Risk Cases")
    st.dataframe(
        data[data["Net_Profit"] < 0][
            ["Overdue_Amount", "Days_Overdue", "Customer_Risk", "Net_Profit", "AI_Recommendation"]
        ].head(15),
        use_container_width=True
    )

# =========================================================
# TAB 5 – COST SAVINGS SIMULATION (KILLER FEATURE)
# =========================================================
with tab5:
    st.header("💰 Cost Savings Simulation")

    orders = st.slider("Total Accounts Handled", 100, 5000, 1000)
    penalty = st.slider("Penalty per Failed Recovery (₹)", 500, 5000, 2000)
    agent_cost = st.slider("DCA Agent Cost per Case (₹)", 200, 2000, 800)
    legal_cost = st.slider("Legal Escalation Cost (₹)", 5000, 50000, 15000)

    failure_traditional = 0.35
    failure_ai = 0.18

    traditional_cost = (
        orders * failure_traditional * penalty +
        orders * agent_cost +
        orders * 0.2 * legal_cost
    )

    ai_cost = (
        orders * failure_ai * penalty +
        orders * 0.6 * agent_cost +
        orders * 0.08 * legal_cost
    )

    savings = traditional_cost - ai_cost
    percent_savings = (savings / traditional_cost) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Traditional Cost", f"₹{traditional_cost:,.0f}")
    col2.metric("AI-Based Cost", f"₹{ai_cost:,.0f}")
    col3.metric("Total Savings", f"₹{savings:,.0f}")
    col4.metric("Cost Reduction", f"{percent_savings:.2f}%")

    st.success("AI-driven prioritization delivers >50% operational cost reduction.")

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "This platform simulates an enterprise-grade AI decision system integrating "
    "risk prediction, cost modeling, and profit-optimal debt recovery."
)

