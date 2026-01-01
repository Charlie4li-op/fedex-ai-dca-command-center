# 📦 FedEx AI-Powered DCA Command Center

## Overview
This project reimagines **FedEx’s Debt Collection Agency (DCA) management** using an AI-driven decision intelligence platform.  
Instead of manual tracking through spreadsheets and emails, the system provides **real-time recovery prediction, cost modeling, and profit-optimized decision support**.

The solution is designed as an **internal FedEx dashboard** for recovery managers and leadership teams to improve recovery efficiency, reduce cost leakage, and strengthen governance.

---

## Problem Statement
FedEx manages a large volume of overdue customer accounts through multiple external DCAs.  
The current process faces challenges such as:
- Manual case allocation and tracking
- Limited visibility into true recovery cost vs value
- Delayed escalations and weak governance
- Uniform effort spent on both low-value and high-risk cases
- Lack of performance-based accountability for DCAs

---

## Solution Summary
The **AI-Powered DCA Command Center** addresses these gaps by combining:
- Explainable machine learning (Logistic Regression)
- Cost and profit intelligence
- Real-time decision recommendations
- Governance and risk visibility
- Cost-savings simulation for business impact estimation

---

## Key Features
- 📊 **Executive Dashboard** with live KPIs (overdue exposure, recovery, profit)
- 🤖 **AI Case Analyzer** for real-time recovery probability and action recommendation
- 💰 **Cost & Profit Modeling** (calls, manpower, legal, capital lock-in)
- 🔁 **What-If Simulation** to test operational decisions
- 📈 **DCA Performance Analytics** based on profitability
- ⚖️ **Governance & Risk Monitoring** for write-off and escalation cases
- 💡 **Cost Savings Simulation** comparing traditional vs AI-driven approach

---

## Solution Pipeline
1. Synthetic enterprise-scale overdue account data generation  
2. Feature engineering (risk, ageing, payment behavior)  
3. Logistic Regression model for recovery prediction  
4. Cost modeling (operational, legal, opportunity cost)  
5. Net profit and ROI calculation  
6. AI-driven decision logic (recover, escalate, write-off)  
7. Real-time Streamlit dashboard for interaction and analysis  

---

## Business Assumptions (Simulation)
- Call cost and manpower cost are average operational estimates
- Legal escalation triggered after 160 days overdue
- Capital cost represents blocked working capital
- Model is trained on synthetic data for demonstration purposes
- Decision thresholds are configurable and policy-driven

---

## Tech Stack
- **Python**
- **Streamlit** (UI & Dashboard)
- **Scikit-learn** (Logistic Regression)
- **Pandas & NumPy** (Data handling)
- **Plotly** (Visual analytics)

---

## How to Run the Application
```bash
pip install -r requirements.txt
streamlit run app.py
