import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ----------------------
# PAGE CONFIG
# ----------------------
st.set_page_config(page_title="Nykaa Premium Discount Intelligence", layout="wide")

# ----------------------
# APPLE-INSPIRED DARK THEME CSS
# ----------------------
st.markdown("""
<style>
body {
    background-color: #0b0b0f;
    color: #f5f5f7;
}
.metric-card {
    background: #111114;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 30px rgba(255,255,255,0.05);
}
.stSlider > div > div {
    color: #f5f5f7;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# MOCK DATA GENERATION
# ----------------------
np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "discount_percent": np.random.choice([0,10,20,30,40], n),
    "order_value": np.random.normal(2000, 500, n),
    "repeat_purchase": np.random.choice([0,1], n, p=[0.4,0.6]),
    "days_between_orders": np.random.normal(45, 15, n),
})

# Simulated churn label

data["churn"] = np.where((data["discount_percent"] > 20) & (data["repeat_purchase"] == 0), 1, 0)

# ----------------------
# HEADER
# ----------------------
st.title("Nykaa Premium Brand Intelligence Dashboard")
st.markdown("### Is Discounting Killing Premium Brand Perception?")

# ----------------------
# KPI SECTION
# ----------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Revenue (Simulated)", f"₹{int(data['order_value'].sum()):,}")

with col2:
    st.metric("Avg Discount %", f"{data['discount_percent'].mean():.1f}%")

with col3:
    st.metric("Repeat Rate", f"{data['repeat_purchase'].mean()*100:.1f}%")

with col4:
    st.metric("Churn Rate", f"{data['churn'].mean()*100:.1f}%")

st.divider()

# ----------------------
# DISCOUNT DEPENDENCY ANALYSIS
# ----------------------
st.subheader("Discount vs Repeat Behavior")
fig = px.box(data, x="discount_percent", y="days_between_orders", 
             color="discount_percent",
             template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# CHURN PREDICTION MODEL
# ----------------------
st.subheader("Churn Risk Modeling")

X = data[["discount_percent", "order_value", "days_between_orders"]]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, preds)

st.metric("Churn Model AUC", f"{auc:.2f}")

# ----------------------
# SIMULATION TOOL
# ----------------------
st.subheader("Discount Strategy Simulator")

new_discount = st.slider("Select Proposed Discount %", 0, 50, 20)

simulated_revenue = data["order_value"].mean() * (1 - new_discount/100)
simulated_margin = simulated_revenue * 0.35

colA, colB = st.columns(2)

with colA:
    st.metric("Projected Avg Order Value", f"₹{simulated_revenue:.0f}")

with colB:
    st.metric("Projected Contribution Margin", f"₹{simulated_margin:.0f}")

st.divider()

# ----------------------
# EXECUTIVE INSIGHT
# ----------------------
st.subheader("Strategic Insight")

st.write("""
Heavy discounting correlates with increased churn probability and longer repeat purchase cycles.

Recommendation:
- Reduce blanket discounting
- Target only high-value churn-risk customers
- Protect premium full-price loyalists
""")
