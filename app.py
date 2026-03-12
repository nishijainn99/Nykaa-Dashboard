import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Nykaa Premium Discount Intelligence", layout="wide")

st.markdown("""

""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("nykaa_premium_discount_data.csv")

data = load_data()

st.title("Nykaa Premium Brand Intelligence Dashboard")
st.markdown("### Is Discounting Killing Premium Brand Perception?")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Revenue", f"₹{int(data['order_value'].sum()):,}")

with col2:
    st.metric("Avg Discount %", f"{data['discount_percent'].mean():.1f}%")

with col3:
    st.metric("Repeat Rate", f"{data['repeat_purchase'].mean()*100:.1f}%")

with col4:
    st.metric("Churn Rate", f"{data['churn'].mean()*100:.1f}%")

st.divider()

st.subheader("Discount vs Repeat Behavior")
fig = px.box(
    data,
    x="discount_percent",
    y="days_between_orders",
    color="discount_percent",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Churn Risk Modeling")
X = data[["discount_percent", "order_value", "days_between_orders"]]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
st.metric("Churn Model AUC", f"{auc:.2f}")

st.subheader("Discount Strategy Simulator")
new_discount = st.slider("Select Proposed Discount %", 0, 50, 20)

simulated_revenue = data["order_value"].mean() * (1 - new_discount / 100)
simulated_margin = simulated_revenue * 0.35

colA, colB = st.columns(2)
with colA:
    st.metric("Projected Avg Order Value", f"₹{simulated_revenue:.0f}")
with colB:
    st.metric("Projected Contribution Margin", f"₹{simulated_margin:.0f}")

st.divider()

st.subheader("Strategic Insight")
st.write("""
Heavy discounting correlates with increased churn probability and longer repeat purchase cycles.

Recommendation:
- Reduce blanket discounting
- Target only high-value churn-risk customers
- Protect premium full-price loyalists
""")
