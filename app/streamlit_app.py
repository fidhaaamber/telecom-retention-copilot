import os
from pathlib import Path
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Retention Copilot", page_icon="📡", layout="wide")
ROOT = Path(__file__).parents[1]; RAW = ROOT / "data/raw/cell2celltrain.csv"
# Cloud deployments set API_URL in Streamlit secrets; local use defaults to FastAPI.
API = os.getenv("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")
@st.cache_data
def data():
    return pd.read_csv(RAW) if RAW.exists() else pd.DataFrame()
def request(path, customer):
    if not API:
        st.error("API URL is not configured. Set API_URL in Streamlit secrets.")
        return None
    try: return httpx.post(f"{API.rstrip('/')}{path}", json={"customer": customer}, timeout=20).json()
    except Exception as exc: st.error(f"API unavailable: {exc}."); return None

page = st.sidebar.radio("Screen", ["Portfolio & EDA", "Single Customer Risk", "High-Risk Work Queue", "Retention Copilot"])
df = data()
if page == "Portfolio & EDA":
    st.title("Portfolio & EDA Dashboard")
    if df.empty: st.warning("Dataset not found in data/raw."); st.stop()
    a,b,c,d = st.columns(4); a.metric("Customers", f"{len(df):,}"); b.metric("Columns", len(df.columns)); c.metric("Churn rate", f"{(df.Churn == 'Yes').mean():.1%}"); d.metric("Missing cells", f"{df.isna().sum().sum():,}")
    left,right=st.columns(2)
    left.plotly_chart(px.bar(df.Churn.value_counts().rename_axis("Churn").reset_index(name="Customers"),x="Churn",y="Customers",title="Churn distribution"),use_container_width=True)
    right.plotly_chart(px.histogram(df,x="MonthlyRevenue",color="Churn",nbins=40,title="Monthly revenue by churn"),use_container_width=True)
    st.plotly_chart(px.bar(df.isna().mean().sort_values(ascending=False).head(15).reset_index(),x="index",y=0,title="Top missingness rates"),use_container_width=True)
elif page in ("Single Customer Risk", "Retention Copilot"):
    st.title(page)
    if df.empty: st.stop()
    customer_id = st.selectbox("Customer ID", df.CustomerID.astype(str).head(1000))
    customer = df.loc[df.CustomerID.astype(str) == customer_id].iloc[0].where(pd.notna, None).to_dict()
    result = request("/predict", customer)
    if result:
        st.metric("Churn probability", f"{result['churn_probability']:.1%}", result["risk_band"])
        st.dataframe(pd.DataFrame(result["drivers"]), use_container_width=True)
        if page == "Retention Copilot" and st.button("Generate grounded recommendation"):
            response = request("/recommend", customer)
            if response: st.json(response)
else:
    st.title("High-Risk Work Queue")
    scored = ROOT / "reports/scored_customers.csv"
    if not scored.exists(): st.warning("Run training to generate the scored queue."); st.stop()
    queue = pd.read_csv(scored); minimum = st.slider("Minimum churn probability", 0.0, 1.0, .6, .05)
    queue = queue[queue.churn_probability >= minimum].sort_values("churn_probability", ascending=False)
    st.dataframe(queue, use_container_width=True)
    st.download_button("Export CSV", queue.to_csv(index=False), "high_risk_queue.csv", "text/csv")
