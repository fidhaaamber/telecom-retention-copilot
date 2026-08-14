import os
from pathlib import Path
import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Telecom Retention Copilot", page_icon="📡", layout="wide")
ROOT = Path(__file__).parents[1]
RAW = ROOT / "data/raw/cell2celltrain.csv"

# Cloud deployments set API_URL in Streamlit secrets; local use defaults to FastAPI.
API = os.getenv("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")

@st.cache_data
def load_data():
    return pd.read_csv(RAW) if RAW.exists() else pd.DataFrame()

def request_api(path, customer):
    if not API:
        st.error("API URL is not configured. Set API_URL in Streamlit secrets.")
        return None
    try:
        res = httpx.post(f"{API.rstrip('/')}{path}", json={"customer": customer}, timeout=25)
        res.raise_for_status()
        return res.json()
    except Exception as exc:
        st.error(f"API unavailable: {exc}.")
        return None

df = load_data()

st.sidebar.title("📡 Retention Copilot")
page = st.sidebar.radio(
    "Navigation",
    ["Portfolio & EDA", "Single Customer Risk", "High-Risk Work Queue", "Retention Copilot"]
)

# ----------------------------------------------------
# SCREEN 1: Portfolio & EDA Dashboard
# ----------------------------------------------------
if page == "Portfolio & EDA":
    st.title("📊 Portfolio & EDA Dashboard")
    st.caption("Overview of telecom customer churn, key metrics, missingness, and feature distributions.")

    if df.empty:
        st.warning("Dataset cell2celltrain.csv not found in data/raw.")
        st.stop()

    # Filters
    st.sidebar.subheader("Dashboard Filters")
    churn_filter = st.sidebar.multiselect("Filter by Churn", options=df["Churn"].unique(), default=df["Churn"].unique())
    filtered_df = df[df["Churn"].isin(churn_filter)]

    # Dataset KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(filtered_df):,}")
    k2.metric("Total Attributes", len(filtered_df.columns))
    churn_pct = (filtered_df["Churn"] == "Yes").mean() if not filtered_df.empty else 0.0
    k3.metric("Churn Rate", f"{churn_pct:.1%}")
    missing_cells = filtered_df.isna().sum().sum()
    k4.metric("Missing Values", f"{missing_cells:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        churn_counts = filtered_df["Churn"].value_counts().rename_axis("Churn").reset_index(name="Count")
        fig_churn = px.pie(churn_counts, names="Churn", values="Count", color="Churn",
                           color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
                           title="Customer Churn Ratio", hole=0.4)
        st.plotly_chart(fig_churn, use_container_width=True)

    with col2:
        st.subheader("Monthly Revenue by Churn")
        fig_rev = px.histogram(filtered_df, x="MonthlyRevenue", color="Churn", nbins=40,
                               color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
                               title="Monthly Revenue Distribution ($)", marginal="box")
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top Missingness Rates")
        missing_df = filtered_df.isna().mean().sort_values(ascending=False).head(12).reset_index()
        missing_df.columns = ["Attribute", "Missing Ratio"]
        fig_miss = px.bar(missing_df, x="Attribute", y="Missing Ratio", color="Missing Ratio",
                          color_continuous_scale="Reds", title="Top 12 Attributes with Missing Data")
        st.plotly_chart(fig_miss, use_container_width=True)

    with col4:
        st.subheader("Equipment Age vs Churn")
        if "EquipmentAge" in filtered_df.columns:
            fig_eq = px.box(filtered_df, x="Churn", y="EquipmentAge", color="Churn",
                            color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"},
                            title="Equipment Age (Months) by Churn Status")
            st.plotly_chart(fig_eq, use_container_width=True)

# ----------------------------------------------------
# SCREEN 2: Single Customer Risk
# ----------------------------------------------------
elif page == "Single Customer Risk":
    st.title("🎯 Single Customer Risk Assessment")
    st.caption("Inspect individual customer churn probability, risk band, and key directional risk drivers.")

    if df.empty:
        st.warning("Dataset not found in data/raw.")
        st.stop()

    customer_id = st.selectbox("Select Customer ID", df["CustomerID"].astype(str).head(1000))
    customer_row = df.loc[df["CustomerID"].astype(str) == customer_id].iloc[0]
    customer_dict = customer_row.where(pd.notna, None).to_dict()

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customer ID", customer_id)
    c2.metric("Monthly Revenue", f"${customer_row.get('MonthlyRevenue', 0):.2f}" if pd.notna(customer_row.get('MonthlyRevenue')) else "N/A")
    c3.metric("Tenure (Months)", f"{customer_row.get('Tenure', 0)}" if pd.notna(customer_row.get('Tenure')) else "N/A")
    c4.metric("Overage Minutes", f"{customer_row.get('OverageMinutes', 0)}" if pd.notna(customer_row.get('OverageMinutes')) else "N/A")

    if st.button("Calculate Churn Risk", type="primary"):
        with st.spinner("Scoring customer churn risk..."):
            result = request_api("/predict", customer_dict)

        if result:
            prob = result["churn_probability"]
            band = result["risk_band"]
            thresh = result["decision_threshold"]

            st.markdown("### Risk Level Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Churn Probability", f"{prob:.1%}")
            m2.metric("Risk Band", band)
            m3.metric("Decision Threshold", f"{thresh:.0%}")

            st.markdown("### Key Risk Drivers (TreeSHAP Impact)")
            drivers_df = pd.DataFrame(result["drivers"])
            if not drivers_df.empty:
                pos_drivers = drivers_df[drivers_df["direction"] == "increases risk"]
                neg_drivers = drivers_df[drivers_df["direction"] == "reduces risk"]

                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.error("⚠️ **Top Factors Increasing Churn Risk**")
                    if not pos_drivers.empty:
                        st.table(pos_drivers[["feature", "impact"]])
                    else:
                        st.write("No major risk-increasing drivers detected.")

                with col_neg:
                    st.success("✅ **Top Factors Reducing Churn Risk**")
                    if not neg_drivers.empty:
                        st.table(neg_drivers[["feature", "impact"]])
                    else:
                        st.write("No major risk-reducing drivers detected.")

# ----------------------------------------------------
# SCREEN 3: High-Risk Work Queue
# ----------------------------------------------------
elif page == "High-Risk Work Queue":
    st.title("📋 High-Risk Work Queue")
    st.caption("Ranked list of accounts prioritized for proactive retention analyst outreach.")

    scored_path = ROOT / "reports/scored_customers.csv"
    if not scored_path.exists():
        st.warning("Scored queue file reports/scored_customers.csv not found. Run training script first.")
        st.stop()

    queue_df = pd.read_csv(scored_path)

    st.sidebar.subheader("Queue Filters")
    min_prob = st.sidebar.slider("Minimum Churn Probability", 0.0, 1.0, 0.50, 0.05)

    filtered_queue = queue_df[queue_df["churn_probability"] >= min_prob].sort_values("churn_probability", ascending=False)

    q1, q2, q3 = st.columns(3)
    q1.metric("Total High-Risk Accounts", f"{len(filtered_queue):,}")
    avg_risk = filtered_queue["churn_probability"].mean() if not filtered_queue.empty else 0.0
    q2.metric("Average Churn Probability", f"{avg_risk:.1%}")
    high_band_count = (filtered_queue["risk_band"] == "High").sum() if "risk_band" in filtered_queue.columns else 0
    q3.metric("High-Risk Band Accounts", f"{high_band_count:,}")

    st.markdown("---")
    st.dataframe(filtered_queue, use_container_width=True)

    csv_data = filtered_queue.to_csv(index=False)
    st.download_button(
        label="📥 Export Work Queue CSV",
        data=csv_data,
        file_name="high_risk_retention_queue.csv",
        mime="text/csv",
        type="primary"
    )

# ----------------------------------------------------
# SCREEN 4: Retention Copilot
# ----------------------------------------------------
elif page == "Retention Copilot":
    st.title("🤖 Grounded Retention Copilot")
    st.caption("Evidence-based retention decision support powered by FastAPI & Local Llama 3 / Approved Rules.")

    if df.empty:
        st.warning("Dataset not found in data/raw.")
        st.stop()

    customer_id = st.selectbox("Select Customer ID for Copilot Review", df["CustomerID"].astype(str).head(1000))
    customer_row = df.loc[df["CustomerID"].astype(str) == customer_id].iloc[0]
    customer_dict = customer_row.where(pd.notna, None).to_dict()

    st.markdown("---")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Customer ID", customer_id)
    s2.metric("Monthly Revenue", f"${customer_row.get('MonthlyRevenue', 0):.2f}" if pd.notna(customer_row.get('MonthlyRevenue')) else "N/A")
    s3.metric("Tenure", f"{customer_row.get('Tenure', 0)} mos" if pd.notna(customer_row.get('Tenure')) else "N/A")
    s4.metric("Overage Mins", f"{customer_row.get('OverageMinutes', 0)}" if pd.notna(customer_row.get('OverageMinutes')) else "N/A")

    if st.button("🤖 Generate Grounded Recommendation", type="primary"):
        with st.spinner("Invoking Retention Copilot (FastAPI + Llama 3 / Rules engine)..."):
            rec_response = request_api("/recommend", customer_dict)

        if rec_response:
            st.markdown("---")
            st.success(f"### 💡 Recommended Action: {rec_response.get('recommended_action', 'N/A')}")

            col_a, col_b = st.columns([3, 2])

            with col_a:
                st.subheader("📌 Rationale")
                st.info(rec_response.get("rationale", "No rationale provided."))

                st.subheader("📋 Observed Customer Facts")
                facts = rec_response.get("observed_facts", [])
                for fact in facts:
                    st.markdown(f"- {fact}")

            with col_b:
                st.subheader("⚠️ Governance & Limitations")
                st.warning(rec_response.get("limitations", "Decision support only."))

                st.subheader("🧑‍💼 Analyst Approval Controls")
                approval = st.checkbox("Analyst Approved Action", value=False)
                notes = st.text_area("Analyst Case Notes", placeholder="Enter review notes before contacting customer...")
                if st.button("Submit Analyst Decision"):
                    if approval:
                        st.success(f"Decision recorded for Customer {customer_id}: Action Approved by Analyst.")
                    else:
                        st.info(f"Decision recorded for Customer {customer_id}: Under Analyst Review.")
