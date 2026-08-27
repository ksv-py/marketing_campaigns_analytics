from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import build_model, campaign_summary, load_campaign_data


st.set_page_config(
    page_title="Campaign Lens",
    page_icon="CL",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .hero { padding: 1.5rem 0 1rem; border-bottom: 1px solid #d9e2df; }
    .eyebrow { color: #d45d3f; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; }
    .hero h1 { color: #183a3a; font-size: 2.8rem; margin: .25rem 0; }
    .hero p { color: #54706d; max-width: 720px; font-size: 1.05rem; }
    [data-testid="stMetricValue"] { color: #183a3a; }
    </style>
    """,
    unsafe_allow_html=True,
)


data_path = Path(__file__).parent / "data" / "marketing_campaign.csv"
if not data_path.exists():
    st.error("Original dataset not found. Download marketing_campaign.csv from the Kaggle source below and place it in the data folder.")
    st.link_button("Open original dataset source", "https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign")
    st.stop()
df = load_campaign_data(data_path)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Campaign Lens / customer response lab</div>
      <h1>Spend attention where it actually converts.</h1>
      <p>A small, decision-focused view of campaign reach, customer value, and the audiences most likely to respond next.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Slice the audience")
    education = st.multiselect("Education", sorted(df["education"].unique()), default=sorted(df["education"].unique()))
    channels = st.multiselect("Preferred channel", sorted(df["preferred_channel"].unique()), default=sorted(df["preferred_channel"].unique()))
    income_range = st.slider("Annual income", int(df.income.min()), int(df.income.max()), (int(df.income.min()), int(df.income.max())))

filtered = df[
    df.education.isin(education)
    & df.preferred_channel.isin(channels)
    & df.income.between(*income_range)
].copy()

summary = campaign_summary(filtered)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers in view", f"{len(filtered):,}")
col2.metric("Overall response", f"{summary['response_rate']:.1%}")
col3.metric("Revenue influenced", f"${summary['revenue_influenced']:,.0f}")
col4.metric("Avg. customer value", f"${summary['avg_customer_value']:,.0f}")

st.caption("Source: the original Marketing Campaign customer export. Derived fields are documented in analysis.py; revenue here is observed product spend, not causal campaign lift.")

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Response by campaign")
    campaign_view = (
        filtered.groupby("campaign", as_index=False)
        .agg(responses=("responded", "sum"), customers=("responded", "size"))
        .assign(response_rate=lambda x: x.responses / x.customers)
        .sort_values("response_rate", ascending=False)
    )
    fig = px.bar(campaign_view, x="campaign", y="response_rate", text_auto=".1%", color="response_rate", color_continuous_scale=["#d9ece7", "#d45d3f"])
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Response rate", xaxis_title=None, coloraxis_showscale=False, height=340, margin=dict(t=15, b=15))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Who is responding?")
    segment_view = filtered.groupby("segment", as_index=False).agg(response_rate=("responded", "mean"), customers=("responded", "size"))
    fig = px.scatter(segment_view, x="customers", y="response_rate", size="customers", color="segment", text="segment", hover_data={"response_rate": ":.1%"})
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Response rate", xaxis_title="Customers", height=340, showlegend=False, margin=dict(t=15, b=15))
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Channel economics")
channel_view = filtered.groupby("preferred_channel", as_index=False).agg(
    customers=("responded", "size"), response_rate=("responded", "mean"), revenue=("revenue_influenced", "sum")
).sort_values("revenue", ascending=False)
st.dataframe(
    channel_view.style.format({"response_rate": "{:.1%}", "revenue": "${:,.0f}"}),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Response propensity model"):
    model, metrics = build_model(df)
    st.write(f"A logistic regression baseline scores **{metrics['roc_auc']:.2f} ROC AUC** on a held-out sample.")
    st.write("Top signals in this demo are shown as directional indicators, not causal claims.")
    st.dataframe(metrics["signals"].head(8), use_container_width=True, hide_index=True)
