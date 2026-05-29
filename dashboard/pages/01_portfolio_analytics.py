"""Dashboard Page 1: Portfolio Analytics"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import load_bond_portfolio
from src.part1_analytics import PortfolioAnalytics

st.set_page_config(page_title="Portfolio Analytics", page_icon="", layout="wide")
st.title(" Portfolio Duration & Convexity Analytics")

df = load_bond_portfolio()
portfolio = PortfolioAnalytics(df)
summary = portfolio.summary()

# Key Metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Modified Duration", f"{summary['Portfolio_Modified_Duration']:.2f}")
c2.metric("Macaulay Duration", f"{summary['Portfolio_Macaulay_Duration']:.2f}")
c3.metric("Convexity", f"{summary['Portfolio_Convexity']:.1f}")
c4.metric("DV01 (INR)", f"₹{summary['Portfolio_DV01_INR']/1e5:.1f}L")
c5.metric("YTM", f"{summary['Portfolio_YTM']*100:.2f}%")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Duration vs Convexity", "Sector Analysis", "Key Rate Duration", "Sensitivity"])

with tab1:
    fig = px.scatter(df, x='ModifiedDuration', y='Convexity', color='Sector',
                     size='MarketValue_INR', hover_data=['BondID', 'CouponRate', 'YieldToMaturity'],
                     title='Duration vs Convexity by Sector',
                     labels={'ModifiedDuration': 'Modified Duration (years)', 'Convexity': 'Convexity'})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    contrib = portfolio.duration_contribution_by('Sector')
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(contrib, x='Duration_Contribution', y='Sector', orientation='h',
                     title='Duration Contribution by Sector', color='Duration_Contribution',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(contrib, values='DV01_Contribution', names='Sector',
                     title='DV01 Distribution by Sector')
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(contrib, use_container_width=True, hide_index=True)

with tab3:
    krd = portfolio.key_rate_duration_profile()
    if not krd.empty:
        fig = px.bar(krd, x='KeyRateBucket', y='KRD_Contribution',
                     title='Key Rate Duration Profile', color='KRD_Contribution',
                     color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(krd, use_container_width=True, hide_index=True)

with tab4:
    shocks = st.slider("Shock Range (bps)", -300, 300, (-200, 200), step=25)
    shock_list = list(range(shocks[0], shocks[1]+1, 25))
    shock_list = [s for s in shock_list if s != 0]
    sens = portfolio.price_sensitivity_analysis(shock_list)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sens['Shock_bps'], y=sens['Total_PnL_INR']/1e5, name='Total P&L',
                         marker_color=['#4CAF50' if x > 0 else '#F44336' for x in sens['Total_PnL_INR']]))
    fig.update_layout(title='Portfolio P&L Sensitivity', xaxis_title='Yield Shock (bps)',
                      yaxis_title='P&L (₹ Lakhs)', height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sens, use_container_width=True, hide_index=True)
