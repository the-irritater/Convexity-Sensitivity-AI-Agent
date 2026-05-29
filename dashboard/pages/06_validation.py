"""Dashboard Page 6: Validation"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import load_bond_portfolio, load_monte_carlo
from src.part1_analytics import BondAnalytics, PortfolioAnalytics

st.set_page_config(page_title="Validation", page_icon="✅", layout="wide")
st.title("✅ AI Agent Validation")

df = load_bond_portfolio()
mc_df = load_monte_carlo()

tab1, tab2, tab3 = st.tabs(["Bond Accuracy", "Shock Scenarios", "MC Cross-Validation"])

with tab1:
    st.markdown("### Bond-Level Calculation Verification")
    st.markdown("Computing duration & convexity from first principles and comparing against CSV values.")
    
    n_samples = st.slider("Sample Size", 10, 100, 30)
    np.random.seed(42)
    sample_idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
    
    results = []
    for idx in sample_idx:
        row = df.iloc[idx]
        try:
            bond = BondAnalytics(row['FaceValue'], row['CouponRate'], row['CouponFrequency'],
                                 row['YearsToMaturity'], row['YieldToMaturity'])
            computed = bond.full_metrics()
            results.append({
                'BondID': row['BondID'],
                'CSV Duration': f"{row['ModifiedDuration']:.4f}",
                'Computed Duration': f"{computed['ModifiedDuration']:.4f}",
                'Duration Error %': f"{abs(computed['ModifiedDuration']-row['ModifiedDuration'])/max(row['ModifiedDuration'],0.001)*100:.2f}%",
                'CSV Convexity': f"{row['Convexity']:.2f}",
                'Computed Convexity': f"{computed['Convexity']:.2f}",
            })
        except:
            pass
    
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Multi-Scenario Yield Curve Shock Validation")
    
    portfolio = PortfolioAnalytics(df)
    port_dur = portfolio.portfolio_duration()
    port_conv = portfolio.portfolio_convexity()
    total_mv = portfolio.total_market_value
    
    shocks = list(range(-300, 325, 25))
    shocks = [s for s in shocks if s != 0]
    
    shock_results = []
    for s in shocks:
        dy = s / 10000.0
        dur_only = -port_dur * dy * total_mv
        dur_conv = (-port_dur * dy + 0.5 * port_conv * dy**2) * total_mv
        shock_results.append({'Shock_bps': s, 'Duration_Only': dur_only, 'Duration_Convexity': dur_conv,
                             'Convexity_Benefit': 0.5 * port_conv * dy**2 * total_mv})
    
    shock_df = pd.DataFrame(shock_results)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shock_df['Shock_bps'], y=shock_df['Duration_Only']/1e5,
                              mode='lines', name='Duration Only', line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=shock_df['Shock_bps'], y=shock_df['Duration_Convexity']/1e5,
                              mode='lines', name='Duration + Convexity', line=dict(color='red', width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title='P&L: Duration vs Duration+Convexity Across Shocks',
                      xaxis_title='Yield Shock (bps)', yaxis_title='P&L (₹ Lakhs)', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.bar(shock_df, x='Shock_bps', y='Convexity_Benefit', title='Convexity Benefit Across Shocks',
                  color='Convexity_Benefit', color_continuous_scale='YlOrRd',
                  labels={'Convexity_Benefit': 'Benefit (INR)', 'Shock_bps': 'Shock (bps)'})
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Monte Carlo vs Analytical Cross-Validation")
    
    mc_check = mc_df.copy()
    dy = mc_check['ParallelShift_bps'] / 10000.0
    mc_check['Analytical_PnL'] = (-port_dur * dy + 0.5 * port_conv * dy**2) * total_mv
    mc_check['Error'] = (mc_check['Analytical_PnL'] - mc_check['PnL_Total_INR']).abs()
    
    corr = np.corrcoef(mc_check['Analytical_PnL'], mc_check['PnL_Total_INR'])[0, 1]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Correlation", f"{corr:.6f}")
    c2.metric("Mean Error", f"₹{mc_check['Error'].mean()/1e5:.2f}L")
    c3.metric("Scenarios", f"{len(mc_check)}")
    
    fig = px.scatter(mc_check, x='PnL_Total_INR', y='Analytical_PnL',
                     title=f'MC vs Analytical P&L (Correlation: {corr:.4f})',
                     labels={'PnL_Total_INR': 'MC P&L (INR)', 'Analytical_PnL': 'Analytical P&L (INR)'},
                     opacity=0.5)
    min_v = min(mc_check['PnL_Total_INR'].min(), mc_check['Analytical_PnL'].min())
    max_v = max(mc_check['PnL_Total_INR'].max(), mc_check['Analytical_PnL'].max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines',
                              name='Perfect Fit', line=dict(dash='dash', color='red')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
