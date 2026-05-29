"""Dashboard Page 3: Monte Carlo Simulation"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import load_bond_portfolio, load_monte_carlo
from src.part3_monte_carlo import (VasicekModel, CIRModel, compute_var_cvar, run_stress_tests)

st.set_page_config(page_title="Monte Carlo", page_icon="", layout="wide")
st.title(" Monte Carlo Simulation & Risk Analysis")

bond_df = load_bond_portfolio()
mc_df = load_monte_carlo()

total_mv = bond_df['MarketValue_INR'].sum()
weights = bond_df['MarketValue_INR'] / total_mv
port_dur = (weights * bond_df['ModifiedDuration']).sum()
port_conv = (weights * bond_df['Convexity']).sum()

tab1, tab2, tab3, tab4 = st.tabs(["Rate Simulation", "P&L Distribution", "VaR Analysis", "Stress Tests"])

with tab1:
    st.markdown("### Interest Rate Path Simulation")
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Model", ["Vasicek", "CIR"])
    with col2:
        n_paths = st.slider("Number of Paths", 100, 5000, 1000)
    with col3:
        horizon = st.slider("Horizon (Years)", 0.5, 5.0, 2.0, 0.5)

    if model_type == "Vasicek":
        model = VasicekModel(kappa=0.5, theta=0.065, sigma=0.01, r0=0.065)
    else:
        model = CIRModel(kappa=0.5, theta=0.065, sigma=0.05, r0=0.065)

    times, paths = model.simulate(T=horizon, n_steps=int(252*horizon), n_paths=n_paths, seed=42)

    fig = go.Figure()
    for i in range(min(30, n_paths)):
        fig.add_trace(go.Scatter(x=times, y=paths[i]*100, mode='lines', opacity=0.15,
                                  line=dict(width=0.5, color='steelblue'), showlegend=False))
    mean_path = np.mean(paths, axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    fig.add_trace(go.Scatter(x=times, y=mean_path*100, mode='lines', name='Mean',
                              line=dict(width=3, color='red')))
    fig.add_trace(go.Scatter(x=times, y=p95*100, mode='lines', name='95th %ile',
                              line=dict(width=1, dash='dash', color='orange')))
    fig.add_trace(go.Scatter(x=times, y=p5*100, mode='lines', name='5th %ile',
                              line=dict(width=1, dash='dash', color='orange')))
    fig.update_layout(title=f'{model_type} Model: Simulated Rate Paths', xaxis_title='Time (Years)',
                      yaxis_title='Rate (%)', height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Portfolio P&L Distribution (Provided MC Scenarios)")
    fig = px.histogram(mc_df, x='PnL_Total_INR', nbins=50, title='P&L Distribution',
                       labels={'PnL_Total_INR': 'P&L (INR)'}, color_discrete_sequence=['#2196F3'])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean P&L", f"₹{mc_df['PnL_Total_INR'].mean()/1e5:.1f}L")
    c2.metric("Std Dev", f"₹{mc_df['PnL_Total_INR'].std()/1e5:.1f}L")
    c3.metric("Min P&L", f"₹{mc_df['PnL_Total_INR'].min()/1e5:.1f}L")
    c4.metric("Max P&L", f"₹{mc_df['PnL_Total_INR'].max()/1e5:.1f}L")

with tab3:
    st.markdown("### Value at Risk & Expected Shortfall")
    var_results = compute_var_cvar(mc_df['PnL_Total_INR'].values)

    st.dataframe(var_results, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc_df['PnL_Total_INR']/1e5, nbinsx=50, name='P&L',
                                marker_color='#2196F3', opacity=0.7))
    colors = ['#FF9800', '#F44336', '#9C27B0']
    for i, row in var_results.iterrows():
        fig.add_vline(x=-row['VaR_INR']/1e5, line_dash="dash",
                      line_color=colors[i], annotation_text=f"VaR {row['Confidence']}")
    fig.update_layout(title='P&L Distribution with VaR Lines', xaxis_title='P&L (₹ Lakhs)',
                      yaxis_title='Count', height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Predefined Stress Test Scenarios")
    stress = run_stress_tests(port_dur, port_conv, total_mv)

    fig = px.bar(stress, x='Total_PnL_INR', y='Scenario', orientation='h',
                 color='Total_PnL_INR', color_continuous_scale='RdYlGn',
                 title='Stress Test Results', text='PnL_Pct')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(stress, use_container_width=True, hide_index=True)
