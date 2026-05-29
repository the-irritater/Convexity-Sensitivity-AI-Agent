"""Dashboard Page 2: Yield Curve Analysis"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import load_bond_portfolio, load_yield_curve
from src.part2_yield_curve import (NelsonSiegelSvensson, NelsonSiegel, YieldCurveSpline,
                                    compute_portfolio_dv01_ladder, yield_curve_pca)

st.set_page_config(page_title="Yield Curve", page_icon="", layout="wide")
st.title(" Yield Curve Modelling & DV01 Sensitivity")

yc_df = load_yield_curve()
bond_df = load_bond_portfolio()

tab1, tab2, tab3, tab4 = st.tabs(["Yield Curves", "Model Fitting", "DV01 Ladder", "PCA Analysis"])

with tab1:
    pivot = yc_df.pivot_table(values='Yield', index='CurveDate', columns='Tenor_Years')
    dates = sorted(pivot.index.unique())

    selected_dates = st.multiselect("Select curve dates", dates, default=dates[-3:] if len(dates) >= 3 else dates)

    fig = go.Figure()
    for date in selected_dates:
        row = pivot.loc[date].dropna()
        fig.add_trace(go.Scatter(x=row.index, y=row.values*100, mode='lines+markers', name=str(date)[:10]))
    fig.update_layout(title='Historical INR Yield Curves', xaxis_title='Tenor (Years)',
                      yaxis_title='Yield (%)', height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    latest_date = yc_df['CurveDate'].max()
    latest = yc_df[yc_df['CurveDate'] == latest_date].sort_values('Tenor_Years')
    tenors = latest['Tenor_Years'].values
    yields = latest['Yield'].values

    st.markdown(f"**Fitting to latest curve: {str(latest_date)[:10]}**")

    nss = NelsonSiegelSvensson()
    nss.fit(tenors, yields)
    ns = NelsonSiegel()
    ns.fit(tenors, yields)
    spline = YieldCurveSpline()
    spline.fit(tenors, yields)

    smooth_t = np.linspace(min(tenors), max(tenors), 200)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tenors, y=yields*100, mode='markers', name='Actual',
                             marker=dict(size=10, color='red')))
    fig.add_trace(go.Scatter(x=smooth_t, y=nss.predict(smooth_t)*100, mode='lines',
                             name='NSS', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=smooth_t, y=ns.predict(smooth_t)*100, mode='lines',
                             name='Nelson-Siegel', line=dict(width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=smooth_t, y=spline.predict(smooth_t)*100, mode='lines',
                             name='Cubic Spline', line=dict(width=2, dash='dot')))
    fig.update_layout(title='Yield Curve Model Fits', xaxis_title='Tenor (Years)',
                      yaxis_title='Yield (%)', height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**NSS Parameters:**")
    factors = nss.get_factors()
    st.json(factors)

with tab3:
    ladder = compute_portfolio_dv01_ladder(bond_df)
    if not ladder.empty:
        fig = px.bar(ladder, x='KeyRateBucket', y='Total_DV01', title='DV01 Ladder by Tenor Bucket',
                     color='Total_DV01', color_continuous_scale='Reds', text='Total_DV01')
        fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ladder, use_container_width=True, hide_index=True)

with tab4:
    pca_results = yield_curve_pca(yc_df)
    if pca_results:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            colors = ['#1976D2', '#E64A19', '#4CAF50']
            for i, (comp, name) in enumerate(zip(pca_results['components'], pca_results['component_names'])):
                fig.add_trace(go.Scatter(x=pca_results['tenors'], y=comp, mode='lines+markers',
                                         name=f"{name} ({pca_results['explained_variance_ratio'][i]*100:.1f}%)",
                                         line=dict(color=colors[i])))
            fig.update_layout(title='PCA Factor Loadings', xaxis_title='Tenor', yaxis_title='Loading', height=400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(x=[f"PC{i+1}" for i in range(len(pca_results['explained_variance_ratio']))],
                         y=pca_results['explained_variance_ratio']*100,
                         title='Explained Variance by Component',
                         labels={'x': 'Component', 'y': 'Variance (%)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for PCA analysis")
