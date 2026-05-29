"""
Convexity Sensitivity AI Agent — Streamlit Dashboard
=====================================================
Main dashboard entry point with navigation to all analysis pages.
"""

import streamlit as st
import pandas as pd
import sys, os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Convexity Sensitivity AI Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0f3460;
        color: white;
    }
    .metric-card h3 {
        color: #e94560;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #eee;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0f3460 0%, #533483 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e94560;
    }

    /* Achievement badges */
    .achievement {
        display: inline-block;
        background: linear-gradient(135deg, #f7971e, #ffd200);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        color: #1a1a2e;
    }

    /* Tables */
    .dataframe {
        font-size: 0.85rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Landing Page
# ─────────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# Convexity AI Agent")
        st.markdown("---")
        st.markdown("""
        ### Navigation
        Use the sidebar pages to explore:

        1. **Portfolio Analytics**
        2. **Yield Curve**
        3. **Monte Carlo**
        4. **ML Predictions**
        5. **Bond Risk Lab**
        6. **Validation**
        """)
        st.markdown("---")
        st.markdown("*Built with Python, Streamlit, TensorFlow, XGBoost*")

    # Main content
    st.markdown("# Convexity Sensitivity AI Agent")
    st.markdown("### Bond Portfolio Duration, Convexity & Risk Analytics Platform")
    st.markdown("---")

    # Hero metrics
    try:
        from src.utils import load_bond_portfolio, load_monte_carlo, load_yield_curve

        bond_df = load_bond_portfolio()
        mc_df = load_monte_carlo()

        total_mv = bond_df['MarketValue_INR'].sum()
        weights = bond_df['MarketValue_INR'] / total_mv
        port_dur = (weights * bond_df['ModifiedDuration']).sum()
        port_conv = (weights * bond_df['Convexity']).sum()
        port_dv01 = (bond_df['ModifiedDuration'] * bond_df['MarketValue_INR'] * 0.0001).sum()
        port_ytm = (weights * bond_df['YieldToMaturity']).sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(" Total Bonds", f"{len(bond_df)}")
        with col2:
            st.metric(" Portfolio Value", f"₹{total_mv/1e7:.1f} Cr")
        with col3:
            st.metric(" Modified Duration", f"{port_dur:.2f} yrs")
        with col4:
            st.metric(" Portfolio YTM", f"{port_ytm*100:.2f}%")

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(" Convexity", f"{port_conv:.1f}")
        with col6:
            st.metric(" DV01 (Total)", f"₹{port_dv01/1e5:.1f} L")
        with col7:
            st.metric(" MC Scenarios", f"{len(mc_df)}")
        with col8:
            n_sectors = bond_df['Sector'].nunique()
            st.metric(" Sectors", f"{n_sectors}")

        st.markdown("---")

        # Quick summary cards
        st.markdown("### Quick Insights")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Duration & Risk Profile")

            # Sector breakdown
            sector_summary = bond_df.groupby('Sector').apply(
                lambda g: pd.Series({
                    'Bonds': len(g),
                    'Weight': f"{g['MarketValue_INR'].sum()/total_mv*100:.1f}%",
                    'Avg Duration': f"{(g['ModifiedDuration']*g['MarketValue_INR']).sum()/g['MarketValue_INR'].sum():.2f}",
                    'Avg Convexity': f"{(g['Convexity']*g['MarketValue_INR']).sum()/g['MarketValue_INR'].sum():.1f}",
                }),
                include_groups=False
            ).reset_index()
            st.dataframe(sector_summary, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### Sensitivity Summary")
            shocks = [-100, -50, 50, 100]
            sens_data = []
            for s in shocks:
                dy = s / 10000.0
                pnl = (-port_dur * dy + 0.5 * port_conv * dy**2) * total_mv
                sens_data.append({
                    'Shock': f"{s:+d} bps",
                    'P&L': f"₹{pnl/1e5:+,.0f} L",
                    'P&L %': f"{pnl/total_mv*100:+.2f}%"
                })
            st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("""
        ### Platform Features

        | Module | Description | Status |
        |--------|------------|--------|
        | **Portfolio Analytics** | Duration, convexity, DV01 framework | Active |
        | **Yield Curve** | NSS fitting, DV01 ladder, PCA analysis | Active |
        | **Monte Carlo** | Vasicek/CIR models, VaR, stress tests | Active |
        | **ML Predictions** | RF, XGBoost, Neural Network ensemble | Active |
        | **Bond Risk Lab** | Gamified training platform | Active |
        | **Validation** | Multi-scenario shock validation | Active |
        """)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure the CSV data files are in the project root directory.")

    st.markdown('<div class="footer">Convexity Sensitivity AI Agent v1.0 | Bond Risk Analytics Platform</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
