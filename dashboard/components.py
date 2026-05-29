"""Shared UI components for the dashboard."""

import streamlit as st


def metric_card(title, value, icon=""):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <div class="value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def format_inr(value):
    """Format INR value."""
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:.2f} L"
    return f"₹{value:,.0f}"
