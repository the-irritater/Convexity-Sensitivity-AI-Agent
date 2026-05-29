"""
Shared Utilities Module
=======================
Common functions, constants, and data loading utilities used across all parts.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
POWERBI_DIR = OUTPUT_DIR / "powerbi_exports"

# Ensure output directories exist
for d in [FIGURES_DIR, REPORTS_DIR, POWERBI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Data Loading Functions
# ─────────────────────────────────────────────────────────────

def load_bond_portfolio():
    """Load and clean the bond portfolio dataset."""
    df = pd.read_csv(DATA_DIR / "bond_portfolio_data.csv")
    # Clean column names (remove trailing whitespace/carriage returns)
    df.columns = df.columns.str.strip()
    # Parse dates
    for col in ['IssueDate', 'MaturityDate', 'ValuationDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'CallDate' in df.columns:
        df['CallDate'] = pd.to_datetime(df['CallDate'], errors='coerce')
    # Convert boolean-like strings
    for col in ['IsCallable', 'IsPutable', 'IsFloatingRate']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(
                {'True': True, 'False': False, 'true': True, 'false': False}
            ).fillna(False)
    return df


def load_yield_curve():
    """Load and clean the yield curve history dataset."""
    df = pd.read_csv(DATA_DIR / "yield_curve_history.csv")
    df.columns = df.columns.str.strip()
    if 'CurveDate' in df.columns:
        df['CurveDate'] = pd.to_datetime(df['CurveDate'], errors='coerce')
    return df


def load_monte_carlo():
    """Load and clean the Monte Carlo scenarios dataset."""
    df = pd.read_csv(DATA_DIR / "monte_carlo_scenarios.csv")
    df.columns = df.columns.str.strip()
    return df


# ─────────────────────────────────────────────────────────────
# Financial Calculation Helpers
# ─────────────────────────────────────────────────────────────

def discount_factor(rate, t):
    """Calculate discount factor: 1 / (1 + r)^t"""
    return 1.0 / (1.0 + rate) ** t


def present_value_cashflows(cashflows, times, ytm):
    """
    Calculate present value of a series of cashflows.

    Parameters
    ----------
    cashflows : array-like — Cash flow amounts
    times : array-like — Time to each cash flow (in years)
    ytm : float — Yield to maturity (annualized, decimal)

    Returns
    -------
    float — Total present value
    """
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)
    dfs = np.array([discount_factor(ytm, t) for t in times])
    return np.sum(cashflows * dfs)


def generate_bond_cashflows(face_value, coupon_rate, coupon_freq, years_to_maturity):
    """
    Generate cashflow schedule for a fixed-rate bond.

    Parameters
    ----------
    face_value : float — Face/par value
    coupon_rate : float — Annual coupon rate (decimal)
    coupon_freq : int — Number of coupon payments per year
    years_to_maturity : float — Years until maturity

    Returns
    -------
    tuple — (cashflows, times) arrays
    """
    period_coupon = face_value * coupon_rate / coupon_freq
    n_periods = int(np.ceil(years_to_maturity * coupon_freq))

    if n_periods <= 0:
        return np.array([face_value]), np.array([years_to_maturity])

    times = np.array([(i + 1) / coupon_freq for i in range(n_periods)])
    # Adjust last time to exactly match maturity
    times[-1] = min(times[-1], years_to_maturity)

    cashflows = np.full(n_periods, period_coupon)
    cashflows[-1] += face_value # Principal repayment at maturity

    return cashflows, times


# ─────────────────────────────────────────────────────────────
# Formatting Helpers
# ─────────────────────────────────────────────────────────────

def fmt_pct(value, decimals=2):
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def fmt_bps(value, decimals=1):
    """Format a decimal as basis points string."""
    return f"{value * 10000:.{decimals}f} bps"


def fmt_inr(value, decimals=0):
    """Format value as INR currency."""
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif abs(value) >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.{decimals}f}"


def print_section_header(title, char="═", width=70):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def print_subsection(title, char="─", width=50):
    """Print a formatted subsection header."""
    print(f"\n {char * width}")
    print(f" {title}")
    print(f" {char * width}\n")
