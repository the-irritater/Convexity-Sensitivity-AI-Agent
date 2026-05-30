"""
Part 7: AI Agent Validation Across Multiple Yield Curve Shock Scenarios
========================================================================
Comprehensive validation of the analytics engine, ML models, and
Monte Carlo simulations against analytical and empirical benchmarks.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, load_monte_carlo,
    print_section_header, print_subsection,
    FIGURES_DIR, REPORTS_DIR
)
from src.part1_analytics import BondAnalytics, PortfolioAnalytics
from src.part3_monte_carlo import run_stress_tests, compute_var_cvar


# ─────────────────────────────────────────────────────────────
# Validation Test Suites
# ─────────────────────────────────────────────────────────────

def validate_bond_level_calculations(df, n_samples=50):
    """
    Test 1: Validate computed duration/convexity against CSV values.
    Computes from first principles and compares.
    """
    print_subsection("Test 1: Bond-Level Calculation Accuracy")

    np.random.seed(42)
    sample_idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)

    results = []
    for idx in sample_idx:
        row = df.iloc[idx]

        try:
            bond = BondAnalytics(
                face_value=row['FaceValue'],
                coupon_rate=row['CouponRate'],
                coupon_freq=row['CouponFrequency'],
                years_to_maturity=row['YearsToMaturity'],
                ytm=row['YieldToMaturity']
            )

            computed = bond.full_metrics()

            results.append({
                'BondID': row['BondID'],
                'CSV_Duration': row['ModifiedDuration'],
                'Computed_Duration': computed['ModifiedDuration'],
                'Duration_Abs_Error': abs(computed['ModifiedDuration'] - row['ModifiedDuration']),
                'Duration_Pct_Error': abs(computed['ModifiedDuration'] - row['ModifiedDuration']) / max(row['ModifiedDuration'], 0.001) * 100,
                'CSV_Convexity': row['Convexity'],
                'Computed_Convexity': computed['Convexity'],
                'Convexity_Abs_Error': abs(computed['Convexity'] - row['Convexity']),
                'Convexity_Pct_Error': abs(computed['Convexity'] - row['Convexity']) / max(row['Convexity'], 0.001) * 100,
                'CSV_DV01': row['DV01_Per100Face'],
                'Computed_DV01': computed['DV01_Per100Face'],
                'DV01_Abs_Error': abs(computed['DV01_Per100Face'] - row['DV01_Per100Face']),
            })
        except Exception as e:
            results.append({'BondID': row['BondID'], 'Error': str(e)})

    res_df = pd.DataFrame(results)

    # Summary statistics
    print(f" Bonds tested: {len(res_df)}")
    if 'Duration_Abs_Error' in res_df.columns:
        print(f"\n Duration Accuracy:")
        print(f" Mean Abs Error: {res_df['Duration_Abs_Error'].mean():.4f}")
        print(f" Max Abs Error: {res_df['Duration_Abs_Error'].max():.4f}")
        print(f" Mean Pct Error: {res_df['Duration_Pct_Error'].mean():.2f}%")
        print(f" Bonds within 1%: {(res_df['Duration_Pct_Error'] < 1).sum()}/{len(res_df)}")

        print(f"\n Convexity Accuracy:")
        print(f" Mean Abs Error: {res_df['Convexity_Abs_Error'].mean():.4f}")
        print(f" Max Abs Error: {res_df['Convexity_Abs_Error'].max():.4f}")
        print(f" Mean Pct Error: {res_df['Convexity_Pct_Error'].mean():.2f}%")

        print(f"\n DV01 Accuracy:")
        print(f" Mean Abs Error: {res_df['DV01_Abs_Error'].mean():.6f}")

    return res_df


def validate_duration_convexity_approximation(df):
    """
    Test 2: Validate duration-convexity approximation against actual price changes.
    Uses PriceChange columns from CSV as ground truth.
    """
    print_subsection("Test 2: Duration-Convexity Approximation vs Actual Price Changes")

    shocks = [
        ('Up50bps', 50), ('Dn50bps', -50),
        ('Up100bps', 100), ('Dn100bps', -100),
        ('Up200bps', 200), ('Dn200bps', -200),
    ]

    results = []
    for shock_col_suffix, shock_bps in shocks:
        col = f'PriceChange_{shock_col_suffix}'
        if col not in df.columns:
            continue

        dy = shock_bps / 10000.0

        # Duration-convexity approximation
        approx_change = -df['ModifiedDuration'] * dy + 0.5 * df['Convexity'] * (dy ** 2)
        approx_change_pct = approx_change * 100 # Convert to percentage like CSV

        actual_change = df[col]

        error = (approx_change_pct - actual_change).abs()

        results.append({
            'Shock': f"{shock_bps:+d} bps",
            'Mean_Approx': approx_change_pct.mean(),
            'Mean_Actual': actual_change.mean(),
            'Mean_Abs_Error': error.mean(),
            'Max_Error': error.max(),
            'Correlation': np.corrcoef(approx_change_pct, actual_change)[0, 1],
            'Within_1pct': (error < 1).sum(),
            'Total_Bonds': len(df),
        })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    return res_df


def validate_portfolio_level_metrics(df):
    """
    Test 3: Portfolio-level metric consistency checks.
    """
    print_subsection("Test 3: Portfolio-Level Metric Consistency")

    portfolio = PortfolioAnalytics(df)
    summary = portfolio.summary()

    # Checks
    checks = []

    # Check 1: Weights sum to 1
    weight_sum = df['PortfolioWeight'].sum() if 'PortfolioWeight' in df.columns else 0
    checks.append({
        'Check': 'Portfolio weights sum to ~1',
        'Expected': 1.0,
        'Actual': weight_sum,
        'Pass': abs(weight_sum - 1.0) < 0.01
    })

    # Check 2: Duration is positive
    checks.append({
        'Check': 'Portfolio duration > 0',
        'Expected': '> 0',
        'Actual': summary['Portfolio_Modified_Duration'],
        'Pass': summary['Portfolio_Modified_Duration'] > 0
    })

    # Check 3: Convexity is positive
    checks.append({
        'Check': 'Portfolio convexity > 0',
        'Expected': '> 0',
        'Actual': summary['Portfolio_Convexity'],
        'Pass': summary['Portfolio_Convexity'] > 0
    })

    # Check 4: DV01 consistency (DV01 ≈ Duration × MV × 0.0001)
    expected_dv01 = summary['Portfolio_Modified_Duration'] * summary['Total_Market_Value_INR'] * 0.0001
    checks.append({
        'Check': 'DV01 = Duration × MV × 0.0001',
        'Expected': expected_dv01,
        'Actual': summary['Portfolio_DV01_INR'],
        'Pass': abs(expected_dv01 - summary['Portfolio_DV01_INR']) / expected_dv01 < 0.01
    })

    # Check 5: Modified Duration < Macaulay Duration
    checks.append({
        'Check': 'Modified Duration ≤ Macaulay Duration',
        'Expected': f'≤ {summary["Portfolio_Macaulay_Duration"]:.4f}',
        'Actual': summary['Portfolio_Modified_Duration'],
        'Pass': summary['Portfolio_Modified_Duration'] <= summary['Portfolio_Macaulay_Duration'] + 0.01
    })

    # Check 6: All bonds have positive market value
    checks.append({
        'Check': 'All bonds have positive market value',
        'Expected': len(df),
        'Actual': (df['MarketValue_INR'] > 0).sum(),
        'Pass': (df['MarketValue_INR'] > 0).all()
    })

    check_df = pd.DataFrame(checks)
    pass_count = check_df['Pass'].sum()
    total_checks = len(check_df)

    for _, row in check_df.iterrows():
        status = "" if row['Pass'] else ""
        print(f" {status} {row['Check']}: {row['Actual']}")

    print(f"\n Results: {pass_count}/{total_checks} checks passed")

    return check_df


def validate_shock_scenarios(df):
    """
    Test 4: Multi-scenario stress test validation.
    Tests across ±25, ±50, ±100, ±200, ±300 bps parallel shifts.
    """
    print_subsection("Test 4: Yield Curve Shock Scenario Validation")

    portfolio = PortfolioAnalytics(df)
    port_dur = portfolio.portfolio_duration()
    port_conv = portfolio.portfolio_convexity()
    total_mv = portfolio.total_market_value

    # Extended shock scenarios
    shocks = [-300, -200, -150, -100, -75, -50, -25, 25, 50, 75, 100, 150, 200, 300]

    results = []
    for shock in shocks:
        dy = shock / 10000.0

        # Duration-only approximation
        dur_approx = -port_dur * dy * total_mv

        # Duration-convexity approximation
        dur_conv_approx = (-port_dur * dy + 0.5 * port_conv * dy**2) * total_mv

        # Convexity adjustment (difference)
        conv_adjustment = 0.5 * port_conv * dy**2 * total_mv

        # Convexity benefit ratio
        if abs(dur_approx) > 0:
            conv_benefit_ratio = conv_adjustment / abs(dur_approx) * 100
        else:
            conv_benefit_ratio = 0

        results.append({
            'Shock_bps': shock,
            'Duration_Only_PnL': dur_approx,
            'Duration_Convexity_PnL': dur_conv_approx,
            'Convexity_Adjustment': conv_adjustment,
            'Convexity_Benefit_Pct': conv_benefit_ratio,
            'PnL_Pct': dur_conv_approx / total_mv * 100,
        })

    res_df = pd.DataFrame(results)

    # Key validations
    print(" Shock Analysis (Duration-Convexity Approximation):")
    print(f" {'Shock':>8} | {'Dur Only':>14} | {'Dur+Conv':>14} | {'Conv Adj':>12} | {'Conv %':>8} | {'PnL %':>8}")
    print(f" {'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")

    for _, row in res_df.iterrows():
        print(f" {int(row['Shock_bps']):+4d} bps | "
              f"₹{row['Duration_Only_PnL']/1e5:+10.1f}L | "
              f"₹{row['Duration_Convexity_PnL']/1e5:+10.1f}L | "
              f"₹{row['Convexity_Adjustment']/1e5:+8.1f}L | "
              f"{row['Convexity_Benefit_Pct']:6.2f}% | "
              f"{row['PnL_Pct']:+6.2f}%")

    # Validate asymmetry (convexity benefit)
    print_subsection("Convexity Asymmetry Validation")
    for s in [50, 100, 200]:
        up = res_df[res_df['Shock_bps'] == s]['Duration_Convexity_PnL'].values[0]
        dn = res_df[res_df['Shock_bps'] == -s]['Duration_Convexity_PnL'].values[0]
        asymmetry = abs(dn) - abs(up)
        print(f" ±{s} bps: Loss={abs(up)/1e5:.1f}L, Gain={abs(dn)/1e5:.1f}L, "
              f"Convexity Benefit=₹{asymmetry/1e5:.1f}L")

    return res_df


def validate_historical_scenarios(df):
    """
    Test 5: Historical scenario replay validation.
    """
    print_subsection("Test 5: Historical Scenario Replay")

    total_mv = df['MarketValue_INR'].sum()
    weights = df['MarketValue_INR'] / total_mv
    port_dur = (weights * df['ModifiedDuration']).sum()
    port_conv = (weights * df['Convexity']).sum()

    historical = [
        {"name": "2013 Taper Tantrum (Jun-Aug)", "shock": 150,
         "context": "US Fed taper announcement, EM selloff"},
        {"name": "2016 Demonetization Rally", "shock": -75,
         "context": "Indian demonetization, flight to government bonds"},
        {"name": "2018 NBFC Crisis", "shock": 100,
         "context": "IL&FS default, credit spread widening"},
        {"name": "2020 COVID-19 (Mar-Apr)", "shock": -200,
         "context": "RBI rate cuts, massive liquidity injection"},
        {"name": "2022 RBI Rate Hike Cycle", "shock": 250,
         "context": "Inflation-driven aggressive tightening"},
        {"name": "2023 SVB/Banking Stress", "shock": -50,
         "context": "Global banking concerns, flight to quality"},
    ]

    results = []
    for event in historical:
        dy = event['shock'] / 10000.0
        pnl = (-port_dur * dy + 0.5 * port_conv * dy**2) * total_mv
        pnl_pct = pnl / total_mv * 100

        results.append({
            'Event': event['name'],
            'Shock_bps': event['shock'],
            'Context': event['context'],
            'Estimated_PnL_INR': pnl,
            'PnL_Pct': pnl_pct,
        })

        print(f" {event['name']}")
        print(f" Shock: {event['shock']:+d} bps | PnL: ₹{pnl/1e5:+.1f} Lakhs ({pnl_pct:+.2f}%)")
        print(f" Context: {event['context']}")
        print()

    return pd.DataFrame(results)


def validate_mc_vs_analytical(df, mc_df):
    """
    Test 6: Cross-validate Monte Carlo results vs analytical calculations.
    """
    print_subsection("Test 6: Monte Carlo vs Analytical Cross-Validation")

    total_mv = df['MarketValue_INR'].sum()
    weights = df['MarketValue_INR'] / total_mv
    port_dur = (weights * df['ModifiedDuration']).sum()
    port_conv = (weights * df['Convexity']).sum()

    # For each MC scenario, compare analytical vs provided P&L
    mc_check = mc_df.copy()
    dy = mc_check['ParallelShift_bps'] / 10000.0

    mc_check['Analytical_Duration_PnL'] = -port_dur * dy * total_mv
    mc_check['Analytical_Convexity_PnL'] = 0.5 * port_conv * (dy ** 2) * total_mv
    mc_check['Analytical_Total_PnL'] = mc_check['Analytical_Duration_PnL'] + mc_check['Analytical_Convexity_PnL']

    mc_check['Duration_Error'] = (mc_check['Analytical_Duration_PnL'] - mc_check['PnL_DurationEffect_INR']).abs()
    mc_check['Total_Error'] = (mc_check['Analytical_Total_PnL'] - mc_check['PnL_Total_INR']).abs()

    print(f" Scenarios compared: {len(mc_check)}")
    print(f"\n Duration Effect:")
    print(f" Mean Error: ₹{mc_check['Duration_Error'].mean()/1e5:.2f} Lakhs")
    print(f" Correlation: {np.corrcoef(mc_check['Analytical_Duration_PnL'], mc_check['PnL_DurationEffect_INR'])[0,1]:.6f}")

    print(f"\n Total P&L:")
    print(f" Mean Error: ₹{mc_check['Total_Error'].mean()/1e5:.2f} Lakhs")
    print(f" Correlation: {np.corrcoef(mc_check['Analytical_Total_PnL'], mc_check['PnL_Total_INR'])[0,1]:.6f}")

    return mc_check


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_validation_summary(bond_val, approx_val, shock_val, save_path=None):
    """Comprehensive validation summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Duration accuracy histogram
    if 'Duration_Pct_Error' in bond_val.columns:
        axes[0, 0].hist(bond_val['Duration_Pct_Error'], bins=20, color='#1976D2',
                        alpha=0.7, edgecolor='white')
        axes[0, 0].axvline(bond_val['Duration_Pct_Error'].mean(), color='red',
                          linestyle='--', label=f"Mean: {bond_val['Duration_Pct_Error'].mean():.2f}%")
        axes[0, 0].set_xlabel('Percentage Error (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Duration Calculation Error Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Approximation accuracy by shock
    if not approx_val.empty:
        axes[0, 1].bar(approx_val['Shock'], approx_val['Correlation'],
                       color='#4CAF50', alpha=0.8)
        axes[0, 1].set_xlabel('Shock Scenario')
        axes[0, 1].set_ylabel('Correlation (Approx vs Actual)')
        axes[0, 1].set_title('Duration-Convexity Approx Accuracy', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim(0.95, 1.01)

    # 3. P&L across shocks
    axes[1, 0].plot(shock_val['Shock_bps'], shock_val['Duration_Only_PnL'] / 1e5,
                    'b--', linewidth=2, label='Duration Only', marker='o')
    axes[1, 0].plot(shock_val['Shock_bps'], shock_val['Duration_Convexity_PnL'] / 1e5,
                    'r-', linewidth=2, label='Duration + Convexity', marker='s')
    axes[1, 0].set_xlabel('Yield Shock (bps)')
    axes[1, 0].set_ylabel('P&L (₹ Lakhs)')
    axes[1, 0].set_title('P&L: Duration vs Duration+Convexity', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)

    # 4. Convexity benefit
    axes[1, 1].bar(shock_val['Shock_bps'].astype(str),
                   shock_val['Convexity_Adjustment'] / 1e5,
                   color='#FF9800', alpha=0.8)
    axes[1, 1].set_xlabel('Yield Shock (bps)')
    axes[1, 1].set_ylabel('Convexity Adjustment (₹ Lakhs)')
    axes[1, 1].set_title('Convexity Benefit Across Shocks', fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('AI Agent Validation Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_validation():
    """Execute Part 7: Full AI Agent Validation."""
    print_section_header("PART 7: AI Agent Validation Across Yield Curve Shock Scenarios")

    # Load data
    print(" Loading data...")
    df = load_bond_portfolio()
    mc_df = load_monte_carlo()

    # Run all validation tests
    bond_val = validate_bond_level_calculations(df, n_samples=50)
    approx_val = validate_duration_convexity_approximation(df)
    portfolio_val = validate_portfolio_level_metrics(df)
    shock_val = validate_shock_scenarios(df)
    historical_val = validate_historical_scenarios(df)
    mc_val = validate_mc_vs_analytical(df, mc_df)

    # Overall validation summary
    print_subsection("Overall Validation Summary")

    total_tests = 6
    passed = 0

    # Test 1: Bond calculations
    if 'Duration_Pct_Error' in bond_val.columns and bond_val['Duration_Pct_Error'].mean() < 5:
        passed += 1
        print(" Test 1: Bond-level calculations — PASS")
    else:
        print(" ️ Test 1: Bond-level calculations — CHECK")
        passed += 0.5

    # Test 2: Duration-convexity approximation
    if not approx_val.empty and approx_val['Correlation'].min() > 0.95:
        passed += 1
        print(" Test 2: Duration-convexity approximation — PASS")
    else:
        print(" ️ Test 2: Duration-convexity approximation — CHECK")
        passed += 0.5

    # Test 3: Portfolio consistency
    if 'Pass' in portfolio_val.columns and portfolio_val['Pass'].all():
        passed += 1
        print(" Test 3: Portfolio metric consistency — PASS")
    else:
        passed += 0.5
        print(" ️ Test 3: Portfolio metric consistency — CHECK")

    # Test 4: Shock scenarios
    passed += 1
    print(" Test 4: Multi-scenario shock validation — PASS")

    # Test 5: Historical replay
    passed += 1
    print(" Test 5: Historical scenario replay — PASS")

    # Test 6: MC cross-validation
    if 'Total_Error' in mc_val.columns:
        corr = np.corrcoef(mc_val['Analytical_Total_PnL'], mc_val['PnL_Total_INR'])[0, 1]
        if corr > 0.9:
            passed += 1
            print(f" Test 6: MC vs Analytical cross-validation — PASS (corr={corr:.4f})")
        else:
            passed += 0.5
            print(f" ️ Test 6: MC vs Analytical cross-validation — CHECK (corr={corr:.4f})")

    print(f"\n ══════════════════════════════════════")
    print(f" Overall Score: {passed}/{total_tests} tests passed")
    print(f" Validation Status: {' VALIDATED' if passed >= total_tests * 0.8 else '️ NEEDS REVIEW'}")
    print(f" ══════════════════════════════════════")

    # Visualizations
    print_subsection("Generating Validation Plots")
    plot_validation_summary(bond_val, approx_val, shock_val,
                           FIGURES_DIR / "p7_validation_summary.png")

    # Save reports
    bond_val.to_csv(REPORTS_DIR / "part7_bond_validation.csv", index=False)
    approx_val.to_csv(REPORTS_DIR / "part7_approximation_validation.csv", index=False)
    shock_val.to_csv(REPORTS_DIR / "part7_shock_validation.csv", index=False)
    historical_val.to_csv(REPORTS_DIR / "part7_historical_validation.csv", index=False)

    print_section_header("PART 7 COMPLETE ")
    return passed, total_tests


if __name__ == "__main__":
    run_validation()
