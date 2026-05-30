"""
VaR Backtesting Module
=======================
Backtesting Value at Risk calculations using the Kupiec Proportion
of Failures (POF) test and comparison of parametric vs Monte Carlo VaR.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, load_monte_carlo,
    fmt_inr, print_section_header, print_subsection,
    FIGURES_DIR, REPORTS_DIR
)
from src.part3_monte_carlo import compute_var_cvar, parametric_var, historical_var


# ─────────────────────────────────────────────────────────────
# Kupiec POF Test
# ─────────────────────────────────────────────────────────────

def kupiec_pof_test(violations, n_observations, confidence_level=0.95):
    """
    Kupiec Proportion of Failures (POF) test for VaR model adequacy.

    Tests whether the observed violation rate is consistent with
    the expected violation rate at the given confidence level.

    Parameters
    ----------
    violations : int — Number of VaR breaches observed
    n_observations : int — Total number of observations
    confidence_level : float — VaR confidence level (e.g., 0.95)

    Returns
    -------
    dict — Test statistics and pass/fail result
    """
    expected_rate = 1 - confidence_level
    observed_rate = violations / n_observations

    # Avoid log(0) issues
    eps = 1e-10
    p = max(observed_rate, eps)
    p0 = expected_rate

    # Likelihood ratio test statistic
    if violations == 0:
        lr_stat = -2 * n_observations * np.log(1 - p0)
    elif violations == n_observations:
        lr_stat = -2 * n_observations * np.log(p0)
    else:
        lr_stat = -2 * (
            np.log((1 - p0) ** (n_observations - violations) * p0 ** violations)
            - np.log((1 - p) ** (n_observations - violations) * p ** violations)
        )

    # LR ~ chi-squared with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    # Critical value at 5% significance
    critical_value = stats.chi2.ppf(0.95, df=1)

    return {
        'Confidence_Level': f"{confidence_level*100:.0f}%",
        'N_Observations': n_observations,
        'N_Violations': violations,
        'Expected_Rate': expected_rate,
        'Observed_Rate': observed_rate,
        'LR_Statistic': lr_stat,
        'Critical_Value': critical_value,
        'P_Value': p_value,
        'Pass': lr_stat < critical_value,
        'Result': 'PASS ✅' if lr_stat < critical_value else 'FAIL ❌',
    }


# ─────────────────────────────────────────────────────────────
# VaR Backtesting Engine
# ─────────────────────────────────────────────────────────────

class VaRBacktester:
    """
    VaR backtesting engine that compares different VaR methods
    and runs the Kupiec POF test.
    """

    def __init__(self, pnl_series, confidence_levels=None):
        """
        Parameters
        ----------
        pnl_series : array-like — Historical or simulated P&L series
        confidence_levels : list — Confidence levels to test
        """
        self.pnl = np.asarray(pnl_series, dtype=float)
        self.n = len(self.pnl)
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]

    def compute_all_var_methods(self):
        """
        Compute VaR using multiple methods and compare.

        Returns
        -------
        pd.DataFrame — VaR comparison across methods
        """
        results = []
        for cl in self.confidence_levels:
            alpha = 1 - cl

            # Method 1: Historical VaR
            hist_var = -np.percentile(self.pnl, alpha * 100)

            # Method 2: Parametric (Gaussian) VaR
            mu = np.mean(self.pnl)
            sigma = np.std(self.pnl)
            z = stats.norm.ppf(alpha)
            param_var = -(mu + z * sigma)

            # Method 3: Cornish-Fisher VaR (adjusts for skew/kurtosis)
            skew = stats.skew(self.pnl)
            kurt = stats.kurtosis(self.pnl)
            z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24 - (2*z**3 - 5*z) * skew**2 / 36
            cf_var = -(mu + z_cf * sigma)

            # Difference analysis
            diff_pct = abs(hist_var - param_var) / hist_var * 100 if hist_var > 0 else 0

            results.append({
                'Confidence': f"{cl*100:.0f}%",
                'Historical_VaR': hist_var,
                'Parametric_VaR': param_var,
                'CornishFisher_VaR': cf_var,
                'Hist_Param_Diff_Pct': diff_pct,
                'Within_5pct': diff_pct < 5,
            })

        return pd.DataFrame(results)

    def run_backtest(self, var_estimates=None, window_size=None):
        """
        Run VaR backtest using rolling window or provided estimates.

        Parameters
        ----------
        var_estimates : dict — {confidence_level: var_value}
        window_size : int — Rolling window size for expanding VaR

        Returns
        -------
        pd.DataFrame — Kupiec test results for each confidence level
        """
        results = []

        for cl in self.confidence_levels:
            alpha = 1 - cl

            if var_estimates and cl in var_estimates:
                var_val = var_estimates[cl]
            else:
                var_val = -np.percentile(self.pnl, alpha * 100)

            # Count violations (actual loss > VaR)
            violations = np.sum(self.pnl < -var_val)

            # Kupiec POF test
            kupiec = kupiec_pof_test(violations, self.n, cl)
            results.append(kupiec)

        return pd.DataFrame(results)

    def traffic_light_test(self):
        """
        Basel II traffic light test for VaR adequacy.

        Uses 99% VaR and classifies by violation count:
        - Green: ≤ 4 violations in 250 days
        - Yellow: 5-9 violations
        - Red: ≥ 10 violations
        """
        var_99 = -np.percentile(self.pnl, 1)
        violations = np.sum(self.pnl < -var_99)

        # Scale to 250-day equivalent
        scaled_violations = violations * 250 / self.n

        if scaled_violations <= 4:
            zone = 'GREEN ✅'
            interpretation = 'VaR model is adequate'
        elif scaled_violations <= 9:
            zone = 'YELLOW ⚠️'
            interpretation = 'VaR model needs monitoring'
        else:
            zone = 'RED ❌'
            interpretation = 'VaR model is inadequate — review required'

        return {
            'VaR_99': var_99,
            'Violations': violations,
            'Scaled_Violations_250d': scaled_violations,
            'Zone': zone,
            'Interpretation': interpretation,
        }


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_var_comparison(var_df, save_path=None):
    """Bar chart comparing VaR methods across confidence levels."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(var_df))
    width = 0.25

    bars1 = ax.bar(x - width, var_df['Historical_VaR'] / 1e5, width,
                   label='Historical', color='#1976D2', alpha=0.8)
    bars2 = ax.bar(x, var_df['Parametric_VaR'] / 1e5, width,
                   label='Parametric', color='#E64A19', alpha=0.8)
    bars3 = ax.bar(x + width, var_df['CornishFisher_VaR'] / 1e5, width,
                   label='Cornish-Fisher', color='#4CAF50', alpha=0.8)

    ax.set_xlabel('Confidence Level', fontsize=12)
    ax.set_ylabel('VaR (₹ Lakhs)', fontsize=12)
    ax.set_title('VaR Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(var_df['Confidence'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_backtest_results(pnl, var_val, confidence, save_path=None):
    """Plot P&L time series with VaR breach markers."""
    fig, ax = plt.subplots(figsize=(14, 6))

    indices = np.arange(len(pnl))
    pnl_lakhs = pnl / 1e5

    ax.plot(indices, pnl_lakhs, 'b-', linewidth=0.5, alpha=0.7, label='P&L')
    ax.axhline(-var_val / 1e5, color='red', linestyle='--', linewidth=2,
               label=f'VaR {confidence} = ₹{var_val/1e5:.1f}L')

    # Mark breaches
    breaches = pnl < -var_val
    if breaches.any():
        ax.scatter(indices[breaches], pnl_lakhs[breaches],
                   color='red', s=50, zorder=5, label=f'Breaches ({breaches.sum()})')

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('P&L (₹ Lakhs)', fontsize=12)
    ax.set_title(f'VaR Backtest — {confidence} Confidence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_kupiec_summary(backtest_df, save_path=None):
    """Summary plot of Kupiec test results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Expected vs observed violation rates
    x = np.arange(len(backtest_df))
    width = 0.35
    axes[0].bar(x - width/2, backtest_df['Expected_Rate'] * 100, width,
                label='Expected', color='#2196F3', alpha=0.8)
    axes[0].bar(x + width/2, backtest_df['Observed_Rate'] * 100, width,
                label='Observed', color='#FF9800', alpha=0.8)
    axes[0].set_xlabel('Confidence Level')
    axes[0].set_ylabel('Violation Rate (%)')
    axes[0].set_title('Expected vs Observed Violation Rate', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(backtest_df['Confidence_Level'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # P-values
    colors = ['#4CAF50' if p else '#F44336' for p in backtest_df['Pass']]
    axes[1].bar(backtest_df['Confidence_Level'], backtest_df['P_Value'],
                color=colors, alpha=0.8)
    axes[1].axhline(0.05, color='red', linestyle='--', linewidth=2, label='5% significance')
    axes[1].set_xlabel('Confidence Level')
    axes[1].set_ylabel('P-Value')
    axes[1].set_title('Kupiec POF Test P-Values', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_var_backtest():
    """Execute VaR Backtesting with Kupiec POF Test."""
    print_section_header("VaR Backtesting & Validation")

    # 1. Load data
    print("  Loading Monte Carlo scenarios...")
    mc_df = load_monte_carlo()
    pnl = mc_df['PnL_Total_INR'].values

    print(f"  Scenarios: {len(pnl)}")
    print(f"  P&L range: [{fmt_inr(pnl.min())}, {fmt_inr(pnl.max())}]")
    print(f"  P&L mean: {fmt_inr(pnl.mean())}, std: {fmt_inr(pnl.std())}")

    # 2. VaR method comparison
    print_subsection("VaR Method Comparison")
    backtester = VaRBacktester(pnl)
    var_comparison = backtester.compute_all_var_methods()

    print(var_comparison.to_string(index=False))

    # Check parametric vs MC alignment (guide: within 5%)
    for _, row in var_comparison.iterrows():
        status = "✅" if row['Within_5pct'] else "⚠️"
        print(f"\n  {status} {row['Confidence']}: Hist-Param diff = {row['Hist_Param_Diff_Pct']:.2f}%"
              f" {'(within 5%)' if row['Within_5pct'] else '(exceeds 5%)'}")

    # 3. Kupiec POF Test
    print_subsection("Kupiec Proportion of Failures (POF) Test")
    backtest_results = backtester.run_backtest()

    for _, row in backtest_results.iterrows():
        print(f"\n  {row['Confidence_Level']}:")
        print(f"    Violations: {row['N_Violations']}/{row['N_Observations']}")
        print(f"    Expected rate: {row['Expected_Rate']:.4f}, Observed: {row['Observed_Rate']:.4f}")
        print(f"    LR Statistic: {row['LR_Statistic']:.4f} (critical: {row['Critical_Value']:.4f})")
        print(f"    P-Value: {row['P_Value']:.4f}")
        print(f"    Result: {row['Result']}")

    # 4. Traffic Light Test
    print_subsection("Basel II Traffic Light Test")
    tl_result = backtester.traffic_light_test()
    print(f"  VaR 99%: {fmt_inr(tl_result['VaR_99'])}")
    print(f"  Violations: {tl_result['Violations']}")
    print(f"  Scaled (250d): {tl_result['Scaled_Violations_250d']:.1f}")
    print(f"  Zone: {tl_result['Zone']}")
    print(f"  {tl_result['Interpretation']}")

    # 5. Visualizations
    print_subsection("Generating VaR Backtest Visualizations")
    plot_var_comparison(var_comparison, FIGURES_DIR / "var_method_comparison.png")
    plot_kupiec_summary(backtest_results, FIGURES_DIR / "var_kupiec_summary.png")

    var_95 = -np.percentile(pnl, 5)
    plot_backtest_results(pnl, var_95, "95%", FIGURES_DIR / "var_backtest_95.png")

    # 6. Save reports
    var_comparison.to_csv(REPORTS_DIR / "var_method_comparison.csv", index=False)
    backtest_results.to_csv(REPORTS_DIR / "var_backtest_kupiec.csv", index=False)

    print_section_header("VAR BACKTEST COMPLETE ✅")
    return var_comparison, backtest_results


if __name__ == "__main__":
    run_var_backtest()
