"""
Part 1: Bond Portfolio Duration & Convexity Analytics Framework
================================================================
Computes duration, convexity, DV01 from first principles.
Provides portfolio-level aggregation, sector analysis, and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, generate_bond_cashflows, discount_factor,
    present_value_cashflows, fmt_pct, fmt_bps, fmt_inr,
    print_section_header, print_subsection, FIGURES_DIR, REPORTS_DIR
)


# ─────────────────────────────────────────────────────────────
# Bond-Level Analytics
# ─────────────────────────────────────────────────────────────

class BondAnalytics:
    """
    Compute bond risk metrics from first principles.

    Calculates Macaulay duration, modified duration, convexity,
    DV01, and price sensitivity for individual bonds.
    """

    def __init__(self, face_value, coupon_rate, coupon_freq, years_to_maturity, ytm):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon_freq = coupon_freq
        self.years_to_maturity = years_to_maturity
        self.ytm = ytm

        # Generate cashflows
        self.cashflows, self.times = generate_bond_cashflows(
            face_value, coupon_rate, coupon_freq, years_to_maturity
        )

        # Periodic yield
        self.periodic_yield = ytm / coupon_freq if coupon_freq > 0 else ytm

    def dirty_price(self, yield_override=None):
        """Calculate the dirty (full) price of the bond."""
        y = yield_override if yield_override is not None else self.ytm
        return present_value_cashflows(self.cashflows, self.times, y)

    def macaulay_duration(self):
        """
        Macaulay Duration — weighted average time to cashflows.
        D_mac = Σ [t_i × CF_i × DF(t_i)] / Price
        """
        price = self.dirty_price()
        if price <= 0:
            return 0.0

        weighted_sum = 0.0
        for cf, t in zip(self.cashflows, self.times):
            pv_cf = cf * discount_factor(self.ytm, t)
            weighted_sum += t * pv_cf

        return weighted_sum / price

    def modified_duration(self):
        """
        Modified Duration — price sensitivity to yield changes.
        D_mod = D_mac / (1 + y/m)
        """
        d_mac = self.macaulay_duration()
        return d_mac / (1 + self.periodic_yield)

    def convexity(self):
        """
        Convexity — second-order price sensitivity.
        C = [1/P] × Σ [t_i × (t_i + 1/m) × CF_i × DF(t_i)] / (1 + y/m)^2
        """
        price = self.dirty_price()
        if price <= 0:
            return 0.0

        m = self.coupon_freq if self.coupon_freq > 0 else 1
        conv_sum = 0.0
        for cf, t in zip(self.cashflows, self.times):
            pv_cf = cf * discount_factor(self.ytm, t)
            conv_sum += t * (t + 1.0 / m) * pv_cf

        return conv_sum / (price * (1 + self.periodic_yield) ** 2)

    def dv01(self):
        """
        DV01 (Dollar Value of 01) — price change for 1 basis point yield shift.
        DV01 = Modified Duration × Price × 0.0001
        """
        return self.modified_duration() * self.dirty_price() * 0.0001

    def dv01_per_100_face(self):
        """DV01 per 100 face value."""
        return self.modified_duration() * (self.dirty_price() / self.face_value * 100) * 0.0001

    def effective_duration(self, shock_bps=1):
        """
        Effective Duration — numerical approximation.
        D_eff = (P_down - P_up) / (2 × P_0 × Δy)
        """
        dy = shock_bps / 10000.0
        p0 = self.dirty_price()
        p_up = self.dirty_price(self.ytm + dy)
        p_down = self.dirty_price(self.ytm - dy)

        if p0 <= 0:
            return 0.0
        return (p_down - p_up) / (2 * p0 * dy)

    def effective_convexity(self, shock_bps=1):
        """
        Effective Convexity — numerical approximation.
        C_eff = (P_up + P_down - 2×P_0) / (P_0 × Δy²)
        """
        dy = shock_bps / 10000.0
        p0 = self.dirty_price()
        p_up = self.dirty_price(self.ytm + dy)
        p_down = self.dirty_price(self.ytm - dy)

        if p0 <= 0:
            return 0.0
        return (p_up + p_down - 2 * p0) / (p0 * dy ** 2)

    def price_change_duration_convexity(self, delta_y_bps):
        """
        Approximate price change using duration-convexity approximation.
        ΔP/P ≈ -D_mod × Δy + 0.5 × C × Δy²

        Parameters
        ----------
        delta_y_bps : float — Yield change in basis points

        Returns
        -------
        tuple — (pct_change, dollar_change)
        """
        dy = delta_y_bps / 10000.0
        d_mod = self.modified_duration()
        conv = self.convexity()
        price = self.dirty_price()

        pct_change = -d_mod * dy + 0.5 * conv * dy ** 2
        dollar_change = pct_change * price

        return pct_change, dollar_change

    def full_metrics(self):
        """Return all bond metrics as a dictionary."""
        return {
            'DirtyPrice': self.dirty_price(),
            'MacaulayDuration': self.macaulay_duration(),
            'ModifiedDuration': self.modified_duration(),
            'Convexity': self.convexity(),
            'DV01': self.dv01(),
            'DV01_Per100Face': self.dv01_per_100_face(),
            'EffectiveDuration': self.effective_duration(),
            'EffectiveConvexity': self.effective_convexity(),
        }


# ─────────────────────────────────────────────────────────────
# Portfolio-Level Analytics
# ─────────────────────────────────────────────────────────────

class PortfolioAnalytics:
    """
    Portfolio-level duration, convexity, and risk analytics.
    Aggregates bond-level metrics using market-value weighting.
    """

    def __init__(self, bond_df):
        """
        Parameters
        ----------
        bond_df : pd.DataFrame — Bond portfolio data with pre-computed metrics
        """
        self.df = bond_df.copy()
        self._compute_weights()

    def _compute_weights(self):
        """Compute market-value weights for each bond."""
        self.df['MarketValue'] = self.df['MarketValue_INR'].astype(float)
        total_mv = self.df['MarketValue'].sum()
        self.df['Weight'] = self.df['MarketValue'] / total_mv
        self.total_market_value = total_mv

    def portfolio_duration(self):
        """Weighted average modified duration."""
        return (self.df['Weight'] * self.df['ModifiedDuration']).sum()

    def portfolio_macaulay_duration(self):
        """Weighted average Macaulay duration."""
        return (self.df['Weight'] * self.df['MacaulayDuration']).sum()

    def portfolio_convexity(self):
        """Weighted average convexity."""
        return (self.df['Weight'] * self.df['Convexity']).sum()

    def portfolio_dv01(self):
        """Total portfolio DV01 in INR."""
        return (self.df['ModifiedDuration'] * self.df['MarketValue'] * 0.0001).sum()

    def portfolio_ytm(self):
        """Weighted average yield to maturity."""
        return (self.df['Weight'] * self.df['YieldToMaturity']).sum()

    def duration_contribution_by(self, group_col):
        """
        Duration contribution breakdown by a grouping column.

        Returns DataFrame with duration contribution = weight × duration for each group.
        """
        grouped = self.df.groupby(group_col).apply(
            lambda g: pd.Series({
                'Count': len(g),
                'MarketValue': g['MarketValue'].sum(),
                'Weight': g['Weight'].sum(),
                'Avg_Duration': (g['Weight'] * g['ModifiedDuration']).sum() / g['Weight'].sum() if g['Weight'].sum() > 0 else 0,
                'Duration_Contribution': (g['Weight'] * g['ModifiedDuration']).sum(),
                'Avg_Convexity': (g['Weight'] * g['Convexity']).sum() / g['Weight'].sum() if g['Weight'].sum() > 0 else 0,
                'Convexity_Contribution': (g['Weight'] * g['Convexity']).sum(),
                'DV01_Contribution': (g['ModifiedDuration'] * g['MarketValue'] * 0.0001).sum(),
            }),
            include_groups=False
        ).reset_index()

        return grouped.sort_values('Duration_Contribution', ascending=False)

    def key_rate_duration_profile(self):
        """
        Key rate duration profile across tenor buckets.
        Groups bonds by KeyRateBucket and computes contribution.
        """
        if 'KeyRateBucket' not in self.df.columns:
            return pd.DataFrame()

        krd = self.df.groupby('KeyRateBucket').apply(
            lambda g: pd.Series({
                'Count': len(g),
                'Weight': g['Weight'].sum(),
                'KRD_Contribution': (g['Weight'] * g['ModifiedDuration']).sum(),
                'DV01': (g['ModifiedDuration'] * g['MarketValue'] * 0.0001).sum(),
            }),
            include_groups=False
        ).reset_index()

        # Sort by tenor bucket logically
        bucket_order = ['0-2Y', '2-3Y', '3-5Y', '5-7Y', '7-10Y', '10-15Y', '15-20Y', '20Y+']
        krd['SortKey'] = krd['KeyRateBucket'].apply(
            lambda x: bucket_order.index(x) if x in bucket_order else 99
        )
        return krd.sort_values('SortKey').drop(columns='SortKey')

    def price_sensitivity_analysis(self, shock_bps_list=None):
        """
        Portfolio price sensitivity across multiple yield shocks.
        Uses duration-convexity approximation.

        Parameters
        ----------
        shock_bps_list : list — Yield shocks in basis points

        Returns
        -------
        pd.DataFrame — Shock scenarios with P&L estimates
        """
        if shock_bps_list is None:
            shock_bps_list = [-200, -100, -50, -25, 25, 50, 100, 200]

        port_dur = self.portfolio_duration()
        port_conv = self.portfolio_convexity()
        total_mv = self.total_market_value

        results = []
        for shock in shock_bps_list:
            dy = shock / 10000.0
            pct_change = -port_dur * dy + 0.5 * port_conv * dy ** 2
            duration_effect = -port_dur * dy * total_mv
            convexity_effect = 0.5 * port_conv * dy ** 2 * total_mv
            total_pnl = pct_change * total_mv

            results.append({
                'Shock_bps': shock,
                'Delta_Yield': dy,
                'Pct_Change': pct_change,
                'Duration_Effect_INR': duration_effect,
                'Convexity_Effect_INR': convexity_effect,
                'Total_PnL_INR': total_pnl,
            })

        return pd.DataFrame(results)

    def summary(self):
        """Return a comprehensive portfolio summary."""
        return {
            'Total_Market_Value_INR': self.total_market_value,
            'Number_of_Bonds': len(self.df),
            'Portfolio_Modified_Duration': self.portfolio_duration(),
            'Portfolio_Macaulay_Duration': self.portfolio_macaulay_duration(),
            'Portfolio_Convexity': self.portfolio_convexity(),
            'Portfolio_DV01_INR': self.portfolio_dv01(),
            'Portfolio_YTM': self.portfolio_ytm(),
            'Avg_Coupon': self.df['CouponRate'].mean(),
            'Avg_Years_to_Maturity': self.df['YearsToMaturity'].mean(),
        }


# ─────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────

def plot_duration_convexity_scatter(df, save_path=None):
    """Scatter plot of Modified Duration vs Convexity, colored by sector."""
    fig, ax = plt.subplots(figsize=(12, 8))

    sectors = df['Sector'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(sectors)))

    for sector, color in zip(sectors, colors):
        mask = df['Sector'] == sector
        ax.scatter(
            df.loc[mask, 'ModifiedDuration'],
            df.loc[mask, 'Convexity'],
            c=[color], label=sector, alpha=0.7, s=50, edgecolors='white', linewidth=0.5
        )

    ax.set_xlabel('Modified Duration (years)', fontsize=12)
    ax.set_ylabel('Convexity', fontsize=12)
    ax.set_title('Bond Portfolio: Duration vs Convexity by Sector', fontsize=14, fontweight='bold')
    ax.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_duration_distribution(df, save_path=None):
    """Distribution of modified duration across the portfolio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    axes[0].hist(df['ModifiedDuration'], bins=30, color='#2196F3', alpha=0.7, edgecolor='white')
    axes[0].axvline(df['ModifiedDuration'].mean(), color='red', linestyle='--', label=f"Mean: {df['ModifiedDuration'].mean():.2f}")
    axes[0].axvline(df['ModifiedDuration'].median(), color='orange', linestyle='--', label=f"Median: {df['ModifiedDuration'].median():.2f}")
    axes[0].set_xlabel('Modified Duration (years)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Modified Duration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot by sector
    sector_data = [group['ModifiedDuration'].values for _, group in df.groupby('Sector')]
    sector_labels = [name for name, _ in df.groupby('Sector')]
    bp = axes[1].boxplot(sector_data, labels=sector_labels, patch_artist=True)
    colors_bp = plt.cm.Set2(np.linspace(0, 1, len(sector_labels)))
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_xlabel('Sector')
    axes[1].set_ylabel('Modified Duration (years)')
    axes[1].set_title('Duration Distribution by Sector')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_sector_contribution(portfolio, save_path=None):
    """Bar chart of duration and DV01 contribution by sector."""
    contrib = portfolio.duration_contribution_by('Sector')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Duration contribution
    bars1 = axes[0].barh(contrib['Sector'], contrib['Duration_Contribution'], color='#1976D2', alpha=0.8)
    axes[0].set_xlabel('Duration Contribution (years)')
    axes[0].set_title('Duration Contribution by Sector', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # DV01 contribution
    bars2 = axes[1].barh(contrib['Sector'], contrib['DV01_Contribution'], color='#E64A19', alpha=0.8)
    axes[1].set_xlabel('DV01 Contribution (INR)')
    axes[1].set_title('DV01 Contribution by Sector', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_sensitivity_profile(portfolio, save_path=None):
    """Price sensitivity profile across yield shocks."""
    sens = portfolio.price_sensitivity_analysis()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # P&L profile
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in sens['Total_PnL_INR']]
    axes[0].bar(sens['Shock_bps'].astype(str), sens['Total_PnL_INR'] / 1e5, color=colors, alpha=0.8)
    axes[0].set_xlabel('Yield Shock (bps)')
    axes[0].set_ylabel('P&L (₹ Lakhs)')
    axes[0].set_title('Portfolio P&L Sensitivity', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='black', linewidth=0.5)

    # Duration vs Convexity effect stacked
    axes[1].bar(sens['Shock_bps'].astype(str), sens['Duration_Effect_INR'] / 1e5,
                label='Duration Effect', color='#2196F3', alpha=0.8)
    axes[1].bar(sens['Shock_bps'].astype(str), sens['Convexity_Effect_INR'] / 1e5,
                bottom=sens['Duration_Effect_INR'] / 1e5,
                label='Convexity Effect', color='#FF9800', alpha=0.8)
    axes[1].set_xlabel('Yield Shock (bps)')
    axes[1].set_ylabel('Effect (₹ Lakhs)')
    axes[1].set_title('Duration vs Convexity Effect Decomposition', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_key_rate_duration(portfolio, save_path=None):
    """Key rate duration profile across tenor buckets."""
    krd = portfolio.key_rate_duration_profile()
    if krd.empty:
        print(" ️ No KeyRateBucket data available")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(krd['KeyRateBucket'], krd['KRD_Contribution'],
                  color='#7B1FA2', alpha=0.8, edgecolor='white')

    # Add value labels
    for bar, val in zip(bars, krd['KRD_Contribution']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Tenor Bucket')
    ax.set_ylabel('Key Rate Duration Contribution')
    ax.set_title('Key Rate Duration Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_convexity_heatmap(df, save_path=None):
    """Heatmap of average convexity by Sector × Credit Rating."""
    pivot = df.pivot_table(values='Convexity', index='Sector',
                           columns='CreditRating', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, linecolor='white')
    ax.set_title('Average Convexity: Sector × Credit Rating', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Verification: Compare computed vs CSV values
# ─────────────────────────────────────────────────────────────

def verify_calculations(df, n_samples=10):
    """
    Verify computed analytics against CSV-provided values.
    Tests on a sample of bonds for accuracy.
    """
    print_subsection("Verification: Computed vs CSV Values")

    errors = []
    np.random.seed(42)
    sample_idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)

    for idx in sample_idx:
        row = df.iloc[idx]
        bond = BondAnalytics(
            face_value=row['FaceValue'],
            coupon_rate=row['CouponRate'],
            coupon_freq=row['CouponFrequency'],
            years_to_maturity=row['YearsToMaturity'],
            ytm=row['YieldToMaturity']
        )

        computed = bond.full_metrics()
        csv_dur = row['ModifiedDuration']
        csv_conv = row['Convexity']

        dur_err = abs(computed['ModifiedDuration'] - csv_dur)
        conv_err = abs(computed['Convexity'] - csv_conv)

        errors.append({
            'BondID': row['BondID'],
            'Computed_Duration': computed['ModifiedDuration'],
            'CSV_Duration': csv_dur,
            'Duration_Error': dur_err,
            'Computed_Convexity': computed['Convexity'],
            'CSV_Convexity': csv_conv,
            'Convexity_Error': conv_err,
        })

    err_df = pd.DataFrame(errors)
    print(err_df.to_string(index=False))
    print(f"\n Mean Duration Error: {err_df['Duration_Error'].mean():.4f}")
    print(f" Mean Convexity Error: {err_df['Convexity_Error'].mean():.4f}")

    return err_df


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_part1():
    """Execute Part 1: Bond Portfolio Duration & Convexity Analytics."""
    print_section_header("PART 1: Bond Portfolio Duration & Convexity Analytics Framework")

    # 1. Load data
    print(" Loading bond portfolio data...")
    df = load_bond_portfolio()
    print(f" Loaded {len(df)} bonds with {len(df.columns)} columns")

    # 2. Portfolio Analytics
    print_subsection("Portfolio-Level Summary")
    portfolio = PortfolioAnalytics(df)
    summary = portfolio.summary()

    for key, val in summary.items():
        if 'INR' in key:
            print(f" {key:.<40} {fmt_inr(val)}")
        elif 'YTM' in key or 'Coupon' in key:
            print(f" {key:.<40} {fmt_pct(val)}")
        else:
            print(f" {key:.<40} {val:.4f}" if isinstance(val, float) else f" {key:.<40} {val}")

    # 3. Sector Analysis
    print_subsection("Duration Contribution by Sector")
    sector_contrib = portfolio.duration_contribution_by('Sector')
    print(sector_contrib[['Sector', 'Count', 'Weight', 'Avg_Duration',
                          'Duration_Contribution', 'DV01_Contribution']].to_string(index=False))

    # 4. Credit Rating Analysis
    print_subsection("Duration Contribution by Credit Rating")
    rating_contrib = portfolio.duration_contribution_by('CreditRating')
    print(rating_contrib[['CreditRating', 'Count', 'Weight', 'Avg_Duration',
                          'Duration_Contribution']].to_string(index=False))

    # 5. Key Rate Duration Profile
    print_subsection("Key Rate Duration Profile")
    krd = portfolio.key_rate_duration_profile()
    if not krd.empty:
        print(krd.to_string(index=False))

    # 6. Sensitivity Analysis
    print_subsection("Price Sensitivity Analysis")
    sens = portfolio.price_sensitivity_analysis()
    sens_display = sens.copy()
    sens_display['Pct_Change'] = sens_display['Pct_Change'].apply(lambda x: f"{x*100:.4f}%")
    sens_display['Total_PnL_INR'] = sens_display['Total_PnL_INR'].apply(fmt_inr)
    print(sens_display[['Shock_bps', 'Pct_Change', 'Total_PnL_INR']].to_string(index=False))

    # 7. Verify computations
    verify_calculations(df)

    # 8. Generate visualizations
    print_subsection("Generating Visualizations")
    plot_duration_convexity_scatter(df, FIGURES_DIR / "p1_duration_vs_convexity.png")
    plot_duration_distribution(df, FIGURES_DIR / "p1_duration_distribution.png")
    plot_sector_contribution(portfolio, FIGURES_DIR / "p1_sector_contribution.png")
    plot_sensitivity_profile(portfolio, FIGURES_DIR / "p1_sensitivity_profile.png")
    plot_key_rate_duration(portfolio, FIGURES_DIR / "p1_key_rate_duration.png")
    plot_convexity_heatmap(df, FIGURES_DIR / "p1_convexity_heatmap.png")

    # 9. Save reports
    report_path = REPORTS_DIR / "part1_portfolio_summary.csv"
    pd.DataFrame([summary]).to_csv(report_path, index=False)
    sector_contrib.to_csv(REPORTS_DIR / "part1_sector_analysis.csv", index=False)
    sens.to_csv(REPORTS_DIR / "part1_sensitivity_analysis.csv", index=False)
    print(f"\n Reports saved to {REPORTS_DIR}")

    print_section_header("PART 1 COMPLETE ")
    return portfolio, df


if __name__ == "__main__":
    run_part1()
