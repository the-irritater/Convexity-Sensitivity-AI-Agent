"""
Part 2: Yield Curve Modelling & DV01 Sensitivity Calculations
==============================================================
Nelson-Siegel-Svensson yield curve fitting, cubic spline interpolation,
DV01 ladder, and PCA factor analysis of yield curve movements.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, load_yield_curve,
    fmt_pct, fmt_bps, fmt_inr,
    print_section_header, print_subsection,
    FIGURES_DIR, REPORTS_DIR
)


# ─────────────────────────────────────────────────────────────
# Nelson-Siegel-Svensson Model
# ─────────────────────────────────────────────────────────────

class NelsonSiegelSvensson:
    """
    Nelson-Siegel-Svensson yield curve model.

    y(t) = β0 + β1 × [(1-e^(-t/τ1))/(t/τ1)]
             + β2 × [(1-e^(-t/τ1))/(t/τ1) - e^(-t/τ1)]
             + β3 × [(1-e^(-t/τ2))/(t/τ2) - e^(-t/τ2)]

    Parameters: β0 (level), β1 (slope), β2 (curvature1), β3 (curvature2), τ1, τ2
    """

    def __init__(self):
        self.params = None
        self.fitted = False

    @staticmethod
    def _yield_nss(t, beta0, beta1, beta2, beta3, tau1, tau2):
        """Compute NSS yield for given tenors."""
        t = np.asarray(t, dtype=float)
        t = np.maximum(t, 1e-6) # Avoid division by zero

        x1 = t / tau1
        x2 = t / tau2

        factor1 = (1 - np.exp(-x1)) / x1
        factor2 = factor1 - np.exp(-x1)
        factor3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)

        return beta0 + beta1 * factor1 + beta2 * factor2 + beta3 * factor3

    def fit(self, tenors, yields):
        """
        Fit the NSS model to observed yield data.

        Parameters
        ----------
        tenors : array-like — Tenor points (years)
        yields : array-like — Observed yields (decimal)
        """
        tenors = np.asarray(tenors, dtype=float)
        yields = np.asarray(yields, dtype=float)

        def objective(params):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            if tau1 <= 0.01 or tau2 <= 0.01:
                return 1e10
            predicted = self._yield_nss(tenors, *params)
            return np.sum((predicted - yields) ** 2)

        # Initial guess
        x0 = [yields[-1], yields[0] - yields[-1], 0.0, 0.0, 1.5, 5.0]
        bounds = [
            (0, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2),
            (0.1, 30), (0.1, 30)
        ]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        self.params = result.x
        self.fitted = True
        self.fit_error = result.fun

        return self

    def predict(self, tenors):
        """Predict yields for given tenors using fitted model."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._yield_nss(tenors, *self.params)

    def get_factors(self):
        """Return interpretable factor names and values."""
        if not self.fitted:
            return {}
        names = ['Level (β0)', 'Slope (β1)', 'Curvature1 (β2)',
                 'Curvature2 (β3)', 'Decay1 (τ1)', 'Decay2 (τ2)']
        return dict(zip(names, self.params))


# ─────────────────────────────────────────────────────────────
# Nelson-Siegel (simplified 4-parameter version)
# ─────────────────────────────────────────────────────────────

class NelsonSiegel:
    """
    Standard Nelson-Siegel 4-parameter model.
    y(t) = β0 + β1 × [(1-e^(-t/τ))/(t/τ)]
             + β2 × [(1-e^(-t/τ))/(t/τ) - e^(-t/τ)]
    """

    def __init__(self):
        self.params = None
        self.fitted = False

    @staticmethod
    def _yield_ns(t, beta0, beta1, beta2, tau):
        t = np.asarray(t, dtype=float)
        t = np.maximum(t, 1e-6)
        x = t / tau
        factor1 = (1 - np.exp(-x)) / x
        factor2 = factor1 - np.exp(-x)
        return beta0 + beta1 * factor1 + beta2 * factor2

    def fit(self, tenors, yields):
        tenors = np.asarray(tenors, dtype=float)
        yields = np.asarray(yields, dtype=float)

        def objective(params):
            beta0, beta1, beta2, tau = params
            if tau <= 0.01:
                return 1e10
            predicted = self._yield_ns(tenors, *params)
            return np.sum((predicted - yields) ** 2)

        x0 = [yields[-1], yields[0] - yields[-1], 0.0, 2.0]
        bounds = [(0, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.1, 30)]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        self.params = result.x
        self.fitted = True
        return self

    def predict(self, tenors):
        if not self.fitted:
            raise ValueError("Model not fitted.")
        return self._yield_ns(tenors, *self.params)


# ─────────────────────────────────────────────────────────────
# Cubic Spline Interpolation
# ─────────────────────────────────────────────────────────────

class YieldCurveSpline:
    """Cubic spline interpolation for yield curve smoothing."""

    def __init__(self):
        self.spline = None

    def fit(self, tenors, yields):
        self.tenors = np.asarray(tenors, dtype=float)
        self.yields = np.asarray(yields, dtype=float)
        self.spline = CubicSpline(self.tenors, self.yields, bc_type='natural')
        return self

    def predict(self, tenors):
        return self.spline(tenors)

    def forward_rate(self, t, dt=0.01):
        """Instantaneous forward rate at time t."""
        return self.spline(t, 1) * t / (t + dt) + self.spline(t)


# ─────────────────────────────────────────────────────────────
# DV01 Sensitivity Calculations
# ─────────────────────────────────────────────────────────────

def compute_portfolio_dv01_ladder(bond_df):
    """
    Compute DV01 ladder by key rate bucket.
    Shows portfolio sensitivity to 1bp move at each tenor point.

    Returns
    -------
    pd.DataFrame — DV01 by tenor bucket
    """
    if 'KeyRateBucket' not in bond_df.columns:
        return pd.DataFrame()

    bond_df = bond_df.copy()
    bond_df['BondDV01'] = bond_df['ModifiedDuration'] * bond_df['MarketValue_INR'] * 0.0001

    ladder = bond_df.groupby('KeyRateBucket').agg(
        Count=('BondID', 'count'),
        Total_DV01=('BondDV01', 'sum'),
        Avg_Duration=('ModifiedDuration', 'mean'),
        Total_Market_Value=('MarketValue_INR', 'sum'),
    ).reset_index()

    # Sort by tenor
    bucket_order = ['0-2Y', '2-3Y', '3-5Y', '5-7Y', '7-10Y', '10-15Y', '15-20Y', '20Y+']
    ladder['SortKey'] = ladder['KeyRateBucket'].apply(
        lambda x: bucket_order.index(x) if x in bucket_order else 99
    )
    ladder = ladder.sort_values('SortKey').drop(columns='SortKey')
    ladder['DV01_Pct'] = ladder['Total_DV01'] / ladder['Total_DV01'].sum() * 100

    return ladder


def compute_key_rate_dv01(bond_df, tenors_to_shock=None):
    """
    Compute key rate DV01 using effective duration approximation.
    Shocks each key rate independently and measures portfolio impact.
    """
    if tenors_to_shock is None:
        tenors_to_shock = [2, 3, 5, 7, 10, 15, 20, 30]

    bond_df = bond_df.copy()
    total_mv = bond_df['MarketValue_INR'].sum()

    results = []
    for tenor in tenors_to_shock:
        # Approximate: bonds in this bucket respond fully, others don't
        bucket_map = {
            2: ['0-2Y'], 3: ['2-3Y'], 5: ['3-5Y', '5-7Y'],
            7: ['5-7Y', '7-10Y'], 10: ['7-10Y', '10-15Y'],
            15: ['10-15Y', '15-20Y'], 20: ['15-20Y', '20Y+'],
            30: ['20Y+']
        }

        buckets = bucket_map.get(tenor, [])
        mask = bond_df['KeyRateBucket'].isin(buckets) if 'KeyRateBucket' in bond_df.columns else pd.Series([False]*len(bond_df))

        subset = bond_df[mask]
        kr_dv01 = (subset['ModifiedDuration'] * subset['MarketValue_INR'] * 0.0001).sum()

        results.append({
            'Tenor': f'{tenor}Y',
            'KeyRate_DV01': kr_dv01,
            'Bond_Count': len(subset),
            'Market_Value': subset['MarketValue_INR'].sum(),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Yield Curve PCA Analysis
# ─────────────────────────────────────────────────────────────

def yield_curve_pca(yc_df, n_components=3):
    """
    Principal Component Analysis of yield curve movements.
    Extracts level, slope, and curvature factors.

    Parameters
    ----------
    yc_df : pd.DataFrame — Yield curve history
    n_components : int — Number of PCA components

    Returns
    -------
    dict — PCA results including components, explained variance, scores
    """
    # Pivot to get tenor columns
    pivot = yc_df.pivot_table(values='Yield', index='CurveDate', columns='Tenor_Years')
    pivot = pivot.dropna()

    if len(pivot) < 3:
        return None

    # Compute yield changes
    yield_changes = pivot.diff().dropna()

    # Standardize
    scaler = StandardScaler()
    scaled_changes = scaler.fit_transform(yield_changes)

    # PCA
    pca = PCA(n_components=min(n_components, scaled_changes.shape[1]))
    scores = pca.fit_transform(scaled_changes)

    component_names = ['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)'][:n_components]

    results = {
        'pca': pca,
        'scaler': scaler,
        'components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'scores': scores,
        'tenors': pivot.columns.tolist(),
        'dates': yield_changes.index.tolist(),
        'component_names': component_names,
        'yield_levels': pivot,
        'yield_changes': yield_changes,
    }

    return results


# ─────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────

def plot_yield_curves_history(yc_df, save_path=None):
    """Plot historical yield curves (selected dates)."""
    pivot = yc_df.pivot_table(values='Yield', index='CurveDate', columns='Tenor_Years')

    fig, ax = plt.subplots(figsize=(12, 7))

    # Select representative dates
    dates = pivot.index.unique()
    n_curves = min(8, len(dates))
    selected = dates[np.linspace(0, len(dates)-1, n_curves, dtype=int)]

    colors = plt.cm.viridis(np.linspace(0, 1, n_curves))
    for date, color in zip(selected, colors):
        row = pivot.loc[date].dropna()
        ax.plot(row.index, row.values * 100, 'o-', color=color,
                label=str(date)[:10], markersize=5, linewidth=2)

    ax.set_xlabel('Tenor (Years)', fontsize=12)
    ax.set_ylabel('Yield (%)', fontsize=12)
    ax.set_title('Historical INR Yield Curves', fontsize=14, fontweight='bold')
    ax.legend(title='Curve Date', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_nss_fit(tenors, actual_yields, model, model_name="NSS", save_path=None):
    """Plot actual vs fitted yield curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Fitted curve (smooth)
    smooth_tenors = np.linspace(min(tenors), max(tenors), 200)
    fitted = model.predict(smooth_tenors)

    ax.plot(smooth_tenors, fitted * 100, '-', color='#1976D2', linewidth=2.5, label=f'{model_name} Fitted')
    ax.scatter(tenors, actual_yields * 100, color='#F44336', s=80, zorder=5,
               edgecolors='white', linewidth=1.5, label='Actual')

    ax.set_xlabel('Tenor (Years)', fontsize=12)
    ax.set_ylabel('Yield (%)', fontsize=12)
    ax.set_title(f'Yield Curve Fit: {model_name} Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_dv01_ladder(ladder_df, save_path=None):
    """Plot DV01 ladder across tenor buckets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    bars = axes[0].bar(ladder_df['KeyRateBucket'], ladder_df['Total_DV01'],
                       color='#E64A19', alpha=0.8, edgecolor='white')
    axes[0].set_xlabel('Tenor Bucket')
    axes[0].set_ylabel('DV01 (INR)')
    axes[0].set_title('DV01 Ladder by Tenor Bucket', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Pie chart of DV01 distribution
    axes[1].pie(ladder_df['DV01_Pct'], labels=ladder_df['KeyRateBucket'],
                autopct='%1.1f%%', colors=plt.cm.Set3(np.linspace(0, 1, len(ladder_df))))
    axes[1].set_title('DV01 Distribution', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_pca_results(pca_results, save_path=None):
    """Plot PCA factor loadings and explained variance."""
    if pca_results is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tenors = pca_results['tenors']
    components = pca_results['components']
    var_ratio = pca_results['explained_variance_ratio']

    # Factor loadings
    colors = ['#1976D2', '#E64A19', '#4CAF50']
    for i, (comp, name) in enumerate(zip(components, pca_results['component_names'])):
        axes[0].plot(tenors, comp, 'o-', color=colors[i], label=f'{name} ({var_ratio[i]*100:.1f}%)',
                     linewidth=2, markersize=6)
    axes[0].set_xlabel('Tenor (Years)')
    axes[0].set_ylabel('Factor Loading')
    axes[0].set_title('PCA Factor Loadings', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linewidth=0.5)

    # Explained variance
    axes[1].bar(range(1, len(var_ratio)+1), var_ratio * 100, color='#7B1FA2', alpha=0.8)
    axes[1].plot(range(1, len(var_ratio)+1), pca_results['cumulative_variance'] * 100,
                 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance (%)')
    axes[1].set_title('Variance Explained by PC', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Factor scores time series
    scores = pca_results['scores']
    dates = pca_results['dates']
    for i, name in enumerate(pca_results['component_names']):
        axes[2].plot(dates, scores[:, i], color=colors[i], label=name, alpha=0.7, linewidth=1)
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Factor Score')
    axes[2].set_title('PCA Factor Scores Over Time', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_part2():
    """Execute Part 2: Yield Curve Modelling & DV01 Sensitivity."""
    print_section_header("PART 2: Yield Curve Modelling & DV01 Sensitivity Calculations")

    # 1. Load data
    print(" Loading data...")
    bond_df = load_bond_portfolio()
    yc_df = load_yield_curve()
    print(f" Bonds: {len(bond_df)}, Yield curve records: {len(yc_df)}")

    # 2. Fit Nelson-Siegel-Svensson to latest curve
    print_subsection("Nelson-Siegel-Svensson Model Fitting")

    latest_date = yc_df['CurveDate'].max()
    latest_curve = yc_df[yc_df['CurveDate'] == latest_date].sort_values('Tenor_Years')
    tenors = latest_curve['Tenor_Years'].values
    yields = latest_curve['Yield'].values

    print(f" Latest curve date: {latest_date}")
    print(f" Tenors: {tenors}")
    print(f" Yields: {[f'{y*100:.2f}%' for y in yields]}")

    # Fit NSS
    nss = NelsonSiegelSvensson()
    nss.fit(tenors, yields)
    factors = nss.get_factors()
    print("\n NSS Parameters:")
    for name, val in factors.items():
        print(f" {name:.<30} {val:.6f}")

    # Fit NS (simplified)
    ns = NelsonSiegel()
    ns.fit(tenors, yields)

    # Fit Cubic Spline
    spline = YieldCurveSpline()
    spline.fit(tenors, yields)

    # Compare fits
    print_subsection("Model Comparison (RMSE)")
    smooth_t = np.linspace(min(tenors), max(tenors), 100)
    for name, model in [('NSS', nss), ('NS', ns), ('Spline', spline)]:
        predicted = model.predict(tenors)
        rmse = np.sqrt(np.mean((predicted - yields) ** 2))
        print(f" {name:.<20} RMSE = {rmse:.8f}")

    # 3. DV01 Ladder
    print_subsection("DV01 Sensitivity Ladder")
    ladder = compute_portfolio_dv01_ladder(bond_df)
    if not ladder.empty:
        print(ladder.to_string(index=False))
        print(f"\n Total Portfolio DV01: {fmt_inr(ladder['Total_DV01'].sum())}")

    # 4. Key Rate DV01
    print_subsection("Key Rate DV01 Analysis")
    kr_dv01 = compute_key_rate_dv01(bond_df)
    print(kr_dv01.to_string(index=False))

    # 5. Yield Curve PCA
    print_subsection("Yield Curve PCA Analysis")
    pca_results = yield_curve_pca(yc_df)
    if pca_results:
        print(" Explained Variance Ratios:")
        for name, var in zip(pca_results['component_names'], pca_results['explained_variance_ratio']):
            print(f" {name:.<30} {var*100:.2f}%")
        print(f" {'Cumulative':.<30} {pca_results['cumulative_variance'][-1]*100:.2f}%")

    # 6. Visualizations
    print_subsection("Generating Visualizations")
    plot_yield_curves_history(yc_df, FIGURES_DIR / "p2_yield_curves_history.png")
    plot_nss_fit(tenors, yields, nss, "NSS", FIGURES_DIR / "p2_nss_fit.png")
    plot_nss_fit(tenors, yields, ns, "Nelson-Siegel", FIGURES_DIR / "p2_ns_fit.png")
    plot_nss_fit(tenors, yields, spline, "Cubic Spline", FIGURES_DIR / "p2_spline_fit.png")
    if not ladder.empty:
        plot_dv01_ladder(ladder, FIGURES_DIR / "p2_dv01_ladder.png")
    if pca_results:
        plot_pca_results(pca_results, FIGURES_DIR / "p2_pca_analysis.png")

    # 7. Save reports
    if not ladder.empty:
        ladder.to_csv(REPORTS_DIR / "part2_dv01_ladder.csv", index=False)
    kr_dv01.to_csv(REPORTS_DIR / "part2_key_rate_dv01.csv", index=False)

    print_section_header("PART 2 COMPLETE ")
    return nss, ladder, pca_results


if __name__ == "__main__":
    run_part2()
