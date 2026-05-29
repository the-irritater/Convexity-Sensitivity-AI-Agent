"""
Part 3: Monte Carlo Simulation for Interest Rate Scenario Analysis
===================================================================
Vasicek and CIR short-rate models, correlated multi-factor simulation,
VaR/CVaR calculations, and stress testing.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, load_monte_carlo, load_yield_curve,
    fmt_pct, fmt_inr, print_section_header, print_subsection,
    FIGURES_DIR, REPORTS_DIR
)


# ─────────────────────────────────────────────────────────────
# Short-Rate Models
# ─────────────────────────────────────────────────────────────

class VasicekModel:
    """
    Vasicek Interest Rate Model.
    dr = κ(θ - r)dt + σ dW
    
    Parameters
    ----------
    kappa : float — Speed of mean reversion
    theta : float — Long-run mean level
    sigma : float — Volatility
    r0 : float — Initial short rate
    """
    
    def __init__(self, kappa=0.5, theta=0.065, sigma=0.01, r0=0.065):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0
    
    def calibrate_from_data(self, rate_series, dt=1/252):
        """
        Calibrate parameters from historical rate data using OLS.
        dr = κ(θ - r)dt + σdW → r(t+1) - r(t) = a + b*r(t) + ε
        """
        rates = np.asarray(rate_series, dtype=float)
        dr = np.diff(rates)
        r_lag = rates[:-1]
        
        # OLS: dr = a + b*r + ε
        X = np.column_stack([np.ones_like(r_lag), r_lag])
        beta = np.linalg.lstsq(X, dr, rcond=None)[0]
        
        a, b = beta
        residuals = dr - X @ beta
        
        self.kappa = -b / dt
        self.theta = -a / (b) if abs(b) > 1e-10 else self.theta
        self.sigma = np.std(residuals) / np.sqrt(dt)
        self.r0 = rates[-1]
        
        return self
    
    def simulate(self, T=1.0, n_steps=252, n_paths=1000, seed=42):
        """
        Simulate interest rate paths using Euler-Maruyama.
        
        Parameters
        ----------
        T : float — Time horizon (years)
        n_steps : int — Number of time steps
        n_paths : int — Number of simulation paths
        
        Returns
        -------
        tuple — (times, paths) where paths is (n_paths, n_steps+1)
        """
        np.random.seed(seed)
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0
        
        for i in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            dr = self.kappa * (self.theta - paths[:, i]) * dt + self.sigma * dW
            paths[:, i + 1] = paths[:, i] + dr
        
        return times, paths
    
    def bond_price_analytical(self, T, r=None):
        """Analytical zero-coupon bond price under Vasicek."""
        if r is None:
            r = self.r0
        B = (1 - np.exp(-self.kappa * T)) / self.kappa
        A = np.exp(
            (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B - T)
            - self.sigma**2 * B**2 / (4 * self.kappa)
        )
        return A * np.exp(-B * r)


class CIRModel:
    """
    Cox-Ingersoll-Ross Interest Rate Model.
    dr = κ(θ - r)dt + σ√r dW
    
    Ensures non-negative rates (Feller condition: 2κθ > σ²).
    """
    
    def __init__(self, kappa=0.5, theta=0.065, sigma=0.05, r0=0.065):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0
    
    def feller_condition(self):
        """Check if 2κθ > σ² (ensures rates stay positive)."""
        return 2 * self.kappa * self.theta > self.sigma ** 2
    
    def simulate(self, T=1.0, n_steps=252, n_paths=1000, seed=42):
        """Simulate CIR rate paths using Euler-Maruyama with reflection."""
        np.random.seed(seed)
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0
        
        for i in range(n_steps):
            r_current = np.maximum(paths[:, i], 1e-8)
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            dr = self.kappa * (self.theta - r_current) * dt + self.sigma * np.sqrt(r_current) * dW
            paths[:, i + 1] = np.maximum(paths[:, i] + dr, 0)
        
        return times, paths


# ─────────────────────────────────────────────────────────────
# Multi-Factor Simulation
# ─────────────────────────────────────────────────────────────

class MultiFactorSimulation:
    """
    Correlated multi-factor yield curve simulation.
    Simulates parallel shift, twist, and butterfly factors.
    """
    
    def __init__(self, mc_df=None):
        """
        Parameters
        ----------
        mc_df : pd.DataFrame — Monte Carlo scenarios with factor columns
        """
        self.mc_df = mc_df
        if mc_df is not None:
            self._estimate_parameters()
    
    def _estimate_parameters(self):
        """Estimate factor distribution parameters from data."""
        factors = ['ParallelShift_bps', 'TwistFactor_bps', 'ButterflyFactor_bps']
        self.factor_means = self.mc_df[factors].mean().values
        self.factor_stds = self.mc_df[factors].std().values
        self.factor_corr = self.mc_df[factors].corr().values
        self.factor_cov = self.mc_df[factors].cov().values
    
    def simulate_factors(self, n_scenarios=1000, seed=42):
        """Generate correlated factor scenarios using Cholesky decomposition."""
        np.random.seed(seed)
        L = np.linalg.cholesky(self.factor_cov)
        z = np.random.normal(0, 1, (n_scenarios, 3))
        scenarios = z @ L.T + self.factor_means
        
        return pd.DataFrame(scenarios,
                           columns=['ParallelShift_bps', 'TwistFactor_bps', 'ButterflyFactor_bps'])
    
    def compute_portfolio_pnl(self, scenarios_df, port_duration, port_convexity, total_mv):
        """
        Compute P&L for each scenario using duration-convexity approximation.
        
        Focuses on parallel shift as primary driver.
        """
        results = scenarios_df.copy()
        dy = results['ParallelShift_bps'] / 10000.0
        
        results['Duration_Effect'] = -port_duration * dy * total_mv
        results['Convexity_Effect'] = 0.5 * port_convexity * (dy ** 2) * total_mv
        results['Total_PnL'] = results['Duration_Effect'] + results['Convexity_Effect']
        results['PnL_Pct'] = results['Total_PnL'] / total_mv * 100
        
        return results


# ─────────────────────────────────────────────────────────────
# VaR & CVaR Calculations
# ─────────────────────────────────────────────────────────────

def compute_var_cvar(pnl_series, confidence_levels=None):
    """
    Compute Value at Risk and Conditional VaR (Expected Shortfall).
    
    Parameters
    ----------
    pnl_series : array-like — P&L values (positive = gain)
    confidence_levels : list — Confidence levels (e.g., [0.90, 0.95, 0.99])
    
    Returns
    -------
    pd.DataFrame — VaR and CVaR at each confidence level
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]
    
    pnl = np.asarray(pnl_series, dtype=float)
    
    results = []
    for cl in confidence_levels:
        alpha = 1 - cl
        var = -np.percentile(pnl, alpha * 100)
        # CVaR = expected loss given loss exceeds VaR
        tail_losses = pnl[pnl <= -var]
        cvar = -tail_losses.mean() if len(tail_losses) > 0 else var
        
        results.append({
            'Confidence': f"{cl*100:.0f}%",
            'VaR_INR': var,
            'CVaR_INR': cvar,
            'VaR_Pct': var / abs(pnl.mean()) * 100 if abs(pnl.mean()) > 0 else 0,
        })
    
    return pd.DataFrame(results)


def historical_var(mc_df, confidence=0.95):
    """Compute historical VaR from Monte Carlo scenario data."""
    pnl = mc_df['PnL_Total_INR'].values
    alpha = 1 - confidence
    var = -np.percentile(pnl, alpha * 100)
    return var


def parametric_var(pnl_series, confidence=0.95):
    """Compute parametric (Gaussian) VaR."""
    mu = np.mean(pnl_series)
    sigma = np.std(pnl_series)
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma)


# ─────────────────────────────────────────────────────────────
# Stress Testing
# ─────────────────────────────────────────────────────────────

def run_stress_tests(port_duration, port_convexity, total_mv):
    """
    Run predefined stress scenarios modelled after historical events.
    
    Returns
    -------
    pd.DataFrame — Stress test results
    """
    scenarios = [
        {"Name": "Parallel +100 bps", "Shift_bps": 100, "Description": "Uniform rate increase"},
        {"Name": "Parallel -100 bps", "Shift_bps": -100, "Description": "Uniform rate decrease"},
        {"Name": "Parallel +200 bps", "Shift_bps": 200, "Description": "Aggressive tightening"},
        {"Name": "Parallel +300 bps", "Shift_bps": 300, "Description": "Extreme rate shock"},
        {"Name": "2013 Taper Tantrum", "Shift_bps": 150, "Description": "Sudden EM rate spike"},
        {"Name": "2020 COVID Crash", "Shift_bps": -200, "Description": "Flight-to-quality rally"},
        {"Name": "2022 Rate Hike Cycle", "Shift_bps": 250, "Description": "Aggressive Fed/RBI tightening"},
        {"Name": "Stagflation Scenario", "Shift_bps": 350, "Description": "Persistent high rates"},
        {"Name": "Mild Easing", "Shift_bps": -50, "Description": "Gradual rate cuts"},
        {"Name": "Curve Steepener", "Shift_bps": 75, "Description": "Long-end sells off"},
    ]
    
    results = []
    for scenario in scenarios:
        dy = scenario['Shift_bps'] / 10000.0
        duration_effect = -port_duration * dy * total_mv
        convexity_effect = 0.5 * port_convexity * (dy ** 2) * total_mv
        total_pnl = duration_effect + convexity_effect
        pnl_pct = total_pnl / total_mv * 100
        
        results.append({
            'Scenario': scenario['Name'],
            'Shift_bps': scenario['Shift_bps'],
            'Description': scenario['Description'],
            'Duration_Effect_INR': duration_effect,
            'Convexity_Effect_INR': convexity_effect,
            'Total_PnL_INR': total_pnl,
            'PnL_Pct': pnl_pct,
        })
    
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────

def plot_rate_paths(times, paths, model_name="Vasicek", save_path=None):
    """Plot simulated interest rate paths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sample paths
    n_show = min(50, paths.shape[0])
    for i in range(n_show):
        axes[0].plot(times, paths[i, :] * 100, alpha=0.15, linewidth=0.5, color='#1976D2')
    
    # Mean and percentiles
    mean_path = np.mean(paths, axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    
    axes[0].plot(times, mean_path * 100, 'r-', linewidth=2.5, label='Mean')
    axes[0].fill_between(times, p5 * 100, p95 * 100, alpha=0.2, color='red', label='5th-95th %ile')
    axes[0].set_xlabel('Time (Years)')
    axes[0].set_ylabel('Rate (%)')
    axes[0].set_title(f'{model_name} Model: Simulated Rate Paths', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Terminal rate distribution
    terminal_rates = paths[:, -1] * 100
    axes[1].hist(terminal_rates, bins=40, density=True, alpha=0.7,
                 color='#7B1FA2', edgecolor='white')
    axes[1].axvline(np.mean(terminal_rates), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(terminal_rates):.2f}%')
    axes[1].set_xlabel('Terminal Rate (%)')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'{model_name}: Terminal Rate Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_pnl_distribution(pnl_series, var_results=None, save_path=None):
    """Plot P&L distribution with VaR and CVaR markers."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    pnl_lakhs = np.asarray(pnl_series) / 1e5
    
    ax.hist(pnl_lakhs, bins=50, density=True, alpha=0.7, color='#2196F3',
            edgecolor='white', label='P&L Distribution')
    
    # Add VaR lines
    if var_results is not None:
        colors = ['#FF9800', '#F44336', '#9C27B0']
        for i, row in var_results.iterrows():
            var_val = row['VaR_INR'] / 1e5
            ax.axvline(-var_val, color=colors[i % len(colors)], linestyle='--',
                       linewidth=2, label=f"VaR {row['Confidence']}: ₹{var_val:.1f}L")
    
    ax.set_xlabel('P&L (₹ Lakhs)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Portfolio P&L Distribution with VaR', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_stress_test_waterfall(stress_df, save_path=None):
    """Waterfall chart of stress test P&L impacts."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in stress_df['Total_PnL_INR']]
    bars = ax.barh(stress_df['Scenario'], stress_df['Total_PnL_INR'] / 1e5,
                   color=colors, alpha=0.8, edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, stress_df['Total_PnL_INR'] / 1e5):
        x_pos = bar.get_width() + (2 if val >= 0 else -2)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'₹{val:+.1f}L', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('P&L Impact (₹ Lakhs)', fontsize=12)
    ax.set_title('Stress Test Scenarios: Portfolio Impact', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_scenario_heatmap(mc_df, save_path=None):
    """Heatmap of P&L across factor dimensions."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bins for parallel shift and twist
    mc_df = mc_df.copy()
    mc_df['Shift_Bin'] = pd.cut(mc_df['ParallelShift_bps'], bins=8)
    mc_df['Twist_Bin'] = pd.cut(mc_df['TwistFactor_bps'], bins=8)
    
    pivot = mc_df.pivot_table(values='PnL_Total_INR', index='Twist_Bin',
                               columns='Shift_Bin', aggfunc='mean')
    
    sns.heatmap(pivot / 1e5, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax,
                linewidths=0.5, center=0)
    ax.set_xlabel('Parallel Shift (bps)')
    ax.set_ylabel('Twist Factor (bps)')
    ax.set_title('Average P&L (₹L): Shift × Twist', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_part3():
    """Execute Part 3: Monte Carlo Simulation for Interest Rate Scenario Analysis."""
    print_section_header("PART 3: Monte Carlo Simulation for Interest Rate Scenario Analysis")
    
    # 1. Load data
    print("  📂 Loading data...")
    bond_df = load_bond_portfolio()
    mc_df = load_monte_carlo()
    yc_df = load_yield_curve()
    
    # Portfolio-level metrics
    total_mv = bond_df['MarketValue_INR'].sum()
    weights = bond_df['MarketValue_INR'] / total_mv
    port_duration = (weights * bond_df['ModifiedDuration']).sum()
    port_convexity = (weights * bond_df['Convexity']).sum()
    
    print(f"     Portfolio Duration: {port_duration:.4f}")
    print(f"     Portfolio Convexity: {port_convexity:.4f}")
    print(f"     Total Market Value: {fmt_inr(total_mv)}")
    
    # 2. Vasicek Model
    print_subsection("Vasicek Model Simulation")
    
    # Get historical short rates (3M tenor)
    short_rates = yc_df[yc_df['Tenor_Years'] == 0.25].sort_values('CurveDate')['Yield'].values
    
    vasicek = VasicekModel()
    if len(short_rates) > 10:
        vasicek.calibrate_from_data(short_rates)
        print(f"  Calibrated Parameters:")
        print(f"    κ (mean reversion):  {vasicek.kappa:.4f}")
        print(f"    θ (long-run mean):   {vasicek.theta:.4f}")
        print(f"    σ (volatility):      {vasicek.sigma:.4f}")
        print(f"    r₀ (current rate):   {vasicek.r0:.4f}")
    
    times_v, paths_v = vasicek.simulate(T=2.0, n_steps=504, n_paths=5000)
    print(f"  Simulated {paths_v.shape[0]} paths over {2.0} years")
    print(f"  Terminal rate: mean={np.mean(paths_v[:,-1])*100:.2f}%, "
          f"std={np.std(paths_v[:,-1])*100:.2f}%")
    
    # 3. CIR Model
    print_subsection("CIR Model Simulation")
    cir = CIRModel(kappa=vasicek.kappa, theta=vasicek.theta,
                    sigma=max(vasicek.sigma * 3, 0.03), r0=vasicek.r0)
    print(f"  Feller condition (2κθ > σ²): {cir.feller_condition()}")
    
    times_c, paths_c = cir.simulate(T=2.0, n_steps=504, n_paths=5000)
    print(f"  Terminal rate: mean={np.mean(paths_c[:,-1])*100:.2f}%, "
          f"std={np.std(paths_c[:,-1])*100:.2f}%")
    
    # 4. Multi-Factor Analysis from provided MC data
    print_subsection("Multi-Factor Monte Carlo Analysis (Provided Data)")
    
    print(f"  Scenarios: {len(mc_df)}")
    print(f"  Factor Statistics:")
    for col in ['ParallelShift_bps', 'TwistFactor_bps', 'ButterflyFactor_bps']:
        print(f"    {col}: mean={mc_df[col].mean():.2f}, std={mc_df[col].std():.2f}")
    
    # 5. VaR & CVaR from provided scenarios
    print_subsection("Value at Risk & Expected Shortfall")
    
    pnl_data = mc_df['PnL_Total_INR'].values
    var_results = compute_var_cvar(pnl_data)
    print(var_results.to_string(index=False))
    
    # Compare VaR methods
    print_subsection("VaR Method Comparison")
    hist_var = historical_var(mc_df, 0.95)
    param_var = parametric_var(pnl_data, 0.95)
    print(f"  Historical VaR (95%):  {fmt_inr(hist_var)}")
    print(f"  Parametric VaR (95%):  {fmt_inr(param_var)}")
    
    # 6. Generate new scenarios using multi-factor
    print_subsection("Generated Multi-Factor Scenarios")
    mf_sim = MultiFactorSimulation(mc_df)
    new_scenarios = mf_sim.simulate_factors(n_scenarios=5000, seed=42)
    pnl_results = mf_sim.compute_portfolio_pnl(new_scenarios, port_duration, port_convexity, total_mv)
    
    new_var = compute_var_cvar(pnl_results['Total_PnL'].values)
    print("  VaR from generated scenarios:")
    print(new_var.to_string(index=False))
    
    # 7. Stress Tests
    print_subsection("Stress Testing")
    stress_results = run_stress_tests(port_duration, port_convexity, total_mv)
    print(stress_results[['Scenario', 'Shift_bps', 'Total_PnL_INR', 'PnL_Pct']].to_string(index=False))
    
    # 8. Visualizations
    print_subsection("Generating Visualizations")
    plot_rate_paths(times_v, paths_v, "Vasicek", FIGURES_DIR / "p3_vasicek_paths.png")
    plot_rate_paths(times_c, paths_c, "CIR", FIGURES_DIR / "p3_cir_paths.png")
    plot_pnl_distribution(pnl_data, var_results, FIGURES_DIR / "p3_pnl_distribution.png")
    plot_pnl_distribution(pnl_results['Total_PnL'].values, new_var,
                          FIGURES_DIR / "p3_simulated_pnl.png")
    plot_stress_test_waterfall(stress_results, FIGURES_DIR / "p3_stress_tests.png")
    plot_scenario_heatmap(mc_df, FIGURES_DIR / "p3_scenario_heatmap.png")
    
    # 9. Save reports
    var_results.to_csv(REPORTS_DIR / "part3_var_analysis.csv", index=False)
    stress_results.to_csv(REPORTS_DIR / "part3_stress_tests.csv", index=False)
    mc_df.describe().to_csv(REPORTS_DIR / "part3_mc_summary_stats.csv")
    
    print_section_header("PART 3 COMPLETE ✓")
    return var_results, stress_results


if __name__ == "__main__":
    run_part3()
