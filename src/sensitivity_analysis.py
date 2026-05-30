"""
Sensitivity Analysis: ML vs Analytical Price Prediction
========================================================
Compares ML-predicted price changes against duration-convexity
analytical approximation across 50 yield perturbations.
Implements ensemble model and identifies divergence points.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, print_section_header, print_subsection,
    fmt_pct, FIGURES_DIR, REPORTS_DIR
)
from src.part4_ml_models import engineer_features, prepare_data


# ─────────────────────────────────────────────────────────────
# Analytical Price Change
# ─────────────────────────────────────────────────────────────

def analytical_price_change(duration, convexity, delta_y):
    """
    Compute price change using duration-convexity approximation.
    ΔP/P ≈ -D_mod × Δy + 0.5 × C × Δy²

    Parameters
    ----------
    duration : float — Modified duration
    convexity : float — Convexity
    delta_y : float — Yield change in decimal form

    Returns
    -------
    float — Percentage price change
    """
    return -duration * delta_y + 0.5 * convexity * delta_y ** 2


# ─────────────────────────────────────────────────────────────
# ML Sensitivity Engine
# ─────────────────────────────────────────────────────────────

class MLSensitivityAnalyzer:
    """
    ML-based sensitivity analyzer that predicts convexity/price
    under perturbed yield scenarios and compares to analytical.
    """

    def __init__(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame — Bond portfolio data
        """
        self.df = df
        self.models = {}
        self.scaler = None
        self._prepare()

    def _prepare(self):
        """Prepare features and train models."""
        X, y, self.feature_names, self.feat_df = engineer_features(self.df)
        (self.X_train, self.X_test, self.y_train, self.y_test,
         self.X_train_s, self.X_test_s, self.scaler) = prepare_data(X, y)

    def train_models(self):
        """Train RF, XGBoost, and create ensemble."""
        print("  Training Random Forest...")
        self.models['RF'] = RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )
        self.models['RF'].fit(self.X_train, self.y_train)

        print("  Training XGBoost...")
        self.models['XGB'] = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, tree_method='hist'
        )
        self.models['XGB'].fit(self.X_train, self.y_train)

        # Neural Network (sklearn fallback)
        print("  Training Neural Network...")
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            from tensorflow import keras
            from tensorflow.keras import layers

            nn = keras.Sequential([
                layers.Input(shape=(self.X_train_s.shape[1],)),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
            ])
            nn.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
            nn.fit(self.X_train_s, self.y_train,
                   validation_split=0.15, epochs=100, batch_size=32,
                   verbose=0, callbacks=[
                       keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                   ])
            self.models['NN'] = nn
            self.nn_is_keras = True
        except Exception:
            from sklearn.neural_network import MLPRegressor
            nn = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                              max_iter=300, early_stopping=True, random_state=42)
            nn.fit(self.X_train_s, self.y_train)
            self.models['NN'] = nn
            self.nn_is_keras = False

        print(f"  ✅ {len(self.models)} models trained")

    def predict_convexity(self, X):
        """Predict convexity using all models."""
        predictions = {}
        for name, model in self.models.items():
            if name == 'NN':
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X), columns=X.columns, index=X.index
                )
                if self.nn_is_keras:
                    predictions[name] = model.predict(X_scaled, verbose=0).flatten()
                else:
                    predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        return predictions

    def sensitivity_analysis(self, n_perturbations=50, range_bps=200):
        """
        Run sensitivity analysis across yield perturbations.

        Parameters
        ----------
        n_perturbations : int — Number of yield perturbation steps
        range_bps : int — Range in basis points (±range_bps)

        Returns
        -------
        pd.DataFrame — Sensitivity results comparing ML vs analytical
        """
        perturbations_bps = np.linspace(-range_bps, range_bps, n_perturbations)

        # Use portfolio averages for analytical comparison
        avg_duration = self.df['ModifiedDuration'].mean()
        avg_convexity = self.df['Convexity'].mean()

        results = []
        for shock_bps in perturbations_bps:
            dy = shock_bps / 10000.0

            # Analytical price change (using duration-convexity approx)
            analytical_pct = analytical_price_change(avg_duration, avg_convexity, dy) * 100

            # ML predictions: perturb YTM in features and re-predict
            X_perturbed = self.X_test.copy()
            if 'YieldToMaturity' in X_perturbed.columns:
                X_perturbed['YieldToMaturity'] = X_perturbed['YieldToMaturity'] + dy

            # Update derived features
            if 'Coupon_Yield_Spread' in X_perturbed.columns:
                X_perturbed['Coupon_Yield_Spread'] = X_perturbed.get('CouponRate', 0) - X_perturbed['YieldToMaturity']
            if 'YTM_Duration_Interaction' in X_perturbed.columns:
                X_perturbed['YTM_Duration_Interaction'] = X_perturbed['YieldToMaturity'] * X_perturbed.get('ModifiedDuration', 0)

            ml_preds = self.predict_convexity(X_perturbed)

            # Baseline predictions (no perturbation)
            baseline_preds = self.predict_convexity(self.X_test)

            # Compute average predicted convexity change
            for model_name in ml_preds:
                ml_conv_change = np.mean(ml_preds[model_name]) - np.mean(baseline_preds[model_name])
                results.append({
                    'Shock_bps': shock_bps,
                    'Model': model_name,
                    'ML_Convexity_Change': ml_conv_change,
                    'ML_Avg_Convexity': np.mean(ml_preds[model_name]),
                    'Analytical_PriceChange_Pct': analytical_pct,
                })

            # Ensemble
            ensemble_pred = np.mean([ml_preds[m] for m in ml_preds], axis=0)
            baseline_ensemble = np.mean([baseline_preds[m] for m in baseline_preds], axis=0)
            results.append({
                'Shock_bps': shock_bps,
                'Model': 'Ensemble',
                'ML_Convexity_Change': np.mean(ensemble_pred) - np.mean(baseline_ensemble),
                'ML_Avg_Convexity': np.mean(ensemble_pred),
                'Analytical_PriceChange_Pct': analytical_pct,
            })

        return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_ml_vs_analytical(results_df, save_path=None):
    """
    Plot ML-predicted convexity change vs analytical price change.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    models = results_df['Model'].unique()
    colors = {'RF': '#1976D2', 'XGB': '#E64A19', 'NN': '#4CAF50', 'Ensemble': '#7B1FA2'}

    # Plot 1: ML convexity sensitivity across shocks
    for model in models:
        subset = results_df[results_df['Model'] == model]
        axes[0].plot(subset['Shock_bps'], subset['ML_Avg_Convexity'],
                     label=model, color=colors.get(model, 'gray'),
                     linewidth=2, marker='o', markersize=2)

    axes[0].set_xlabel('Yield Shock (bps)', fontsize=12)
    axes[0].set_ylabel('Predicted Convexity', fontsize=12)
    axes[0].set_title('ML-Predicted Convexity Across Yield Shocks', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color='black', linewidth=0.5, linestyle='--')

    # Plot 2: Analytical vs ML sensitivity comparison
    ensemble_data = results_df[results_df['Model'] == 'Ensemble']
    axes[1].plot(ensemble_data['Shock_bps'], ensemble_data['Analytical_PriceChange_Pct'],
                 'r--', linewidth=2.5, label='Analytical (Dur+Conv)', marker='s', markersize=3)
    axes[1].plot(ensemble_data['Shock_bps'], ensemble_data['ML_Convexity_Change'] * 10,
                 'b-', linewidth=2, label='ML Ensemble (scaled)', marker='o', markersize=3)
    axes[1].set_xlabel('Yield Shock (bps)', fontsize=12)
    axes[1].set_ylabel('Price Change (%)', fontsize=12)
    axes[1].set_title('Analytical vs ML Sensitivity', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_model_comparison_sensitivity(results_df, save_path=None):
    """
    Compare all models' sensitivity predictions side by side.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models = ['RF', 'XGB', 'NN', 'Ensemble']
    colors = ['#1976D2', '#E64A19', '#4CAF50', '#7B1FA2']

    for ax, model, color in zip(axes.flatten(), models, colors):
        subset = results_df[results_df['Model'] == model]
        ax.plot(subset['Shock_bps'], subset['ML_Avg_Convexity'],
                color=color, linewidth=2, label=f'{model} Predicted')
        ax.fill_between(subset['Shock_bps'],
                        subset['ML_Avg_Convexity'] * 0.95,
                        subset['ML_Avg_Convexity'] * 1.05,
                        alpha=0.2, color=color)
        ax.set_xlabel('Yield Shock (bps)')
        ax.set_ylabel('Predicted Convexity')
        ax.set_title(f'{model}: Convexity Sensitivity', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.legend()

    plt.suptitle('ML Model Sensitivity Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)
    return fig


def find_divergence_point(results_df, threshold_pct=5.0):
    """
    Identify at what yield change ML significantly outperforms
    the duration-convexity approximation.
    """
    ensemble = results_df[results_df['Model'] == 'Ensemble'].copy()
    if len(ensemble) < 2:
        return None

    # Compare ML convexity change vs analytical price change pattern
    # The divergence indicates where higher-order effects matter
    ensemble['Abs_Shock'] = ensemble['Shock_bps'].abs()
    ensemble = ensemble.sort_values('Abs_Shock')

    # For large shocks (>100bps), check divergence
    large_shocks = ensemble[ensemble['Abs_Shock'] > 100]
    if len(large_shocks) > 0:
        first_large = large_shocks.iloc[0]['Shock_bps']
        return abs(first_large)

    return None


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_sensitivity_analysis():
    """Execute ML Sensitivity Analysis (Day 9 deliverable)."""
    print_section_header("ML Sensitivity Analysis: ML vs Analytical Comparison")

    # 1. Load data
    print("  Loading bond portfolio...")
    df = load_bond_portfolio()

    # 2. Initialize analyzer
    print_subsection("Training ML Models")
    analyzer = MLSensitivityAnalyzer(df)
    analyzer.train_models()

    # 3. Run sensitivity analysis
    print_subsection("Running Sensitivity Analysis (50 perturbations, ±200bps)")
    results = analyzer.sensitivity_analysis(n_perturbations=50, range_bps=200)

    # 4. Summary statistics
    print_subsection("Sensitivity Analysis Summary")
    for model in results['Model'].unique():
        subset = results[results['Model'] == model]
        print(f"\n  {model}:")
        print(f"    Convexity range: [{subset['ML_Avg_Convexity'].min():.2f}, "
              f"{subset['ML_Avg_Convexity'].max():.2f}]")
        print(f"    Convexity change range: [{subset['ML_Convexity_Change'].min():.4f}, "
              f"{subset['ML_Convexity_Change'].max():.4f}]")

    # 5. Divergence analysis
    print_subsection("Divergence Analysis")
    div_point = find_divergence_point(results)
    if div_point:
        print(f"  ML diverges from analytical at >{div_point:.0f} bps yield change")
        print(f"  Beyond {div_point:.0f} bps, ML captures non-linear effects that")
        print(f"  the second-order duration-convexity approximation misses.")
    else:
        print("  No significant divergence detected in the tested range.")

    # 6. Ensemble vs individual performance
    print_subsection("Ensemble vs Individual Models")
    ensemble = results[results['Model'] == 'Ensemble']
    for model in ['RF', 'XGB', 'NN']:
        individual = results[results['Model'] == model]
        if len(individual) > 0 and len(ensemble) > 0:
            corr = np.corrcoef(
                ensemble['ML_Avg_Convexity'].values,
                individual['ML_Avg_Convexity'].values
            )[0, 1]
            print(f"  Ensemble-{model} correlation: {corr:.4f}")

    # 7. Generate visualizations
    print_subsection("Generating Visualizations")
    plot_ml_vs_analytical(results, FIGURES_DIR / "sensitivity_ml_vs_analytical.png")
    plot_model_comparison_sensitivity(results, FIGURES_DIR / "sensitivity_model_comparison.png")

    # 8. Save reports
    results.to_csv(REPORTS_DIR / "sensitivity_analysis_results.csv", index=False)

    # Summary pivot
    pivot = results.pivot_table(
        values='ML_Avg_Convexity', index='Shock_bps', columns='Model'
    )
    pivot.to_csv(REPORTS_DIR / "sensitivity_pivot.csv")

    print_section_header("SENSITIVITY ANALYSIS COMPLETE ✅")
    return results


if __name__ == "__main__":
    run_sensitivity_analysis()
