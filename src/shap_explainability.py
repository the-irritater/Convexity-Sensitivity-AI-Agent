"""
SHAP Explainability Module
===========================
SHAP (SHapley Additive exPlanations) analysis for the best-performing
ML model to interpret feature contributions to convexity predictions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, print_section_header, print_subsection,
    FIGURES_DIR, REPORTS_DIR
)
from src.part4_ml_models import engineer_features, prepare_data, train_xgboost


# ─────────────────────────────────────────────────────────────
# SHAP Analysis
# ─────────────────────────────────────────────────────────────

def compute_shap_values(model, X_test, feature_names):
    """
    Compute SHAP values for the given model and test data.

    Parameters
    ----------
    model : trained model — Must support tree-based SHAP explainer
    X_test : pd.DataFrame — Test features
    feature_names : list — Feature names

    Returns
    -------
    tuple — (shap_values, explainer)
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        return shap_values, explainer
    except ImportError:
        print("  ⚠️ SHAP package not installed. Install with: pip install shap")
        print("  Generating approximate feature importance instead.")
        return None, None


def plot_shap_summary(shap_values, X_test, feature_names, save_path=None):
    """
    Generate SHAP summary (beeswarm) plot showing feature impact.
    """
    if shap_values is None:
        print("  ⚠️ SHAP values not available, skipping summary plot.")
        return None

    try:
        import shap
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          show=False, max_display=20)
        plt.title('SHAP Feature Impact on Convexity Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Saved: {save_path}")
        plt.close()
        return fig
    except Exception as e:
        print(f"  ⚠️ SHAP summary plot failed: {e}")
        return None


def plot_shap_bar(shap_values, X_test, feature_names, save_path=None):
    """
    Generate SHAP bar plot showing mean absolute SHAP values.
    """
    if shap_values is None:
        return _plot_fallback_importance(X_test, feature_names, save_path)

    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type='bar', show=False, max_display=20)
        plt.title('Mean |SHAP Value| — Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Saved: {save_path}")
        plt.close()
        return fig
    except Exception as e:
        print(f"  ⚠️ SHAP bar plot failed: {e}")
        return None


def plot_shap_dependence(shap_values, X_test, feature_name, interaction_feature=None,
                         save_path=None):
    """
    Generate SHAP dependence plot for a specific feature.

    Parameters
    ----------
    feature_name : str — Primary feature to plot
    interaction_feature : str — Feature to use for coloring (optional)
    """
    if shap_values is None:
        print(f"  ⚠️ SHAP values not available, skipping dependence plot for {feature_name}.")
        return None

    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature_name, shap_values, X_test,
            interaction_index=interaction_feature,
            show=False, ax=ax
        )
        ax.set_title(f'SHAP Dependence: {feature_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Saved: {save_path}")
        plt.close()
        return fig
    except Exception as e:
        print(f"  ⚠️ SHAP dependence plot failed for {feature_name}: {e}")
        return None


def plot_shap_force(explainer, shap_values, X_test, idx=0, save_path=None):
    """
    Generate SHAP force plot for a single prediction.
    """
    if shap_values is None or explainer is None:
        return None

    try:
        import shap
        force_plot = shap.force_plot(
            explainer.expected_value, shap_values[idx, :],
            X_test.iloc[idx, :], matplotlib=True, show=False
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Saved: {save_path}")
        plt.close()
        return force_plot
    except Exception as e:
        print(f"  ⚠️ SHAP force plot failed: {e}")
        return None


def _plot_fallback_importance(X_test, feature_names, save_path=None):
    """Fallback: correlation-based feature importance when SHAP is unavailable."""
    fig, ax = plt.subplots(figsize=(10, 8))
    correlations = X_test.corrwith(pd.Series(np.zeros(len(X_test)))).abs().sort_values(ascending=True)
    ax.barh(range(min(20, len(correlations))),
            correlations.tail(20).values, color='#1976D2', alpha=0.8)
    ax.set_yticks(range(min(20, len(correlations))))
    ax.set_yticklabels(correlations.tail(20).index)
    ax.set_xlabel('Absolute Correlation')
    ax.set_title('Feature Importance (Fallback — Correlation)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
    plt.close(fig)
    return fig


def generate_shap_report(shap_values, X_test, feature_names):
    """
    Generate a tabular SHAP summary report.

    Returns
    -------
    pd.DataFrame — Feature-level SHAP statistics
    """
    if shap_values is None:
        return pd.DataFrame({'Feature': feature_names,
                             'Mean_Abs_SHAP': [0.0] * len(feature_names)})

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    report = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Std_SHAP': np.std(shap_values, axis=0),
        'Max_SHAP': np.max(np.abs(shap_values), axis=0),
        'Rank': np.argsort(-mean_abs_shap) + 1,
    }).sort_values('Mean_Abs_SHAP', ascending=False)

    return report


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_shap_analysis():
    """Execute SHAP explainability analysis on the best ML model."""
    print_section_header("SHAP Explainability Analysis")

    # 1. Load and prepare data
    print("  Loading and preparing data...")
    df = load_bond_portfolio()
    X, y, feature_names, feat_df = engineer_features(df)
    X_train, X_test, y_train, y_test, _, _, _ = prepare_data(X, y)

    # 2. Train best model (XGBoost)
    print_subsection("Training XGBoost Model")
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, tune=False)

    # 3. Compute SHAP values
    print_subsection("Computing SHAP Values")
    shap_values, explainer = compute_shap_values(xgb_model, X_test, feature_names)

    # 4. Generate plots
    print_subsection("Generating SHAP Visualizations")

    plot_shap_summary(shap_values, X_test, feature_names,
                      FIGURES_DIR / "shap_summary.png")
    plot_shap_bar(shap_values, X_test, feature_names,
                  FIGURES_DIR / "shap_bar_importance.png")

    # Dependence plots for key features
    for feat in ['YieldToMaturity', 'YearsToMaturity', 'ModifiedDuration', 'Duration_Squared']:
        if feat in feature_names:
            plot_shap_dependence(
                shap_values, X_test, feat,
                save_path=FIGURES_DIR / f"shap_dependence_{feat.lower()}.png"
            )

    # Force plot for first test sample
    plot_shap_force(explainer, shap_values, X_test, idx=0,
                    save_path=FIGURES_DIR / "shap_force_sample.png")

    # 5. Generate report
    print_subsection("SHAP Feature Importance Report")
    report = generate_shap_report(shap_values, X_test, feature_names)
    print(report.head(15).to_string(index=False))

    report.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    print_section_header("SHAP ANALYSIS COMPLETE ✅")
    return shap_values, report


if __name__ == "__main__":
    run_shap_analysis()
