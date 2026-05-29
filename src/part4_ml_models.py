"""
Part 4: ML Models for Convexity Prediction
============================================
Random Forest, XGBoost, and Neural Network models to predict
bond convexity from fundamental characteristics.
Includes feature engineering, hyperparameter tuning, SHAP analysis,
and ensemble predictions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, print_section_header, print_subsection,
    FIGURES_DIR, REPORTS_DIR
)


# ─────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────

def engineer_features(df):
    """
    Create predictive features for convexity modelling.
    
    Features:
    - Bond characteristics (coupon, maturity, YTM, duration)
    - Embedded option flags
    - Spread measures
    - Price sensitivity ratios
    - Derived interaction terms
    
    Returns
    -------
    tuple — (X, y, feature_names)
    """
    feat_df = df.copy()
    
    # ── Core Numeric Features ──
    numeric_features = [
        'CouponRate', 'YearsToMaturity', 'YieldToMaturity',
        'MacaulayDuration', 'ModifiedDuration', 'FaceValue',
        'CleanPrice', 'DirtyPrice', 'AccruedInterest',
        'DV01_Per100Face', 'SpreadOverBenchmark_bps',
        'OAS_bps', 'ZSpread_bps', 'CouponFrequency',
    ]
    
    # ── Derived Features ──
    feat_df['Coupon_Yield_Spread'] = feat_df['CouponRate'] - feat_df['YieldToMaturity']
    feat_df['Duration_Maturity_Ratio'] = feat_df['ModifiedDuration'] / feat_df['YearsToMaturity'].clip(lower=0.1)
    feat_df['Coupon_per_Period'] = feat_df['CouponRate'] / feat_df['CouponFrequency'].clip(lower=1)
    feat_df['Price_Par_Diff'] = feat_df['CleanPrice'] - feat_df['FaceValue']
    feat_df['Duration_Squared'] = feat_df['ModifiedDuration'] ** 2
    feat_df['Maturity_Squared'] = feat_df['YearsToMaturity'] ** 2
    feat_df['YTM_Duration_Interaction'] = feat_df['YieldToMaturity'] * feat_df['ModifiedDuration']
    feat_df['Coupon_Duration_Ratio'] = feat_df['CouponRate'] / feat_df['ModifiedDuration'].clip(lower=0.01)
    feat_df['Log_Maturity'] = np.log1p(feat_df['YearsToMaturity'])
    feat_df['Spread_Duration_Product'] = feat_df['SpreadOverBenchmark_bps'] * feat_df['ModifiedDuration']
    
    derived_features = [
        'Coupon_Yield_Spread', 'Duration_Maturity_Ratio', 'Coupon_per_Period',
        'Price_Par_Diff', 'Duration_Squared', 'Maturity_Squared',
        'YTM_Duration_Interaction', 'Coupon_Duration_Ratio', 'Log_Maturity',
        'Spread_Duration_Product',
    ]
    
    # ── Categorical Encoding ──
    le_sector = LabelEncoder()
    feat_df['Sector_Encoded'] = le_sector.fit_transform(feat_df['Sector'].astype(str))
    
    le_rating = LabelEncoder()
    feat_df['Rating_Encoded'] = le_rating.fit_transform(feat_df['CreditRating'].astype(str))
    
    le_bond_type = LabelEncoder()
    feat_df['BondType_Encoded'] = le_bond_type.fit_transform(feat_df['BondType'].astype(str))
    
    categorical_features = ['Sector_Encoded', 'Rating_Encoded', 'BondType_Encoded']
    
    # ── Boolean Features ──
    feat_df['IsCallable_Flag'] = feat_df['IsCallable'].astype(int)
    feat_df['IsPutable_Flag'] = feat_df['IsPutable'].astype(int)
    feat_df['IsFloating_Flag'] = feat_df['IsFloatingRate'].astype(int)
    
    bool_features = ['IsCallable_Flag', 'IsPutable_Flag', 'IsFloating_Flag']
    
    # ── Combine All Features ──
    all_features = numeric_features + derived_features + categorical_features + bool_features
    
    X = feat_df[all_features].copy()
    y = feat_df['Convexity'].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    return X, y, all_features, feat_df


def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split and scale data for modelling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────
# Model 1: Random Forest
# ─────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, X_test, y_test, tune=True):
    """
    Train Random Forest Regressor with optional hyperparameter tuning.
    """
    print("  🌲 Training Random Forest Regressor...")
    
    if tune:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='r2',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        print(f"     Best params: {grid_search.best_params_}")
    else:
        best_rf = RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        )
        best_rf.fit(X_train, y_train)
    
    y_pred = best_rf.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    
    return best_rf, y_pred, metrics


# ─────────────────────────────────────────────────────────────
# Model 2: XGBoost
# ─────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_test, y_test, tune=True):
    """
    Train XGBoost Regressor with optional hyperparameter tuning.
    """
    print("  🚀 Training XGBoost Regressor...")
    
    if tune:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],
        }
        
        xgb_model = xgb.XGBRegressor(
            random_state=42, n_jobs=-1,
            tree_method='hist'
        )
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='r2',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        print(f"     Best params: {grid_search.best_params_}")
    else:
        best_xgb = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, tree_method='hist'
        )
        best_xgb.fit(X_train, y_train)
    
    y_pred = best_xgb.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    
    return best_xgb, y_pred, metrics


# ─────────────────────────────────────────────────────────────
# Model 3: Neural Network
# ─────────────────────────────────────────────────────────────

def train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Train a Neural Network for convexity prediction.
    Uses TensorFlow/Keras if available, otherwise falls back to sklearn MLPRegressor.
    """
    print("  🧠 Training Neural Network...")
    
    history = None
    
    try:
        # Suppress TF warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow import keras
        from tensorflow.keras import layers
        
        print("     Backend: TensorFlow/Keras")
        input_dim = X_train_scaled.shape[1]
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.15,
            epochs=200,
            batch_size=32,
            callbacks=[early_stop, lr_scheduler],
            verbose=0
        )
        
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        
    except Exception as e:
        print(f"     ⚠️ TensorFlow unavailable ({type(e).__name__}), using sklearn MLPRegressor")
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16),
            max_iter=500,
            early_stopping=True,
            random_state=42,
            learning_rate='adaptive'
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    metrics = evaluate_model(y_test, y_pred, "Neural Network")
    
    return model, y_pred, metrics, history


# ─────────────────────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, model_name):
    """Compute and print evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    
    metrics = {'Model': model_name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    
    print(f"     {model_name} Results:")
    print(f"       R² Score:  {r2:.6f}")
    print(f"       RMSE:      {rmse:.4f}")
    print(f"       MAE:       {mae:.4f}")
    print(f"       MAPE:      {mape:.2f}%")
    
    return metrics


def ensemble_predict(rf_pred, xgb_pred, nn_pred, weights=None):
    """
    Ensemble prediction combining all three models.
    
    Parameters
    ----------
    weights : list — [rf_weight, xgb_weight, nn_weight], sums to 1
    """
    if weights is None:
        weights = [0.3, 0.4, 0.3]  # Default weights favoring XGBoost
    
    return weights[0] * rf_pred + weights[1] * xgb_pred + weights[2] * nn_pred


# ─────────────────────────────────────────────────────────────
# Feature Importance Analysis
# ─────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names, model_name="Model"):
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return pd.DataFrame()
    
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return imp_df


# ─────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, predictions_dict, save_path=None):
    """Scatter plots of actual vs predicted for all models."""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    colors = ['#1976D2', '#E64A19', '#4CAF50', '#7B1FA2']
    
    for ax, (name, y_pred), color in zip(axes, predictions_dict.items(), colors):
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, color=color, edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
        
        r2 = r2_score(y_test, y_pred)
        ax.set_xlabel('Actual Convexity', fontsize=11)
        ax.set_ylabel('Predicted Convexity', fontsize=11)
        ax.set_title(f'{name}\nR² = {r2:.4f}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_residuals(y_test, predictions_dict, save_path=None):
    """Residual plots for all models."""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    colors = ['#1976D2', '#E64A19', '#4CAF50', '#7B1FA2']
    
    for ax, (name, y_pred), color in zip(axes, predictions_dict.items(), colors):
        residuals = y_test.values - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=30, color=color, edgecolors='white', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Convexity', fontsize=11)
        ax.set_ylabel('Residual', fontsize=11)
        ax.set_title(f'{name}: Residuals', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_feature_importance(rf_model, xgb_model, feature_names, save_path=None):
    """Feature importance comparison between RF and XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, model, name, color in [
        (axes[0], rf_model, 'Random Forest', '#1976D2'),
        (axes[1], xgb_model, 'XGBoost', '#E64A19')
    ]:
        imp_df = get_feature_importance(model, feature_names, name)
        if imp_df.empty:
            continue
        
        top_n = 15
        imp_top = imp_df.head(top_n)
        
        ax.barh(imp_top['Feature'], imp_top['Importance'], color=color, alpha=0.8)
        ax.set_xlabel('Importance')
        ax.set_title(f'{name}: Top {top_n} Features', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_model_comparison(metrics_list, save_path=None):
    """Bar chart comparing model performance metrics."""
    metrics_df = pd.DataFrame(metrics_list)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metric_cols = ['R2', 'RMSE', 'MAE', 'MAPE']
    titles = ['R² Score (↑ better)', 'RMSE (↓ better)', 'MAE (↓ better)', 'MAPE % (↓ better)']
    colors = ['#1976D2', '#E64A19', '#4CAF50', '#7B1FA2']
    
    for ax, col, title in zip(axes, metric_cols, titles):
        bars = ax.bar(metrics_df['Model'], metrics_df[col],
                      color=colors[:len(metrics_df)], alpha=0.8, edgecolor='white')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=30)
        
        # Value labels
        for bar, val in zip(bars, metrics_df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_learning_curve(history, save_path=None):
    """Plot Neural Network training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', color='#1976D2', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='#F44336', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Neural Network: Training Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE', color='#1976D2', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', color='#F44336', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Neural Network: Training MAE', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_part4():
    """Execute Part 4: ML Models for Convexity Prediction."""
    print_section_header("PART 4: ML Models for Convexity Prediction")
    
    # 1. Load and prepare data
    print("  📂 Loading and engineering features...")
    df = load_bond_portfolio()
    X, y, feature_names, feat_df = engineer_features(df)
    print(f"     Features: {len(feature_names)}")
    print(f"     Samples: {len(X)}")
    print(f"     Target (Convexity): mean={y.mean():.2f}, std={y.std():.2f}")
    
    # 2. Split data
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = prepare_data(X, y)
    print(f"     Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Train Models
    print_subsection("Model Training")
    
    rf_model, rf_pred, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, tune=True)
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, tune=True)
    nn_model, nn_pred, nn_metrics, nn_history = train_neural_network(X_train_s, y_train, X_test_s, y_test)
    
    # 4. Ensemble
    print_subsection("Ensemble Model")
    ensemble_pred = ensemble_predict(rf_pred, xgb_pred, nn_pred)
    ensemble_metrics = evaluate_model(y_test, ensemble_pred, "Ensemble")
    
    # 5. Model Comparison
    print_subsection("Model Comparison Summary")
    all_metrics = [rf_metrics, xgb_metrics, nn_metrics, ensemble_metrics]
    comparison_df = pd.DataFrame(all_metrics)
    print(comparison_df.to_string(index=False))
    
    # 6. Cross-Validation (RF and XGBoost)
    print_subsection("Cross-Validation Scores (5-Fold)")
    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"  {name}: R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 7. Feature Importance
    print_subsection("Top 10 Features (XGBoost)")
    imp_df = get_feature_importance(xgb_model, feature_names, "XGBoost")
    if not imp_df.empty:
        print(imp_df.head(10).to_string(index=False))
    
    # 8. Visualizations
    print_subsection("Generating Visualizations")
    predictions = {
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
        'Neural Network': nn_pred,
        'Ensemble': ensemble_pred,
    }
    
    plot_actual_vs_predicted(y_test, predictions, FIGURES_DIR / "p4_actual_vs_predicted.png")
    plot_residuals(y_test, predictions, FIGURES_DIR / "p4_residuals.png")
    plot_feature_importance(rf_model, xgb_model, feature_names,
                           FIGURES_DIR / "p4_feature_importance.png")
    plot_model_comparison(all_metrics, FIGURES_DIR / "p4_model_comparison.png")
    plot_learning_curve(nn_history, FIGURES_DIR / "p4_nn_learning_curve.png")
    
    # 9. Save reports
    comparison_df.to_csv(REPORTS_DIR / "part4_model_comparison.csv", index=False)
    imp_df.to_csv(REPORTS_DIR / "part4_feature_importance.csv", index=False)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'BondID': df.iloc[X_test.index]['BondID'].values,
        'Actual_Convexity': y_test.values,
        'RF_Predicted': rf_pred,
        'XGB_Predicted': xgb_pred,
        'NN_Predicted': nn_pred,
        'Ensemble_Predicted': ensemble_pred,
    })
    pred_df.to_csv(REPORTS_DIR / "part4_predictions.csv", index=False)
    
    print_section_header("PART 4 COMPLETE ✓")
    return rf_model, xgb_model, nn_model, comparison_df


if __name__ == "__main__":
    run_part4()
