"""Dashboard Page 4: ML Predictions"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils import load_bond_portfolio
from src.part4_ml_models import engineer_features, prepare_data, train_random_forest, train_xgboost

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")
st.title("🤖 ML Models for Convexity Prediction")

df = load_bond_portfolio()

@st.cache_data
def run_ml_pipeline():
    X, y, feature_names, feat_df = engineer_features(df)
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = prepare_data(X, y)
    
    rf_model, rf_pred, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, tune=False)
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, tune=False)
    
    # Ensemble
    ensemble_pred = 0.4 * rf_pred + 0.6 * xgb_pred
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    ensemble_metrics = {
        'Model': 'Ensemble', 'R2': r2_score(y_test, ensemble_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        'MAE': mean_absolute_error(y_test, ensemble_pred),
    }
    
    return {
        'rf_pred': rf_pred, 'xgb_pred': xgb_pred, 'ensemble_pred': ensemble_pred,
        'y_test': y_test, 'metrics': [rf_metrics, xgb_metrics, ensemble_metrics],
        'rf_model': rf_model, 'xgb_model': xgb_model, 'feature_names': feature_names,
    }

with st.spinner("Training ML models..."):
    results = run_ml_pipeline()

tab1, tab2, tab3 = st.tabs(["Model Comparison", "Predictions", "Feature Importance"])

with tab1:
    metrics_df = pd.DataFrame(results['metrics'])
    
    c1, c2, c3 = st.columns(3)
    for i, row in metrics_df.iterrows():
        col = [c1, c2, c3][i]
        with col:
            st.markdown(f"### {row['Model']}")
            st.metric("R²", f"{row['R2']:.4f}")
            st.metric("RMSE", f"{row['RMSE']:.4f}")
            st.metric("MAE", f"{row['MAE']:.4f}")
    
    fig = px.bar(metrics_df, x='Model', y='R2', title='Model R² Comparison',
                 color='Model', text='R2')
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost", "Ensemble"])
    pred_map = {'Random Forest': results['rf_pred'], 'XGBoost': results['xgb_pred'], 'Ensemble': results['ensemble_pred']}
    y_pred = pred_map[model_choice]
    y_test = results['y_test']
    
    fig = px.scatter(x=y_test, y=y_pred, title=f'{model_choice}: Actual vs Predicted Convexity',
                     labels={'x': 'Actual Convexity', 'y': 'Predicted Convexity'},
                     opacity=0.6)
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines',
                              name='Perfect Fit', line=dict(dash='dash', color='red')))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if hasattr(results['xgb_model'], 'feature_importances_'):
        imp = pd.DataFrame({
            'Feature': results['feature_names'],
            'Importance': results['xgb_model'].feature_importances_
        }).sort_values('Importance', ascending=True).tail(15)
        
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                     title='XGBoost: Top 15 Feature Importance', color='Importance',
                     color_continuous_scale='Blues')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
