# Convexity Sensitivity AI Agent

A comprehensive AI-powered bond analytics platform for duration, convexity, and DV01 sensitivity analysis with ML-powered predictions and gamified training.

## Project Overview

This platform provides end-to-end fixed income analytics for an INR bond portfolio (300 bonds), including:

- **Part 1**: Bond portfolio duration & convexity analytics framework
- **Part 2**: Yield curve modelling (Nelson-Siegel-Svensson) & DV01 sensitivity
- **Part 3**: Monte Carlo simulation (Vasicek/CIR) with VaR/CVaR
- **Part 4**: ML models (Random Forest, XGBoost, Neural Network) for convexity prediction
- **Part 5**: Power BI DAX measures & dashboard data exports
- **Part 6**: Bond Risk Lab — gamified simulation platform for training
- **Part 7**: Multi-scenario validation across yield curve shocks

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Everything (Single File)
```bash
# Run all 7 parts sequentially
python3 convexity_agent_complete.py

# Run a specific part
python3 convexity_agent_complete.py --part 1
python3 convexity_agent_complete.py --part 4
```

### Run Individual Modules
```bash
python3 -c "from src.part1_analytics import run_part1; run_part1()"
python3 -c "from src.part2_yield_curve import run_part2; run_part2()"
python3 -c "from src.part3_monte_carlo import run_part3; run_part3()"
python3 -c "from src.part4_ml_models import run_part4; run_part4()"
python3 -c "from src.part5_dax_measures import run_part5; run_part5()"
python3 -c "from src.part6_bond_risk_lab import run_part6; run_part6()"
python3 -c "from src.part7_validation import run_validation; run_validation()"
```

### Additional Analysis Modules
```bash
# SHAP Explainability
python3 -c "from src.shap_explainability import run_shap_analysis; run_shap_analysis()"

# ML vs Analytical Sensitivity Analysis
python3 -c "from src.sensitivity_analysis import run_sensitivity_analysis; run_sensitivity_analysis()"

# VaR Backtesting with Kupiec POF Test
python3 -c "from src.var_backtest import run_var_backtest; run_var_backtest()"
```

### Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
# OR
python3 run_dashboard.py
```

### Run Tests
```bash
python3 -m pytest tests/ -v
python3 -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Docker
```bash
# Start dashboard
docker-compose up --build

# Run analytics (one-shot)
docker-compose --profile analytics up analytics
```

## Project Structure

```
Project 2/
├── data/
│   ├── bond_portfolio_data.csv       # 300 INR bonds with 44 columns
│   ├── yield_curve_history.csv       # Historical yield curves (264 records)
│   └── monte_carlo_scenarios.csv     # 1000 MC scenarios with P&L
├── convexity_agent_complete.py       # ⭐ Single-file version (all 7 parts)
├── run_dashboard.py                  # Dashboard launcher
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker build configuration
├── docker-compose.yml                # Docker Compose services
├── LICENSE                           # Zetheta IP attribution
├── .env.example                      # Environment variable template
├── README.md                         # This file
├── src/
│   ├── part1_analytics.py            # Duration & convexity framework
│   ├── part2_yield_curve.py          # Yield curve modelling & DV01
│   ├── part2_yield_curve.R           # R implementation (yield curve)
│   ├── part3_monte_carlo.py          # Monte Carlo & VaR
│   ├── part4_ml_models.py            # ML prediction models
│   ├── part5_dax_measures.py         # Power BI DAX measures (42 measures)
│   ├── part6_bond_risk_lab.py        # Gamified platform
│   ├── part7_validation.py           # Validation suite
│   ├── shap_explainability.py        # SHAP analysis for best model
│   ├── sensitivity_analysis.py       # ML vs analytical sensitivity
│   ├── var_backtest.py               # VaR backtesting (Kupiec POF)
│   ├── duration_convexity_analysis.R # R: GBM/XGBoost/H2O models
│   ├── sensitivity_visualization.R   # R: ggplot2 sensitivity charts
│   └── utils.py                      # Shared utilities
├── dashboard/
│   ├── app.py                        # Streamlit main page
│   └── pages/                        # Dashboard sub-pages (6 pages)
├── outputs/
│   ├── figures/                      # Generated charts
│   ├── reports/                      # CSV analysis reports
│   └── powerbi_exports/              # Power BI data files
├── tests/
│   └── test_all_parts.py             # Test suite (>60% coverage)
└── docs/
    ├── DAX_measures.md               # Power BI DAX reference (42 measures)
    └── project_documentation.md      # Technical documentation
```

## Key Technical Details

### Bond Analytics (Part 1)
- Duration computed from **first principles** using cashflow discounting
- Portfolio metrics use **market-value weighting**
- Key rate duration decomposition across 8 tenor buckets

### Yield Curve (Part 2)
- **Nelson-Siegel-Svensson** 6-parameter model fitting
- **PCA** analysis extracting level/slope/curvature factors
- DV01 ladder showing sensitivity at each tenor

### Monte Carlo (Part 3)
- **Vasicek** and **CIR** short-rate model simulation (5000 paths)
- **VaR** at 90%, 95%, 99% confidence levels
- Stress testing across 10 historical/hypothetical scenarios

### ML Models (Part 4)
- **27 engineered features** from bond characteristics
- **Random Forest**, **XGBoost**, **Neural Network** (TF/Keras)
- **Ensemble** model combining all three
- **SHAP** explainability with summary, dependence, and force plots
- Expected R² > 0.99 (convexity is highly predictable from fundamentals)

### Sensitivity Analysis (Day 9)
- **50 yield perturbations** (-200bps to +200bps)
- ML-predicted vs analytical price change comparison
- Ensemble model outperformance analysis

### Power BI (Part 5)
- **42 DAX measure formulas** ready to paste into Power BI
- **6 CSV exports** formatted for Power BI import
- **What-If parameter** for yield change slider (-300bps to +300bps)

### Bond Risk Lab (Part 6)
- **Scenario challenges** with scoring
- **10 quiz questions** with explanations
- **8 achievement badges** and ranking system
- **4 risk profiles** (Conservative to Barbell)

### VaR Backtesting
- **Kupiec POF test** for VaR model adequacy
- **Basel II traffic light test** classification
- **Parametric vs Historical vs Cornish-Fisher** VaR comparison

## Data Description

| Column | Description |
|--------|------------|
| ModifiedDuration | Price sensitivity per 1% yield change |
| Convexity | Second-order sensitivity (curvature) |
| DV01_Per100Face | Price change per 1bp per ₹100 face |
| PriceChange_Up/Dn | Pre-computed shock impacts |
| KeyRateBucket | Tenor classification for KRD |

## Technology Stack

- **Python 3.13** — Core language
- **NumPy/Pandas/SciPy** — Data processing & scientific computing
- **Scikit-learn** — Random Forest, PCA, preprocessing
- **XGBoost** — Gradient boosted trees
- **TensorFlow/Keras** — Neural network (optional, sklearn fallback)
- **SHAP** — Model explainability
- **Matplotlib/Seaborn/Plotly** — Visualization
- **Streamlit** — Interactive dashboard
- **R** — Alternative implementation (GBM, XGBoost, H2O, ggplot2)
- **Docker** — Containerized deployment

## License

STRICTLY PRIVATE & CONFIDENTIAL — All work remains property of Zetheta Algorithms Private Limited.
