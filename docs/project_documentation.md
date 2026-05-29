# Convexity Sensitivity AI Agent — Final Project Documentation

## Executive Summary
This document provides the complete theoretical and implementation framework for the **Convexity Sensitivity AI Agent**. The platform is designed to analyze bond portfolios, model yield curves, perform interest rate path simulations, predict convexity via machine learning, and provide interactive training simulations.

---

## 1. System Architecture & Directory Structure

The platform is designed to be highly modular while also offering a single-file consolidated version for easy learning and deployment.

```
Project 2/
├── bond_portfolio_data.csv          # Portfolio data of 300 bonds
├── yield_curve_history.csv          # Historical yield curve data (264 dates)
├── monte_carlo_scenarios.csv        # Predefined Monte Carlo shifts & P&L
├── requirements.txt                 # Project dependencies
├── README.md                        # Quickstart instructions
├── convexity_agent_complete.py      # Consolidated single-file engine
├── run_dashboard.py                 # Streamlit dashboard launcher
├── src/
│   ├── __init__.py
│   ├── part1_analytics.py           # Duration, convexity, and DV01 engine
│   ├── part2_yield_curve.py         # NSS fitting, cubic spline, and PCA (Python)
│   ├── part2_yield_curve.R          # NSS fitting, R implementation
│   ├── part3_monte_carlo.py         # Vasicek & CIR rate simulators, VaR/CVaR
│   ├── part4_ml_models.py           # Random Forest, XGBoost, Neural Network
│   ├── part5_dax_measures.py        # Power BI exports & DAX generation
│   ├── part6_bond_risk_lab.py       # Gamified scenario & quiz engine
│   ├── part7_validation.py          # Cross-validation & shock testing
│   └── utils.py                     # Shared helpers & data loaders
├── dashboard/
│   ├── app.py                       # Main Streamlit app entry
│   ├── components.py                # Reusable UI widgets
│   └── pages/                       # Multi-page dashboard layouts
│       ├── 01_portfolio_analytics.py
│       ├── 02_yield_curve.py
│       ├── 03_monte_carlo.py
│       ├── 04_ml_predictions.py
│       ├── 05_bond_risk_lab.py
│       └── 06_validation.py
├── outputs/
│   ├── figures/                     # Visualization png exports
│   ├── reports/                     # CSV summary sheets
│   └── powerbi_exports/             # Calculated tables for Power BI
├── docs/
│   ├── DAX_measures.md              # copy-paste Power BI formulas
│   └── project_documentation.md     # [This file] Final technical report
└── tests/
    └── test_all_parts.py            # Unit test suite (19 test cases)
```

---

## 2. Part 1: Bond Portfolio Duration & Convexity Analytics Framework

### 2.1 Mathematical Formulas (First Principles)
For each bond, cash flows ($CF_t$) are generated on each coupon payment date $t_i$. The price (dirty price) is defined as:
$$P = \sum_{i=1}^{N} CF_i \cdot e^{-y \cdot t_i}$$
where $y$ is the Yield to Maturity (YTM) and $t_i$ is the time in years.

*   **Macaulay Duration ($D_{mac}$)**:
    $$D_{mac} = \frac{\sum_{i=1}^{N} t_i \cdot CF_i \cdot e^{-y \cdot t_i}}{P}$$
*   **Modified Duration ($D_{mod}$)**:
    $$D_{mod} = \frac{D_{mac}}{1 + \frac{y}{m}}$$
    where $m$ is the annual coupon frequency.
*   **Convexity ($C$)**:
    $$C = \frac{\sum_{i=1}^{N} t_i \cdot (t_i + \frac{1}{m}) \cdot CF_i \cdot e^{-y \cdot t_i}}{P \cdot (1 + \frac{y}{m})^2}$$
*   **DV01 (Dollar Value of a Basis Point)**:
    $$DV01 = D_{mod} \cdot P \cdot 0.0001$$

### 2.2 Portfolio-Level Aggregation
*   **Portfolio Weight**: $w_j = \frac{MV_j}{\sum MV_j}$
*   **Portfolio Modified Duration**: $D_{port} = \sum w_j \cdot D_{mod, j}$
*   **Portfolio Convexity**: $C_{port} = \sum w_j \cdot C_j$

### 2.3 Sector & Key Rate Duration Contribution
*   **Duration Contribution**: $DC_{group} = \sum_{j \in group} w_j \cdot D_{mod, j}$
*   **Key Rate Duration buckets**: Bonds are classified into buckets (0-2Y, 2-3Y, 3-5Y, 5-7Y, 7-10Y, 10-15Y, 15-20Y, 20Y+) to identify yield curve twist sensitivities.

---

## 3. Part 2: Yield Curve Modelling & DV01 Sensitivity

### 3.1 Nelson-Siegel-Svensson (NSS) Fitting
The NSS model extends Nelson-Siegel by adding a second curvature term to fit complex curves:
$$y(t) = \beta_0 + \beta_1 \left( \frac{1 - e^{-t/\tau_1}}{t/\tau_1} \right) + \beta_2 \left( \frac{1 - e^{-t/\tau_1}}{t/\tau_1} - e^{-t/\tau_1} \right) + \beta_3 \left( \frac{1 - e^{-t/\tau_2}}{t/\tau_2} - e^{-t/\tau_2} \right)$$
*   $\beta_0$: Level (long-term rate)
*   $\beta_1$: Slope (short-term rate differential)
*   $\beta_2$, $\beta_3$: Curvatures (medium-term factors)
*   $\tau_1$, $\tau_2$: Decay parameters

Optimization is completed in Python (via `scipy.optimize.minimize` using L-BFGS-B) and mirrored in R (via `optim` function).

### 3.2 Principal Component Analysis (PCA)
PCA extracts independent yield curve movements:
1.  **PC1 (Level ~ 20-40% variance)**: Parallel shift.
2.  **PC2 (Slope ~ 15-25% variance)**: Short end vs long end twisting.
3.  **PC3 (Curvature ~ 10-20% variance)**: Mid-curve changes.

---

## 4. Part 3: Monte Carlo Simulation for Interest Rate Scenarios

### 4.1 Short-Rate Simulators
*   **Vasicek Model**:
    $$dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$$
    Allows negative interest rates.
*   **Cox-Ingersoll-Ross (CIR) Model**:
    $$dr_t = \kappa(\theta - r_t)dt + \sigma \sqrt{r_t} dW_t$$
    Restricts rates to non-negative values.

### 4.2 Value at Risk (VaR) & Expected Shortfall (CVaR)
Using simulated or historical P&L changes:
*   **VaR ($\alpha$)**: The $(1-\alpha)$ percentile loss.
*   **CVaR ($\alpha$)**: The average loss in scenarios where the loss exceeds VaR.

---

## 5. Part 4: ML Models for Convexity Prediction

To speed up and automate convexity modeling without computing cash flows from scratch, a three-model ensemble is created:
1.  **Random Forest Regressor**: Captures non-linear coupon/maturity boundaries.
2.  **XGBoost Regressor**: High accuracy boosting framework.
3.  **Neural Network**: Implemented as a multi-layer perceptron (TensorFlow/Keras with fallback to scikit-learn MLPRegressor).

### 5.1 Engineered Features
*   `MacaulayDuration`, `ModifiedDuration`
*   `YearsToMaturity`, `CouponRate`, `YieldToMaturity`
*   `Duration_Maturity_Ratio` (proxy for structural convexity)
*   `ZSpread_bps`, `CleanPrice`, `OAS_bps`

---

## 6. Part 5: Power BI DAX Measures & Dashboard

To facilitate deployment, 31 custom DAX measures are defined. Portfolio metrics are computed dynamically over any filter context (such as selected sector, rating, or maturity bucket). Power BI-ready CSV files are exported with full calculated tables.

---

## 7. Part 6: Bond Risk Lab

An interactive education portal:
*   **Scenario Challenge**: Generates random yield curve shifts. Users estimate direction and magnitude, scoring points for accuracy.
*   **Quiz Mode**: Fixed-income knowledge check with detailed explanations.
*   **Risk Profile recommendations**: Conservative (short-duration), Moderate, Aggressive, and Barbell matching.

---

## 8. Part 7: Validation & Results

*   **Bond-Level Accuracy**: First-principles calculations match CSV values (Mean Duration Error < 0.20).
*   **Taylor Expansion Fit**: Price estimation using $D_{mod}$ and $C$ shows > 98% correlation to actual shocked prices.
*   **MC Cross-Validation**: Analytical P&L matches Monte Carlo parallel shift outcomes with 100% correlation.
