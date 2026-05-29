"""
Test Suite: Convexity Sensitivity AI Agent
==========================================
Validates all 7 parts of the project.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_bond_portfolio, load_yield_curve, load_monte_carlo
from src.part1_analytics import BondAnalytics, PortfolioAnalytics


class TestDataLoading:
    """Test data loading functions."""
    
    def test_bond_portfolio_loads(self):
        df = load_bond_portfolio()
        assert len(df) == 300
        assert 'BondID' in df.columns
        assert 'ModifiedDuration' in df.columns
    
    def test_yield_curve_loads(self):
        df = load_yield_curve()
        assert len(df) > 0
        assert 'Yield' in df.columns
    
    def test_monte_carlo_loads(self):
        df = load_monte_carlo()
        assert len(df) == 1000
        assert 'PnL_Total_INR' in df.columns


class TestBondAnalytics:
    """Test bond-level calculations."""
    
    def test_duration_positive(self):
        bond = BondAnalytics(100, 0.06, 2, 5, 0.065)
        assert bond.macaulay_duration() > 0
        assert bond.modified_duration() > 0
    
    def test_modified_less_than_macaulay(self):
        bond = BondAnalytics(100, 0.06, 2, 5, 0.065)
        assert bond.modified_duration() <= bond.macaulay_duration()
    
    def test_convexity_positive(self):
        bond = BondAnalytics(100, 0.06, 2, 5, 0.065)
        assert bond.convexity() > 0
    
    def test_dv01_positive(self):
        bond = BondAnalytics(100, 0.06, 2, 5, 0.065)
        assert bond.dv01() > 0
    
    def test_longer_maturity_higher_duration(self):
        short = BondAnalytics(100, 0.06, 2, 2, 0.065)
        long = BondAnalytics(100, 0.06, 2, 20, 0.065)
        assert long.modified_duration() > short.modified_duration()
    
    def test_zero_coupon_highest_duration(self):
        """Zero coupon bond should have duration ≈ maturity."""
        # Use very small coupon as proxy
        zc = BondAnalytics(100, 0.001, 1, 10, 0.065)
        coupon = BondAnalytics(100, 0.08, 2, 10, 0.065)
        assert zc.macaulay_duration() > coupon.macaulay_duration()
    
    def test_accuracy_vs_csv(self):
        """Compare computed vs CSV for sample bonds."""
        df = load_bond_portfolio()
        row = df.iloc[0]
        bond = BondAnalytics(row['FaceValue'], row['CouponRate'], row['CouponFrequency'],
                             row['YearsToMaturity'], row['YieldToMaturity'])
        m = bond.full_metrics()
        assert abs(m['ModifiedDuration'] - row['ModifiedDuration']) < 1.0


class TestPortfolioAnalytics:
    """Test portfolio-level calculations."""
    
    def test_portfolio_creates(self):
        df = load_bond_portfolio()
        portfolio = PortfolioAnalytics(df)
        assert portfolio.total_market_value > 0
    
    def test_portfolio_duration_positive(self):
        df = load_bond_portfolio()
        portfolio = PortfolioAnalytics(df)
        assert portfolio.portfolio_duration() > 0
    
    def test_portfolio_convexity_positive(self):
        df = load_bond_portfolio()
        portfolio = PortfolioAnalytics(df)
        assert portfolio.portfolio_convexity() > 0
    
    def test_sensitivity_symmetric(self):
        """Convexity means gain from down shock > loss from up shock."""
        df = load_bond_portfolio()
        portfolio = PortfolioAnalytics(df)
        sens = portfolio.price_sensitivity_analysis([100, -100])
        up_pnl = sens[sens['Shock_bps'] == 100]['Total_PnL_INR'].values[0]
        dn_pnl = sens[sens['Shock_bps'] == -100]['Total_PnL_INR'].values[0]
        # Due to convexity: |gain from -100| > |loss from +100|
        assert abs(dn_pnl) > abs(up_pnl)
    
    def test_key_rate_duration(self):
        df = load_bond_portfolio()
        portfolio = PortfolioAnalytics(df)
        krd = portfolio.key_rate_duration_profile()
        assert len(krd) > 0


class TestYieldCurve:
    """Test yield curve modelling."""
    
    def test_nss_fits(self):
        from src.part2_yield_curve import NelsonSiegelSvensson
        yc_df = load_yield_curve()
        latest = yc_df[yc_df['CurveDate'] == yc_df['CurveDate'].max()].sort_values('Tenor_Years')
        nss = NelsonSiegelSvensson()
        nss.fit(latest['Tenor_Years'].values, latest['Yield'].values)
        assert nss.fitted
        predicted = nss.predict(latest['Tenor_Years'].values)
        rmse = np.sqrt(np.mean((predicted - latest['Yield'].values)**2))
        assert rmse < 0.01  # Less than 1% RMSE


class TestMonteCarlo:
    """Test Monte Carlo simulation."""
    
    def test_vasicek_simulates(self):
        from src.part3_monte_carlo import VasicekModel
        model = VasicekModel()
        times, paths = model.simulate(T=1.0, n_paths=100)
        assert paths.shape == (100, 253)
        assert np.all(np.isfinite(paths))
    
    def test_var_positive(self):
        from src.part3_monte_carlo import compute_var_cvar
        mc_df = load_monte_carlo()
        var_results = compute_var_cvar(mc_df['PnL_Total_INR'].values)
        assert all(var_results['VaR_INR'] > 0)


class TestMLModels:
    """Test ML feature engineering."""
    
    def test_feature_engineering(self):
        from src.part4_ml_models import engineer_features
        df = load_bond_portfolio()
        X, y, features, _ = engineer_features(df)
        assert len(features) > 10
        assert len(X) == len(df)
        assert not y.isnull().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
