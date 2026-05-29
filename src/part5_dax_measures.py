"""
Part 5: Power BI DAX Measures & Dashboard Data Export
======================================================
Generates DAX measure formulas for Power BI, creates Power BI-ready
CSV exports, and documents all measures.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    load_bond_portfolio, load_monte_carlo,
    print_section_header, print_subsection,
    POWERBI_DIR, REPORTS_DIR
)


# ─────────────────────────────────────────────────────────────
# DAX Measure Definitions
# ─────────────────────────────────────────────────────────────

DAX_MEASURES = {
    # ── Portfolio Aggregation Measures ──
    "Total Market Value": {
        "category": "Portfolio Aggregation",
        "formula": 'Total Market Value = SUM(BondPortfolio[MarketValue_INR])',
        "description": "Sum of all bond market values in INR"
    },
    "Bond Count": {
        "category": "Portfolio Aggregation",
        "formula": 'Bond Count = COUNTROWS(BondPortfolio)',
        "description": "Total number of bonds in portfolio"
    },
    "Average Coupon Rate": {
        "category": "Portfolio Aggregation",
        "formula": '''Average Coupon Rate =
    SUMX(BondPortfolio,
        BondPortfolio[CouponRate] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Market-value weighted average coupon rate"
    },

    # ── Duration Measures ──
    "Portfolio Modified Duration": {
        "category": "Duration",
        "formula": '''Portfolio Modified Duration =
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Market-value weighted portfolio modified duration"
    },
    "Portfolio Macaulay Duration": {
        "category": "Duration",
        "formula": '''Portfolio Macaulay Duration =
    SUMX(BondPortfolio,
        BondPortfolio[MacaulayDuration] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Market-value weighted portfolio Macaulay duration"
    },
    "Duration Contribution": {
        "category": "Duration",
        "formula": '''Duration Contribution =
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR]
        / CALCULATE([Total Market Value], ALL(BondPortfolio))
    )''',
        "description": "Duration contribution of current filter context (sector, rating, etc.)"
    },
    "Effective Duration": {
        "category": "Duration",
        "formula": '''Effective Duration =
    SUMX(BondPortfolio,
        BondPortfolio[EffectiveDuration] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Weighted average effective duration (for bonds with optionality)"
    },

    # ── Convexity Measures ──
    "Portfolio Convexity": {
        "category": "Convexity",
        "formula": '''Portfolio Convexity =
    SUMX(BondPortfolio,
        BondPortfolio[Convexity] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Market-value weighted portfolio convexity"
    },
    "Convexity Contribution": {
        "category": "Convexity",
        "formula": '''Convexity Contribution =
    SUMX(BondPortfolio,
        BondPortfolio[Convexity] * BondPortfolio[MarketValue_INR]
        / CALCULATE([Total Market Value], ALL(BondPortfolio))
    )''',
        "description": "Convexity contribution of filtered subset"
    },
    "Effective Convexity": {
        "category": "Convexity",
        "formula": '''Effective Convexity =
    SUMX(BondPortfolio,
        BondPortfolio[EffectiveConvexity] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Weighted average effective convexity"
    },

    # ── DV01 Measures ──
    "Portfolio DV01": {
        "category": "DV01 / Sensitivity",
        "formula": '''Portfolio DV01 =
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR] * 0.0001
    )''',
        "description": "Total portfolio DV01 in INR (price change per 1bp yield move)"
    },
    "DV01 per Crore": {
        "category": "DV01 / Sensitivity",
        "formula": '''DV01 per Crore =
    [Portfolio DV01] / ([Total Market Value] / 10000000)''',
        "description": "DV01 normalized per crore of market value"
    },
    "DV01 Contribution Pct": {
        "category": "DV01 / Sensitivity",
        "formula": '''DV01 Contribution % =
    DIVIDE(
        SUMX(BondPortfolio, BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR] * 0.0001),
        CALCULATE(
            SUMX(BondPortfolio, BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR] * 0.0001),
            ALL(BondPortfolio)
        ),
        0
    )''',
        "description": "DV01 contribution as percentage of total"
    },

    # ── Price Sensitivity Measures ──
    "PnL Impact +50bps": {
        "category": "Price Sensitivity",
        "formula": '''PnL Impact +50bps =
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Up50bps] / 100 * BondPortfolio[MarketValue_INR]
    )''',
        "description": "P&L impact of +50 bps yield shock"
    },
    "PnL Impact -50bps": {
        "category": "Price Sensitivity",
        "formula": '''PnL Impact -50bps =
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Dn50bps] / 100 * BondPortfolio[MarketValue_INR]
    )''',
        "description": "P&L impact of -50 bps yield shock"
    },
    "PnL Impact +100bps": {
        "category": "Price Sensitivity",
        "formula": '''PnL Impact +100bps =
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Up100bps] / 100 * BondPortfolio[MarketValue_INR]
    )''',
        "description": "P&L impact of +100 bps yield shock"
    },
    "PnL Impact -100bps": {
        "category": "Price Sensitivity",
        "formula": '''PnL Impact -100bps =
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Dn100bps] / 100 * BondPortfolio[MarketValue_INR]
    )''',
        "description": "P&L impact of -100 bps yield shock"
    },
    "PnL Asymmetry 100bps": {
        "category": "Price Sensitivity",
        "formula": '''PnL Asymmetry 100bps =
    ABS([PnL Impact -100bps]) - ABS([PnL Impact +100bps])''',
        "description": "Convexity benefit: asymmetry between up/down shock impact"
    },
    "Duration Approx PnL": {
        "category": "Price Sensitivity",
        "formula": '''Duration Approx PnL =
    VAR ShockBps = SELECTEDVALUE(ShockTable[Shock_bps], 100)
    VAR DeltaY = ShockBps / 10000
    RETURN
    -[Portfolio Modified Duration] * DeltaY * [Total Market Value]
        + 0.5 * [Portfolio Convexity] * POWER(DeltaY, 2) * [Total Market Value]''',
        "description": "Duration-convexity approximation of P&L for selected shock"
    },

    # ── Spread Measures ──
    "Weighted Avg Spread": {
        "category": "Spread Analysis",
        "formula": '''Weighted Avg Spread (bps) =
    SUMX(BondPortfolio,
        BondPortfolio[SpreadOverBenchmark_bps] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Market-value weighted average spread over benchmark"
    },
    "Weighted Avg OAS": {
        "category": "Spread Analysis",
        "formula": '''Weighted Avg OAS =
    SUMX(BondPortfolio,
        BondPortfolio[OAS_bps] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Weighted average option-adjusted spread"
    },
    "Weighted Avg ZSpread": {
        "category": "Spread Analysis",
        "formula": '''Weighted Avg Z-Spread =
    SUMX(BondPortfolio,
        BondPortfolio[ZSpread_bps] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Weighted average Z-spread"
    },

    # ── Yield Measures ──
    "Portfolio YTM": {
        "category": "Yield",
        "formula": '''Portfolio YTM =
    SUMX(BondPortfolio,
        BondPortfolio[YieldToMaturity] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]''',
        "description": "Market-value weighted yield to maturity"
    },
    "Yield per Unit Duration": {
        "category": "Yield",
        "formula": '''Yield per Unit Duration =
    DIVIDE([Portfolio YTM], [Portfolio Modified Duration], 0)''',
        "description": "Carry-to-duration efficiency metric"
    },

    # ── Risk Metrics ──
    "VaR 95% (Historical)": {
        "category": "Risk Metrics",
        "formula": '''VaR 95% Historical =
    -PERCENTILE.INC(MonteCarloScenarios[PnL_Total_INR], 0.05)''',
        "description": "Historical Value at Risk at 95% confidence"
    },
    "CVaR 95% (Expected Shortfall)": {
        "category": "Risk Metrics",
        "formula": '''CVaR 95% =
    -AVERAGEX(
        FILTER(MonteCarloScenarios,
            MonteCarloScenarios[PnL_Total_INR] <= PERCENTILE.INC(MonteCarloScenarios[PnL_Total_INR], 0.05)
        ),
        MonteCarloScenarios[PnL_Total_INR]
    )''',
        "description": "Expected Shortfall (average loss beyond VaR)"
    },
    "Max Loss Scenario": {
        "category": "Risk Metrics",
        "formula": '''Max Loss = MIN(MonteCarloScenarios[PnL_Total_INR])''',
        "description": "Maximum loss across all scenarios"
    },
    "Max Gain Scenario": {
        "category": "Risk Metrics",
        "formula": '''Max Gain = MAX(MonteCarloScenarios[PnL_Total_INR])''',
        "description": "Maximum gain across all scenarios"
    },

    # ── Dynamic Filtering Measures ──
    "Selected Sector Duration": {
        "category": "Dynamic Filters",
        "formula": '''Selected Sector Duration =
    IF(HASONEVALUE(BondPortfolio[Sector]),
        [Portfolio Modified Duration],
        BLANK()
    )''',
        "description": "Duration for single selected sector"
    },
    "Sector Weight": {
        "category": "Dynamic Filters",
        "formula": '''Sector Weight =
    DIVIDE(
        SUM(BondPortfolio[MarketValue_INR]),
        CALCULATE(SUM(BondPortfolio[MarketValue_INR]), ALL(BondPortfolio[Sector])),
        0
    )''',
        "description": "Weight of current sector in total portfolio"
    },
    "Rating Bucket Duration": {
        "category": "Dynamic Filters",
        "formula": '''Rating Bucket Duration =
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration]
        * DIVIDE(BondPortfolio[MarketValue_INR], CALCULATE(SUM(BondPortfolio[MarketValue_INR]), ALLSELECTED(BondPortfolio)), 0)
    )''',
        "description": "Duration contribution for selected credit rating bucket"
    },
}


# ─────────────────────────────────────────────────────────────
# Power BI Data Export
# ─────────────────────────────────────────────────────────────

def export_powerbi_data(bond_df, mc_df):
    """
    Export Power BI-ready CSV files with calculated columns.
    """
    print(" Exporting Power BI data files...")

    # 1. Main bond portfolio with calculated columns
    pbi_bonds = bond_df.copy()
    total_mv = pbi_bonds['MarketValue_INR'].sum()
    pbi_bonds['PortfolioWeight_Calc'] = pbi_bonds['MarketValue_INR'] / total_mv
    pbi_bonds['BondDV01_INR'] = pbi_bonds['ModifiedDuration'] * pbi_bonds['MarketValue_INR'] * 0.0001
    pbi_bonds['Duration_Contribution'] = pbi_bonds['ModifiedDuration'] * pbi_bonds['PortfolioWeight_Calc']
    pbi_bonds['Convexity_Contribution'] = pbi_bonds['Convexity'] * pbi_bonds['PortfolioWeight_Calc']
    pbi_bonds['Coupon_Yield_Spread'] = pbi_bonds['CouponRate'] - pbi_bonds['YieldToMaturity']

    pbi_bonds.to_csv(POWERBI_DIR / "pbi_bond_portfolio.csv", index=False)
    print(f" pbi_bond_portfolio.csv ({len(pbi_bonds)} rows)")

    # 2. Sector summary
    sector_summary = bond_df.groupby('Sector').apply(
        lambda g: pd.Series({
            'Count': len(g),
            'MarketValue': g['MarketValue_INR'].sum(),
            'Weight': g['MarketValue_INR'].sum() / total_mv,
            'Avg_Duration': (g['ModifiedDuration'] * g['MarketValue_INR']).sum() / g['MarketValue_INR'].sum(),
            'Avg_Convexity': (g['Convexity'] * g['MarketValue_INR']).sum() / g['MarketValue_INR'].sum(),
            'Total_DV01': (g['ModifiedDuration'] * g['MarketValue_INR'] * 0.0001).sum(),
            'Avg_YTM': (g['YieldToMaturity'] * g['MarketValue_INR']).sum() / g['MarketValue_INR'].sum(),
        }),
        include_groups=False
    ).reset_index()
    sector_summary.to_csv(POWERBI_DIR / "pbi_sector_summary.csv", index=False)
    print(f" pbi_sector_summary.csv ({len(sector_summary)} rows)")

    # 3. Credit rating summary
    rating_summary = bond_df.groupby('CreditRating').apply(
        lambda g: pd.Series({
            'Count': len(g),
            'MarketValue': g['MarketValue_INR'].sum(),
            'Weight': g['MarketValue_INR'].sum() / total_mv,
            'Avg_Duration': (g['ModifiedDuration'] * g['MarketValue_INR']).sum() / g['MarketValue_INR'].sum(),
            'Avg_Convexity': (g['Convexity'] * g['MarketValue_INR']).sum() / g['MarketValue_INR'].sum(),
        }),
        include_groups=False
    ).reset_index()
    rating_summary.to_csv(POWERBI_DIR / "pbi_rating_summary.csv", index=False)
    print(f" pbi_rating_summary.csv ({len(rating_summary)} rows)")

    # 4. Key rate bucket summary
    if 'KeyRateBucket' in bond_df.columns:
        krd_summary = bond_df.groupby('KeyRateBucket').apply(
            lambda g: pd.Series({
                'Count': len(g),
                'MarketValue': g['MarketValue_INR'].sum(),
                'Avg_Duration': g['ModifiedDuration'].mean(),
                'Total_DV01': (g['ModifiedDuration'] * g['MarketValue_INR'] * 0.0001).sum(),
            }),
            include_groups=False
        ).reset_index()
        krd_summary.to_csv(POWERBI_DIR / "pbi_key_rate_buckets.csv", index=False)
        print(f" pbi_key_rate_buckets.csv ({len(krd_summary)} rows)")

    # 5. Sensitivity shock table
    shocks = [-200, -100, -50, -25, 25, 50, 100, 200]
    weights = bond_df['MarketValue_INR'] / total_mv
    port_dur = (weights * bond_df['ModifiedDuration']).sum()
    port_conv = (weights * bond_df['Convexity']).sum()

    shock_table = []
    for s in shocks:
        dy = s / 10000
        dur_effect = -port_dur * dy * total_mv
        conv_effect = 0.5 * port_conv * dy**2 * total_mv
        shock_table.append({
            'Shock_bps': s, 'Duration_Effect': dur_effect,
            'Convexity_Effect': conv_effect, 'Total_PnL': dur_effect + conv_effect,
            'PnL_Pct': (dur_effect + conv_effect) / total_mv * 100
        })
    pd.DataFrame(shock_table).to_csv(POWERBI_DIR / "pbi_shock_analysis.csv", index=False)
    print(f" pbi_shock_analysis.csv ({len(shock_table)} rows)")

    # 6. Monte Carlo results
    mc_df.to_csv(POWERBI_DIR / "pbi_monte_carlo.csv", index=False)
    print(f" pbi_monte_carlo.csv ({len(mc_df)} rows)")

    return pbi_bonds


# ─────────────────────────────────────────────────────────────
# DAX Documentation Generator
# ─────────────────────────────────────────────────────────────

def generate_dax_documentation():
    """Generate markdown documentation for all DAX measures."""
    doc = "# Power BI DAX Measures Reference\n"
    doc += "# Convexity Sensitivity AI Agent — Duration & Convexity Dashboard\n\n"
    doc += "---\n\n"
    doc += f"**Total Measures: {len(DAX_MEASURES)}**\n\n"

    # Group by category
    categories = {}
    for name, info in DAX_MEASURES.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    # Table of contents
    doc += "## Table of Contents\n\n"
    for cat in categories:
        anchor = cat.lower().replace(' ', '-').replace('/', '-')
        doc += f"- [{cat}](#{anchor}) ({len(categories[cat])} measures)\n"
    doc += "\n---\n\n"

    # Measures by category
    for cat, measures in categories.items():
        doc += f"## {cat}\n\n"
        for name, info in measures:
            doc += f"### {name}\n\n"
            doc += f"**Description:** {info['description']}\n\n"
            doc += f"```dax\n{info['formula']}\n```\n\n"
        doc += "---\n\n"

    # Usage notes
    doc += "## Power BI Setup Instructions\n\n"
    doc += "### Data Import\n\n"
    doc += "1. Open Power BI Desktop\n"
    doc += "2. Click **Get Data** → **Text/CSV**\n"
    doc += "3. Import the following files from `outputs/powerbi_exports/`:\n"
    doc += " - `pbi_bond_portfolio.csv` — Main bond data table\n"
    doc += " - `pbi_sector_summary.csv` — Sector aggregations\n"
    doc += " - `pbi_rating_summary.csv` — Rating aggregations\n"
    doc += " - `pbi_key_rate_buckets.csv` — Key rate duration buckets\n"
    doc += " - `pbi_shock_analysis.csv` — Sensitivity shock table\n"
    doc += " - `pbi_monte_carlo.csv` — Monte Carlo scenario data\n\n"
    doc += "### Creating Measures\n\n"
    doc += "1. In Power BI Model view, select the `BondPortfolio` table\n"
    doc += "2. Click **New Measure** on the ribbon\n"
    doc += "3. Copy-paste the DAX formulas above\n"
    doc += "4. Rename the table references if your table names differ\n\n"
    doc += "### Recommended Visuals\n\n"
    doc += "| Visual Type | Measure(s) | Dimension |\n"
    doc += "|------------|------------|----------|\n"
    doc += "| Card | Portfolio Duration, Convexity, DV01 | — |\n"
    doc += "| Bar Chart | Duration Contribution | Sector |\n"
    doc += "| Heatmap | Avg Convexity | Sector × Rating |\n"
    doc += "| Line Chart | Yield Curve | Tenor |\n"
    doc += "| Waterfall | PnL Impact | Shock Scenarios |\n"
    doc += "| Histogram | P&L Distribution | Monte Carlo |\n"
    doc += "| KPI | VaR 95%, CVaR 95% | — |\n"
    doc += "| Treemap | Market Value | Sector → Issuer |\n\n"

    return doc


# ─────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────

def run_part5():
    """Execute Part 5: Power BI DAX Measures & Data Export."""
    print_section_header("PART 5: Power BI DAX Measures & Dashboard Data Export")

    # 1. Load data
    print(" Loading data...")
    bond_df = load_bond_portfolio()
    mc_df = load_monte_carlo()

    # 2. List DAX measures
    print_subsection(f"DAX Measures ({len(DAX_MEASURES)} total)")
    categories = {}
    for name, info in DAX_MEASURES.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1

    for cat, count in categories.items():
        print(f" {cat:.<40} {count} measures")

    # 3. Print sample DAX formulas
    print_subsection("Sample DAX Formulas")
    sample_measures = ['Portfolio Modified Duration', 'Portfolio Convexity', 'Portfolio DV01', 'Duration Approx PnL']
    for name in sample_measures:
        info = DAX_MEASURES[name]
        print(f"\n {name}")
        print(f" {info['description']}")
        print(f" Formula:\n {info['formula']}")

    # 4. Export Power BI data
    print_subsection("Exporting Power BI-Ready Data Files")
    export_powerbi_data(bond_df, mc_df)

    # 5. Generate DAX documentation
    print_subsection("Generating DAX Documentation")
    dax_doc = generate_dax_documentation()

    from src.utils import PROJECT_ROOT
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    doc_path = docs_dir / "DAX_measures.md"
    with open(doc_path, 'w') as f:
        f.write(dax_doc)
    print(f" DAX documentation saved: {doc_path}")

    print_section_header("PART 5 COMPLETE ")
    return DAX_MEASURES


if __name__ == "__main__":
    run_part5()
