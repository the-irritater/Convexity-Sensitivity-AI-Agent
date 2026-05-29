# Power BI DAX Measures Reference
# Convexity Sensitivity AI Agent — Duration & Convexity Dashboard

---

**Total Measures: 31**

## Table of Contents

- [Portfolio Aggregation](#portfolio-aggregation) (3 measures)
- [Duration](#duration) (4 measures)
- [Convexity](#convexity) (3 measures)
- [DV01 / Sensitivity](#dv01---sensitivity) (3 measures)
- [Price Sensitivity](#price-sensitivity) (6 measures)
- [Spread Analysis](#spread-analysis) (3 measures)
- [Yield](#yield) (2 measures)
- [Risk Metrics](#risk-metrics) (4 measures)
- [Dynamic Filters](#dynamic-filters) (3 measures)

---

## Portfolio Aggregation

### Total Market Value

**Description:** Sum of all bond market values in INR

```dax
Total Market Value = SUM(BondPortfolio[MarketValue_INR])
```

### Bond Count

**Description:** Total number of bonds in portfolio

```dax
Bond Count = COUNTROWS(BondPortfolio)
```

### Average Coupon Rate

**Description:** Market-value weighted average coupon rate

```dax
Average Coupon Rate = 
    SUMX(BondPortfolio, 
        BondPortfolio[CouponRate] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

---

## Duration

### Portfolio Modified Duration

**Description:** Market-value weighted portfolio modified duration

```dax
Portfolio Modified Duration = 
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

### Portfolio Macaulay Duration

**Description:** Market-value weighted portfolio Macaulay duration

```dax
Portfolio Macaulay Duration = 
    SUMX(BondPortfolio,
        BondPortfolio[MacaulayDuration] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

### Duration Contribution

**Description:** Duration contribution of current filter context (sector, rating, etc.)

```dax
Duration Contribution = 
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR]
        / CALCULATE([Total Market Value], ALL(BondPortfolio))
    )
```

### Effective Duration

**Description:** Weighted average effective duration (for bonds with optionality)

```dax
Effective Duration = 
    SUMX(BondPortfolio,
        BondPortfolio[EffectiveDuration] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

---

## Convexity

### Portfolio Convexity

**Description:** Market-value weighted portfolio convexity

```dax
Portfolio Convexity = 
    SUMX(BondPortfolio,
        BondPortfolio[Convexity] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

### Convexity Contribution

**Description:** Convexity contribution of filtered subset

```dax
Convexity Contribution = 
    SUMX(BondPortfolio,
        BondPortfolio[Convexity] * BondPortfolio[MarketValue_INR]
        / CALCULATE([Total Market Value], ALL(BondPortfolio))
    )
```

### Effective Convexity

**Description:** Weighted average effective convexity

```dax
Effective Convexity = 
    SUMX(BondPortfolio,
        BondPortfolio[EffectiveConvexity] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

---

## DV01 / Sensitivity

### Portfolio DV01

**Description:** Total portfolio DV01 in INR (price change per 1bp yield move)

```dax
Portfolio DV01 = 
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR] * 0.0001
    )
```

### DV01 per Crore

**Description:** DV01 normalized per crore of market value

```dax
DV01 per Crore = 
    [Portfolio DV01] / ([Total Market Value] / 10000000)
```

### DV01 Contribution Pct

**Description:** DV01 contribution as percentage of total

```dax
DV01 Contribution % = 
    DIVIDE(
        SUMX(BondPortfolio, BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR] * 0.0001),
        CALCULATE(
            SUMX(BondPortfolio, BondPortfolio[ModifiedDuration] * BondPortfolio[MarketValue_INR] * 0.0001),
            ALL(BondPortfolio)
        ),
        0
    )
```

---

## Price Sensitivity

### PnL Impact +50bps

**Description:** P&L impact of +50 bps yield shock

```dax
PnL Impact +50bps = 
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Up50bps] / 100 * BondPortfolio[MarketValue_INR]
    )
```

### PnL Impact -50bps

**Description:** P&L impact of -50 bps yield shock

```dax
PnL Impact -50bps = 
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Dn50bps] / 100 * BondPortfolio[MarketValue_INR]
    )
```

### PnL Impact +100bps

**Description:** P&L impact of +100 bps yield shock

```dax
PnL Impact +100bps = 
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Up100bps] / 100 * BondPortfolio[MarketValue_INR]
    )
```

### PnL Impact -100bps

**Description:** P&L impact of -100 bps yield shock

```dax
PnL Impact -100bps = 
    SUMX(BondPortfolio,
        BondPortfolio[PriceChange_Dn100bps] / 100 * BondPortfolio[MarketValue_INR]
    )
```

### PnL Asymmetry 100bps

**Description:** Convexity benefit: asymmetry between up/down shock impact

```dax
PnL Asymmetry 100bps = 
    ABS([PnL Impact -100bps]) - ABS([PnL Impact +100bps])
```

### Duration Approx PnL

**Description:** Duration-convexity approximation of P&L for selected shock

```dax
Duration Approx PnL = 
    VAR ShockBps = SELECTEDVALUE(ShockTable[Shock_bps], 100)
    VAR DeltaY = ShockBps / 10000
    RETURN
    -[Portfolio Modified Duration] * DeltaY * [Total Market Value]
        + 0.5 * [Portfolio Convexity] * POWER(DeltaY, 2) * [Total Market Value]
```

---

## Spread Analysis

### Weighted Avg Spread

**Description:** Market-value weighted average spread over benchmark

```dax
Weighted Avg Spread (bps) = 
    SUMX(BondPortfolio,
        BondPortfolio[SpreadOverBenchmark_bps] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

### Weighted Avg OAS

**Description:** Weighted average option-adjusted spread

```dax
Weighted Avg OAS = 
    SUMX(BondPortfolio,
        BondPortfolio[OAS_bps] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

### Weighted Avg ZSpread

**Description:** Weighted average Z-spread

```dax
Weighted Avg Z-Spread = 
    SUMX(BondPortfolio,
        BondPortfolio[ZSpread_bps] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

---

## Yield

### Portfolio YTM

**Description:** Market-value weighted yield to maturity

```dax
Portfolio YTM = 
    SUMX(BondPortfolio,
        BondPortfolio[YieldToMaturity] * BondPortfolio[MarketValue_INR]
    ) / [Total Market Value]
```

### Yield per Unit Duration

**Description:** Carry-to-duration efficiency metric

```dax
Yield per Unit Duration = 
    DIVIDE([Portfolio YTM], [Portfolio Modified Duration], 0)
```

---

## Risk Metrics

### VaR 95% (Historical)

**Description:** Historical Value at Risk at 95% confidence

```dax
VaR 95% Historical = 
    -PERCENTILE.INC(MonteCarloScenarios[PnL_Total_INR], 0.05)
```

### CVaR 95% (Expected Shortfall)

**Description:** Expected Shortfall (average loss beyond VaR)

```dax
CVaR 95% = 
    -AVERAGEX(
        FILTER(MonteCarloScenarios,
            MonteCarloScenarios[PnL_Total_INR] <= PERCENTILE.INC(MonteCarloScenarios[PnL_Total_INR], 0.05)
        ),
        MonteCarloScenarios[PnL_Total_INR]
    )
```

### Max Loss Scenario

**Description:** Maximum loss across all scenarios

```dax
Max Loss = MIN(MonteCarloScenarios[PnL_Total_INR])
```

### Max Gain Scenario

**Description:** Maximum gain across all scenarios

```dax
Max Gain = MAX(MonteCarloScenarios[PnL_Total_INR])
```

---

## Dynamic Filters

### Selected Sector Duration

**Description:** Duration for single selected sector

```dax
Selected Sector Duration = 
    IF(HASONEVALUE(BondPortfolio[Sector]),
        [Portfolio Modified Duration],
        BLANK()
    )
```

### Sector Weight

**Description:** Weight of current sector in total portfolio

```dax
Sector Weight = 
    DIVIDE(
        SUM(BondPortfolio[MarketValue_INR]),
        CALCULATE(SUM(BondPortfolio[MarketValue_INR]), ALL(BondPortfolio[Sector])),
        0
    )
```

### Rating Bucket Duration

**Description:** Duration contribution for selected credit rating bucket

```dax
Rating Bucket Duration = 
    SUMX(BondPortfolio,
        BondPortfolio[ModifiedDuration] 
        * DIVIDE(BondPortfolio[MarketValue_INR], CALCULATE(SUM(BondPortfolio[MarketValue_INR]), ALLSELECTED(BondPortfolio)), 0)
    )
```

---

## Power BI Setup Instructions

### Data Import

1. Open Power BI Desktop
2. Click **Get Data** → **Text/CSV**
3. Import the following files from `outputs/powerbi_exports/`:
   - `pbi_bond_portfolio.csv` — Main bond data table
   - `pbi_sector_summary.csv` — Sector aggregations
   - `pbi_rating_summary.csv` — Rating aggregations
   - `pbi_key_rate_buckets.csv` — Key rate duration buckets
   - `pbi_shock_analysis.csv` — Sensitivity shock table
   - `pbi_monte_carlo.csv` — Monte Carlo scenario data

### Creating Measures

1. In Power BI Model view, select the `BondPortfolio` table
2. Click **New Measure** on the ribbon
3. Copy-paste the DAX formulas above
4. Rename the table references if your table names differ

### Recommended Visuals

| Visual Type | Measure(s) | Dimension |
|------------|------------|----------|
| Card | Portfolio Duration, Convexity, DV01 | — |
| Bar Chart | Duration Contribution | Sector |
| Heatmap | Avg Convexity | Sector × Rating |
| Line Chart | Yield Curve | Tenor |
| Waterfall | PnL Impact | Shock Scenarios |
| Histogram | P&L Distribution | Monte Carlo |
| KPI | VaR 95%, CVaR 95% | — |
| Treemap | Market Value | Sector → Issuer |

