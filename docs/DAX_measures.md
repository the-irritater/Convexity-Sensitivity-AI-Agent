# Power BI DAX Measures Reference

## Portfolio Modified Duration
```dax
SUMX(Bonds, Bonds[ModifiedDuration]*Bonds[MarketValue_INR]) / SUM(Bonds[MarketValue_INR])
```

## Portfolio Convexity
```dax
SUMX(Bonds, Bonds[Convexity]*Bonds[MarketValue_INR]) / SUM(Bonds[MarketValue_INR])
```

## Portfolio DV01
```dax
SUMX(Bonds, Bonds[ModifiedDuration]*Bonds[MarketValue_INR]*0.0001)
```

## Duration Contribution
```dax
SUMX(Bonds, Bonds[ModifiedDuration]*Bonds[MarketValue_INR] / CALCULATE(SUM(Bonds[MarketValue_INR]),ALL(Bonds)))
```

## Convexity Contribution
```dax
SUMX(Bonds, Bonds[Convexity]*Bonds[MarketValue_INR] / CALCULATE(SUM(Bonds[MarketValue_INR]),ALL(Bonds)))
```

## PnL +100bps
```dax
SUMX(Bonds, Bonds[PriceChange_Up100bps]/100*Bonds[MarketValue_INR])
```

## PnL -100bps
```dax
SUMX(Bonds, Bonds[PriceChange_Dn100bps]/100*Bonds[MarketValue_INR])
```

## VaR 95%
```dax
-PERCENTILE.INC(MC[PnL_Total_INR], 0.05)
```

## CVaR 95%
```dax
-AVERAGEX(FILTER(MC, MC[PnL_Total_INR]<=PERCENTILE.INC(MC[PnL_Total_INR],0.05)), MC[PnL_Total_INR])
```

## Portfolio YTM
```dax
SUMX(Bonds, Bonds[YieldToMaturity]*Bonds[MarketValue_INR]) / SUM(Bonds[MarketValue_INR])
```

## Sector Weight
```dax
DIVIDE(SUM(Bonds[MarketValue_INR]), CALCULATE(SUM(Bonds[MarketValue_INR]),ALL(Bonds[Sector])))
```

## Weighted Spread
```dax
SUMX(Bonds, Bonds[SpreadOverBenchmark_bps]*Bonds[MarketValue_INR]) / SUM(Bonds[MarketValue_INR])
```

