# ============================================================
# Part 2 (R Implementation): Yield Curve Modelling & DV01
# ============================================================
# R implementation of Nelson-Siegel model and DV01 calculations
# NOTE: Requires R with packages: readr, ggplot2, dplyr, tidyr
# Install packages: install.packages(c("readr","ggplot2","dplyr","tidyr"))
# ============================================================

library(readr)
library(dplyr)
library(tidyr)

# ---- Configuration ----
DATA_DIR <- dirname(sys.frame(1)$ofile)
if (is.null(DATA_DIR)) DATA_DIR <- getwd()
PARENT_DIR <- dirname(DATA_DIR)

cat("=== Part 2 (R): Yield Curve Modelling & DV01 ===\n\n")

# ---- Load Data ----
cat("Loading data...\n")
bonds <- read_csv(file.path(PARENT_DIR, "data", "bond_portfolio_data.csv"), show_col_types = FALSE)
yc <- read_csv(file.path(PARENT_DIR, "data", "yield_curve_history.csv"), show_col_types = FALSE)

cat(sprintf(" Bonds: %d records\n", nrow(bonds)))
cat(sprintf(" Yield Curve: %d records\n", nrow(yc)))

# ---- Nelson-Siegel Model ----
# y(t) = beta0 + beta1 * [(1-exp(-t/tau))/(t/tau)]
# + beta2 * [(1-exp(-t/tau))/(t/tau) - exp(-t/tau)]

nelson_siegel <- function(t, beta0, beta1, beta2, tau) {
  t <- pmax(t, 1e-6)
  x <- t / tau
  factor1 <- (1 - exp(-x)) / x
  factor2 <- factor1 - exp(-x)
  return(beta0 + beta1 * factor1 + beta2 * factor2)
}

fit_nelson_siegel <- function(tenors, yields) {
  objective <- function(params) {
    beta0 <- params[1]
    beta1 <- params[2]
    beta2 <- params[3]
    tau <- params[4]
    if (tau <= 0.01) return(1e10)
    predicted <- nelson_siegel(tenors, beta0, beta1, beta2, tau)
    return(sum((predicted - yields)^2))
  }

  # Initial guess
  x0 <- c(tail(yields, 1), yields[1] - tail(yields, 1), 0, 2)

  result <- optim(x0, objective, method = "L-BFGS-B",
                  lower = c(0, -0.2, -0.2, 0.1),
                  upper = c(0.2, 0.2, 0.2, 30))

  return(list(
    beta0 = result$par[1],
    beta1 = result$par[2],
    beta2 = result$par[3],
    tau = result$par[4],
    rmse = sqrt(mean((nelson_siegel(tenors, result$par[1], result$par[2],
                                      result$par[3], result$par[4]) - yields)^2))
  ))
}

# ---- Fit to Latest Yield Curve ----
cat("\n--- Nelson-Siegel Fitting ---\n")

latest_date <- max(yc$CurveDate)
latest_curve <- yc %>%
  filter(CurveDate == latest_date) %>%
  arrange(Tenor_Years)

tenors <- latest_curve$Tenor_Years
yields <- latest_curve$Yield

cat(sprintf(" Latest curve date: %s\n", latest_date))

ns_fit <- fit_nelson_siegel(tenors, yields)

cat(sprintf(" beta0 (Level): %.6f\n", ns_fit$beta0))
cat(sprintf(" beta1 (Slope): %.6f\n", ns_fit$beta1))
cat(sprintf(" beta2 (Curvature): %.6f\n", ns_fit$beta2))
cat(sprintf(" tau (Decay): %.6f\n", ns_fit$tau))
cat(sprintf(" RMSE: %.8f\n", ns_fit$rmse))

# ---- DV01 Calculations ----
cat("\n--- DV01 Sensitivity Analysis ---\n")

bonds <- bonds %>%
  mutate(
    BondDV01 = ModifiedDuration * MarketValue_INR * 0.0001
  )

# DV01 by Key Rate Bucket
dv01_ladder <- bonds %>%
  group_by(KeyRateBucket) %>%
  summarise(
    Count = n(),
    Total_DV01 = sum(BondDV01, na.rm = TRUE),
    Avg_Duration = mean(ModifiedDuration, na.rm = TRUE),
    Total_MV = sum(MarketValue_INR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(factor(KeyRateBucket, levels = c("0-2Y","2-3Y","3-5Y","5-7Y","7-10Y","10-15Y","15-20Y","20Y+")))

print(dv01_ladder)
cat(sprintf("\n Total Portfolio DV01: %.2f INR\n", sum(dv01_ladder$Total_DV01)))

# ---- Portfolio Summary ----
cat("\n--- Portfolio Duration/Convexity Summary ---\n")

total_mv <- sum(bonds$MarketValue_INR, na.rm = TRUE)
bonds$Weight <- bonds$MarketValue_INR / total_mv

port_duration <- sum(bonds$Weight * bonds$ModifiedDuration, na.rm = TRUE)
port_convexity <- sum(bonds$Weight * bonds$Convexity, na.rm = TRUE)
port_ytm <- sum(bonds$Weight * bonds$YieldToMaturity, na.rm = TRUE)

cat(sprintf(" Portfolio Modified Duration: %.4f\n", port_duration))
cat(sprintf(" Portfolio Convexity: %.4f\n", port_convexity))
cat(sprintf(" Portfolio YTM: %.4f%%\n", port_ytm * 100))
cat(sprintf(" Total Market Value: %.2f INR\n", total_mv))

# ---- Sensitivity Analysis ----
cat("\n--- Price Sensitivity to Yield Shocks ---\n")

shocks_bps <- c(-200, -100, -50, -25, 25, 50, 100, 200)

for (shock in shocks_bps) {
  dy <- shock / 10000
  pct_change <- -port_duration * dy + 0.5 * port_convexity * dy^2
  pnl <- pct_change * total_mv
  cat(sprintf(" Shock %+4d bps: PnL = %+.2f INR (%+.4f%%)\n",
              shock, pnl, pct_change * 100))
}

# ---- Save Results ----
output_dir <- file.path(PARENT_DIR, "outputs", "reports")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write_csv(dv01_ladder, file.path(output_dir, "part2_r_dv01_ladder.csv"))
cat(sprintf("\n Results saved to %s\n", output_dir))

cat("\n=== Part 2 (R) COMPLETE ===\n")
