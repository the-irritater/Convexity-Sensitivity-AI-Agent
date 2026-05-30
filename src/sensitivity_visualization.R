# ============================================================
# Sensitivity Visualization — R Implementation
# ============================================================
# ggplot2 sensitivity charts: dual-axis duration/convexity,
# VaR histogram, and rate shock analysis.
# Required packages: readr, dplyr, ggplot2, scales, tidyr
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

# ---- Configuration ----
DATA_DIR <- dirname(sys.frame(1)$ofile)
if (is.null(DATA_DIR)) DATA_DIR <- getwd()
PARENT_DIR <- dirname(DATA_DIR)
OUTPUT_DIR <- file.path(PARENT_DIR, "outputs", "figures")
REPORTS_DIR <- file.path(PARENT_DIR, "outputs", "reports")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(REPORTS_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Sensitivity Visualization (R) ===\n\n")

# ---- Load Data ----
bonds <- read_csv(file.path(PARENT_DIR, "data", "bond_portfolio_data.csv"),
                  show_col_types = FALSE)
mc <- read_csv(file.path(PARENT_DIR, "data", "monte_carlo_scenarios.csv"),
               show_col_types = FALSE)

cat(sprintf("  Loaded %d bonds, %d MC scenarios\n", nrow(bonds), nrow(mc)))

# ---- Portfolio Metrics ----
total_mv <- sum(bonds$MarketValue_INR, na.rm = TRUE)
bonds$Weight <- bonds$MarketValue_INR / total_mv
port_duration <- sum(bonds$Weight * bonds$ModifiedDuration, na.rm = TRUE)
port_convexity <- sum(bonds$Weight * bonds$Convexity, na.rm = TRUE)

cat(sprintf("  Portfolio Duration: %.4f\n", port_duration))
cat(sprintf("  Portfolio Convexity: %.4f\n", port_convexity))

# ── Chart 1: Duration & Convexity Sensitivity to Interest Rate Changes ──
cat("\n--- Chart 1: Sensitivity Chart (Dual Axis) ---\n")

rate_changes <- seq(-0.02, 0.02, by = 0.002)

sensitivity_data <- data.frame(
  RateChange = rate_changes,
  AdjDuration = port_duration * (1 - rate_changes * port_duration),
  AdjConvexity = port_convexity * (1 - 2 * rate_changes * port_duration)
)

p1 <- ggplot(sensitivity_data, aes(x = RateChange)) +
  geom_line(aes(y = AdjDuration, color = "Duration"), linewidth = 1.2) +
  geom_line(aes(y = AdjConvexity / 10, color = "Convexity/10"), linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  scale_x_continuous(labels = percent) +
  scale_color_manual(values = c("Duration" = "#1976D2", "Convexity/10" = "#E64A19")) +
  labs(
    title = "Duration & Convexity Sensitivity to Interest Rate Changes",
    x = "Yield Change",
    y = "Adjusted Metric",
    color = "Risk Measure"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    legend.position = "bottom"
  )

ggsave(file.path(OUTPUT_DIR, "r_sensitivity_chart.png"), p1,
       width = 10, height = 6, dpi = 150)
cat("  ✅ Saved: r_sensitivity_chart.png\n")

# ── Chart 2: P&L Distribution with VaR (Histogram) ──
cat("\n--- Chart 2: VaR Histogram ---\n")

pnl_lakhs <- mc$PnL_Total_INR / 1e5
var_95 <- -quantile(mc$PnL_Total_INR, 0.05) / 1e5
var_99 <- -quantile(mc$PnL_Total_INR, 0.01) / 1e5

p2 <- ggplot(data.frame(PnL = pnl_lakhs), aes(x = PnL)) +
  geom_histogram(bins = 50, fill = "#2196F3", alpha = 0.7, color = "white") +
  geom_vline(xintercept = -var_95, color = "#FF9800", linetype = "dashed",
             linewidth = 1.2) +
  geom_vline(xintercept = -var_99, color = "#F44336", linetype = "dashed",
             linewidth = 1.2) +
  annotate("text", x = -var_95 - 5, y = Inf, vjust = 2,
           label = sprintf("VaR 95%%\n₹%.1fL", var_95), color = "#FF9800", size = 3.5) +
  annotate("text", x = -var_99 - 5, y = Inf, vjust = 4,
           label = sprintf("VaR 99%%\n₹%.1fL", var_99), color = "#F44336", size = 3.5) +
  labs(
    title = "Portfolio P&L Distribution with VaR",
    x = "P&L (₹ Lakhs)",
    y = "Frequency"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16))

ggsave(file.path(OUTPUT_DIR, "r_var_histogram.png"), p2,
       width = 10, height = 6, dpi = 150)
cat("  ✅ Saved: r_var_histogram.png\n")

# ── Chart 3: Yield Shock P&L Profile ──
cat("\n--- Chart 3: Yield Shock P&L Profile ---\n")

shocks_bps <- c(-300, -200, -150, -100, -50, -25, 25, 50, 100, 150, 200, 300)

shock_results <- data.frame(
  Shock = shocks_bps,
  Duration_Only = sapply(shocks_bps, function(s) {
    -port_duration * (s / 10000) * total_mv / 1e5
  }),
  Duration_Convexity = sapply(shocks_bps, function(s) {
    dy <- s / 10000
    (-port_duration * dy + 0.5 * port_convexity * dy^2) * total_mv / 1e5
  })
)

shock_long <- shock_results %>%
  pivot_longer(cols = c(Duration_Only, Duration_Convexity),
               names_to = "Method", values_to = "PnL_Lakhs")

p3 <- ggplot(shock_long, aes(x = Shock, y = PnL_Lakhs, color = Method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "solid", color = "gray40") +
  scale_color_manual(
    values = c("Duration_Only" = "#1976D2", "Duration_Convexity" = "#F44336"),
    labels = c("Duration Only", "Duration + Convexity")
  ) +
  labs(
    title = "Portfolio P&L: Duration vs Duration+Convexity",
    x = "Yield Shock (bps)",
    y = "P&L (₹ Lakhs)",
    color = "Method"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    legend.position = "bottom"
  )

ggsave(file.path(OUTPUT_DIR, "r_shock_profile.png"), p3,
       width = 10, height = 6, dpi = 150)
cat("  ✅ Saved: r_shock_profile.png\n")

# ── Chart 4: Duration Distribution by Sector ──
cat("\n--- Chart 4: Duration by Sector ---\n")

p4 <- ggplot(bonds, aes(x = reorder(Sector, ModifiedDuration, FUN = median),
                         y = ModifiedDuration, fill = Sector)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  coord_flip() +
  labs(
    title = "Modified Duration Distribution by Sector",
    x = "",
    y = "Modified Duration (years)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    legend.position = "none"
  )

ggsave(file.path(OUTPUT_DIR, "r_duration_by_sector.png"), p4,
       width = 10, height = 6, dpi = 150)
cat("  ✅ Saved: r_duration_by_sector.png\n")

# ---- Export Sensitivity Results to CSV ----
cat("\n--- Exporting Results ---\n")

# Sensitivity results table
sens_export <- data.frame(
  RateChange_Pct = sensitivity_data$RateChange * 100,
  AdjustedDuration = sensitivity_data$AdjDuration,
  AdjustedConvexity = sensitivity_data$AdjConvexity,
  PnL_Duration_Only_Lakhs = -port_duration * sensitivity_data$RateChange * total_mv / 1e5,
  PnL_DurConv_Lakhs = (-port_duration * sensitivity_data$RateChange +
    0.5 * port_convexity * sensitivity_data$RateChange^2) * total_mv / 1e5
)

write_csv(sens_export, file.path(REPORTS_DIR, "sensitivity_results.csv"))
cat(sprintf("  ✅ Saved: sensitivity_results.csv (%d rows)\n", nrow(sens_export)))

# Shock analysis export
write_csv(shock_results, file.path(REPORTS_DIR, "r_shock_analysis.csv"))
cat(sprintf("  ✅ Saved: r_shock_analysis.csv (%d rows)\n", nrow(shock_results)))

cat("\n=== Sensitivity Visualization (R) COMPLETE ===\n")
