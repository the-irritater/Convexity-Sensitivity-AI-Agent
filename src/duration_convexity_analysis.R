# ============================================================
# Duration & Convexity Prediction — R Implementation
# ============================================================
# GBM, XGBoost, and Neural Network (H2O) models for predicting
# duration and convexity from bond characteristics.
# Required packages: caret, xgboost, Metrics, readr, dplyr
# Optional: h2o (requires Java 8+)
# ============================================================

library(readr)
library(dplyr)

# ---- Configuration ----
DATA_DIR <- dirname(sys.frame(1)$ofile)
if (is.null(DATA_DIR)) DATA_DIR <- getwd()
PARENT_DIR <- dirname(DATA_DIR)
OUTPUT_DIR <- file.path(PARENT_DIR, "outputs", "reports")
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

cat("=== Duration & Convexity ML Analysis (R) ===\n\n")

# ---- Load Data ----
cat("Loading data...\n")
bonds <- read_csv(file.path(PARENT_DIR, "data", "bond_portfolio_data.csv"),
                  show_col_types = FALSE)
cat(sprintf("  Loaded %d bonds with %d features\n", nrow(bonds), ncol(bonds)))

# ---- Feature Engineering ----
cat("\n--- Feature Engineering ---\n")

# Create derived features
bonds <- bonds %>%
  mutate(
    Coupon_Yield_Spread = CouponRate - YieldToMaturity,
    Duration_Maturity_Ratio = ModifiedDuration / pmax(YearsToMaturity, 0.1),
    Coupon_per_Period = CouponRate / pmax(CouponFrequency, 1),
    Price_Par_Diff = CleanPrice - FaceValue,
    Duration_Squared = ModifiedDuration^2,
    Maturity_Squared = YearsToMaturity^2,
    YTM_Duration_Interaction = YieldToMaturity * ModifiedDuration,
    Coupon_Duration_Ratio = CouponRate / pmax(ModifiedDuration, 0.01),
    Log_Maturity = log1p(YearsToMaturity),
    Spread_Duration_Product = SpreadOverBenchmark_bps * ModifiedDuration
  )

# Select features for modelling
feature_cols <- c(
  "CouponRate", "YearsToMaturity", "YieldToMaturity",
  "MacaulayDuration", "ModifiedDuration", "FaceValue",
  "CleanPrice", "DirtyPrice", "AccruedInterest",
  "DV01_Per100Face", "SpreadOverBenchmark_bps",
  "CouponFrequency",
  "Coupon_Yield_Spread", "Duration_Maturity_Ratio",
  "Duration_Squared", "Maturity_Squared",
  "YTM_Duration_Interaction", "Log_Maturity"
)

# Filter available features
available_features <- feature_cols[feature_cols %in% names(bonds)]
cat(sprintf("  Features available: %d\n", length(available_features)))

# Prepare data
model_data <- bonds %>%
  select(all_of(available_features), Convexity) %>%
  na.omit()

cat(sprintf("  Training samples: %d\n", nrow(model_data)))

# ---- Train/Test Split ----
set.seed(42)
train_idx <- sample(1:nrow(model_data), size = 0.8 * nrow(model_data))
train_data <- model_data[train_idx, ]
test_data <- model_data[-train_idx, ]

cat(sprintf("  Train: %d, Test: %d\n", nrow(train_data), nrow(test_data)))

# ---- Model 1: GBM via caret ----
cat("\n--- Model 1: Gradient Boosted Machine (GBM) ---\n")

tryCatch({
  library(caret)
  library(gbm)

  ctrl <- trainControl(method = "cv", number = 5)

  gbm_model <- train(
    Convexity ~ .,
    data = train_data,
    method = "gbm",
    trControl = ctrl,
    verbose = FALSE,
    tuneGrid = expand.grid(
      n.trees = c(100, 200),
      interaction.depth = c(3, 5),
      shrinkage = c(0.1),
      n.minobsinnode = c(10)
    )
  )

  gbm_pred <- predict(gbm_model, test_data)
  gbm_rmse <- sqrt(mean((test_data$Convexity - gbm_pred)^2))
  gbm_r2 <- 1 - sum((test_data$Convexity - gbm_pred)^2) /
                 sum((test_data$Convexity - mean(test_data$Convexity))^2)

  cat(sprintf("  Best params: trees=%d, depth=%d\n",
              gbm_model$bestTune$n.trees, gbm_model$bestTune$interaction.depth))
  cat(sprintf("  RMSE: %.4f\n", gbm_rmse))
  cat(sprintf("  R²:   %.6f\n", gbm_r2))
}, error = function(e) {
  cat(sprintf("  ⚠️ GBM training failed: %s\n", e$message))
  cat("  Install with: install.packages(c('caret', 'gbm'))\n")
  gbm_rmse <<- NA
  gbm_r2 <<- NA
})

# ---- Model 2: XGBoost ----
cat("\n--- Model 2: XGBoost ---\n")

tryCatch({
  library(xgboost)

  X_train <- as.matrix(train_data %>% select(-Convexity))
  y_train <- train_data$Convexity
  X_test <- as.matrix(test_data %>% select(-Convexity))
  y_test <- test_data$Convexity

  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dtest <- xgb.DMatrix(X_test, label = y_test)

  xgb_params <- list(
    objective = "reg:squarederror",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    nthread = 4
  )

  xgb_model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 200,
    watchlist = list(test = dtest),
    verbose = 0,
    early_stopping_rounds = 20
  )

  xgb_pred <- predict(xgb_model, dtest)
  xgb_rmse <- sqrt(mean((y_test - xgb_pred)^2))
  xgb_r2 <- 1 - sum((y_test - xgb_pred)^2) / sum((y_test - mean(y_test))^2)

  cat(sprintf("  Best iteration: %d\n", xgb_model$best_iteration))
  cat(sprintf("  RMSE: %.4f\n", xgb_rmse))
  cat(sprintf("  R²:   %.6f\n", xgb_r2))

  # Feature importance
  cat("\n  Top 10 Features:\n")
  imp <- xgb.importance(model = xgb_model)
  print(head(imp, 10))
}, error = function(e) {
  cat(sprintf("  ⚠️ XGBoost training failed: %s\n", e$message))
  cat("  Install with: install.packages('xgboost')\n")
  xgb_rmse <<- NA
  xgb_r2 <<- NA
})

# ---- Model 3: H2O Deep Learning (Optional) ----
cat("\n--- Model 3: H2O Deep Learning (if available) ---\n")

h2o_rmse <- NA
h2o_r2 <- NA

tryCatch({
  library(h2o)
  h2o.init(nthreads = -1, max_mem_size = "2g")

  h2o_train <- as.h2o(train_data)
  h2o_test <- as.h2o(test_data)

  h2o_model <- h2o.deeplearning(
    x = available_features,
    y = "Convexity",
    training_frame = h2o_train,
    hidden = c(128, 64, 32),
    epochs = 100,
    activation = "RectifierWithDropout",
    hidden_dropout_ratios = c(0.3, 0.2, 0.1),
    l2 = 1e-4,
    stopping_metric = "RMSE",
    stopping_rounds = 10,
    seed = 42
  )

  h2o_pred <- as.data.frame(h2o.predict(h2o_model, h2o_test))$predict
  h2o_rmse <- sqrt(mean((test_data$Convexity - h2o_pred)^2))
  h2o_r2 <- 1 - sum((test_data$Convexity - h2o_pred)^2) /
                 sum((test_data$Convexity - mean(test_data$Convexity))^2)

  cat(sprintf("  RMSE: %.4f\n", h2o_rmse))
  cat(sprintf("  R²:   %.6f\n", h2o_r2))

  h2o.shutdown(prompt = FALSE)
}, error = function(e) {
  cat(sprintf("  ⚠️ H2O unavailable: %s\n", e$message))
  cat("  Install with: install.packages('h2o') (requires Java 8+)\n")
  cat("  Skipping H2O deep learning model.\n")
})

# ---- Model Comparison ----
cat("\n--- Model Comparison ---\n")

comparison <- data.frame(
  Model = c("GBM", "XGBoost", "H2O_DeepLearning"),
  RMSE = c(
    ifelse(exists("gbm_rmse"), gbm_rmse, NA),
    ifelse(exists("xgb_rmse"), xgb_rmse, NA),
    h2o_rmse
  ),
  R2 = c(
    ifelse(exists("gbm_r2"), gbm_r2, NA),
    ifelse(exists("xgb_r2"), xgb_r2, NA),
    h2o_r2
  )
)

print(comparison)

# ---- Save Results ----
write_csv(comparison, file.path(OUTPUT_DIR, "r_model_comparison.csv"))
cat(sprintf("\n  Results saved to %s\n", OUTPUT_DIR))

cat("\n=== Duration & Convexity ML Analysis (R) COMPLETE ===\n")
