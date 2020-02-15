# ------------------------------------------------------------------------------
# REGRESSION MODEL EVALUATION
# Function to verify the predictive capabilities of the regression model.
# ------------------------------------------------------------------------------
# Environment:
base::library(reticulate)
reticulate::use_condaenv(condaenv = "GPU_ML_2", required = TRUE)
base::library(Metrics)
base::library(tidyverse)
base::library(tibble)
base::library(knitr)
# ------------------------------------------------------------------------------
Regression_Model_Verification <- function(actual, predicted){
  
  # Packages:
  if (!base::require(tidyverse)){install.packages('tidyverse'); base::require('tidyverse')}
  if (!base::require(tibble)){install.packages('tibble'); base::require('tibble')}
  if (!base::require(knitr)){install.packages('knitr'); base::require('knitr')}
  if (!base::require(Metrics)){install.packages('Metrics'); base::require('Metrics')}
  
  # Metrics:
  N <- base::length(actual)
  N_label <- "Number of Observatons"
  B <- Metrics::bias(actual = actual, predicted = predicted)
  B_label <- "Bias"
  BP <- Metrics::percent_bias(actual = actual, predicted = predicted)
  BP_label <- "Percent Bias"
  MeanAE <- Metrics::mae(actual = actual, predicted = predicted)
  MeanAE_label <- "Mean Absolute Error"
  MedianAE <-  Metrics::mdae(actual = actual, predicted = predicted)
  MedianAE_label <- "Median Absolute Error"
  MeanAPE <- Metrics::mape(actual = actual, predicted = predicted)
  MeanAPE_label <- "Mean Absolute Percentage Error"
  MedianAPE <- stats::median(Metrics::ape(actual = actual, predicted = predicted))
  MedianAPE_label <- "Median Absolute Percentage Error"
  MeanSE <- Metrics::mse(actual = actual, predicted = predicted)
  MeanSE_label <- "Mean Squared Error"
  MedianSE <- stats::median(Metrics::se(actual = actual, predicted = predicted))
  MedianSE_label <- "Median Squared Error"
  MeanSLE <- base::mean(Metrics::sle(actual = actual, predicted = predicted), na.rm = TRUE)
  MeanSLE_label <- "Mean Squared Log Error"
  MedianSLE <- stats::median(Metrics::sle(actual = actual, predicted = predicted), na.rm = TRUE)
  MedianSLE_label <- "Median Squared Log Error"
  RMeanSE <- Metrics::rmse(actual = actual, predicted = predicted)
  RMeanSE_label <- "Root Mean Squared Error"
  RMedianSE <- (stats::median(Metrics::se(actual = actual, predicted = predicted)))^0.5
  RMedianSE_label <- "Root Median Squared Error"
  MeanASE <- Metrics::mase(actual = actual, predicted = predicted)
  MeanASE_label <- "Mean Absolute Scaled Error"
  RAE <- Metrics::rae(actual = actual, predicted = predicted)
  RAE_label <- "Relative Absolute Error"
  RRSE <- Metrics::rrse(actual = actual, predicted = predicted)
  RRSE_label <- "Root Relative Squared Error"
  RSE <- Metrics::rse(actual = actual, predicted = predicted)
  RSE_label <- "Relative Squared Error"
  SMAPE <- Metrics::smape(actual = actual, predicted = predicted)
  SMAPE_label <- "Symmetric Mean Absolute Percentage Error"
  
  # Display metrics:
  result <- tibble::tibble(Metric = base::c(N_label, B_label, BP_label, MeanAE_label,
                                            MedianAE_label, MeanAPE_label, MedianAPE_label, MeanSE_label,
                                            MedianSE_label, MeanSLE_label, MedianSLE_label, RMeanSE_label,
                                            RMedianSE_label, MeanASE_label, RAE_label, RRSE_label,
                                            RSE_label, SMAPE_label),
                           Value = base::c(N, B, BP, MeanAE,
                                           MedianAE, MeanAPE, MedianAPE, MeanSE,
                                           MedianSE, MeanSLE, MedianSLE, RMeanSE, 
                                           RMedianSE, MeanASE, RAE, RRSE,
                                           RSE, SMAPE)) 
  
  result %>% dplyr::mutate(Value = base::round(Value, 6)) %>% knitr::kable() %>% base::print()
  base::invisible(result)
}
# ------------------------------------------------------------------------------