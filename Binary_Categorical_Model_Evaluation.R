# ------------------------------------------------------------------------------
# BINARY MODEL EVALUATION
# Function to verify the predictive and classification capabilities of the binary model.
# ------------------------------------------------------------------------------
# Environment:
base::library(reticulate)
reticulate::use_condaenv(condaenv = "GPU_ML_2", required = TRUE)
base::library(Metrics)
base::library(tidyverse)
base::library(tibble)
base::library(knitr)
base::library(gridExtra)
base::library(gt)
base::library(mlbench)
base::library(randomForest)

# ------------------------------------------------------------------------------
Binary_Classifier_Verification <- function(actual,
                                           predicted,
                                           type_info = "",
                                           cutoff = 0.5,
                                           FN_cost = 1,
                                           FP_cost = 1,
                                           TN_cost = 0,
                                           TP_cost = 0,
                                           save = FALSE,
                                           open = TRUE){
  
  sys_time <- base::Sys.time()
  
  # Libraries:
  if (!base::require(Metrics)){utils::install.packages('Metrics'); base::require('Metrics')}  
  if (!base::require(tidyverse)){utils::install.packages('tidyverse'); base::require('tidyverse')}  
  if (!base::require(tibble)){utils::install.packages('tibble'); base::require('tibble')}  
  if (!base::require(knitr)){utils::install.packages('knitr'); base::require('knitr')}  
  if (!base::require(gt)){utils::install.packages('gt'); base::require('gt')}  
  
  # Confusion matrix explanation:
  result_1 <- tibble::tibble("Confusion Matrix" = base::c("Actual Negative (0)", "Actual Positive (1)"),
                             "Predicted Negative (0)" = base::c("True Negative (TN)", "False Negative (FN)"),
                             "Predicted Positive (1)" = base::c("False Positive (FP)", "True Positive (TP)"))
  result_1_label <- result_1 %>% knitr::kable(.)
  
  probability <- predicted
  if(base::length(base::unique(predicted)) > 2){predicted <- base::ifelse(predicted < cutoff, 0, 1)}
  
  # Confusion matrix result:
  confusion_matrix <- base::table(actual, predicted)
  result_2 <- tibble::tibble("Confusion Matrix" = base::c("Actual Negative (0)", "Actual Positive (1)"),
                             "Predicted Negative (0)" = base::c(confusion_matrix[1, 1], confusion_matrix[2, 1]),
                             "Predicted Positive (1)" = base::c(confusion_matrix[1, 2], confusion_matrix[2, 2])) 
  result_2_label <- result_2 %>% knitr::kable(.)
  
  # Assessment of classifier effectiveness:
  OBS <- base::sum(confusion_matrix); OBS_label <- "= TN + FP + FN + TP"
  TN <- confusion_matrix[1, 1]; TN_label <- "= TN"
  FP <- confusion_matrix[1, 2]; FP_label <- "= FP"
  FN <- confusion_matrix[2, 1]; FN_label <- "= FN"
  TP <- confusion_matrix[2, 2]; TP_label <- "= TP"
  P <- FN + TP; P_label <- "= FN + TP"
  N <- TN + FP; N_label <- "= TN + FP"
  
  # Accuracy (ACC):
  ACC <- (TN + TP)/(TN + FN + FP + TP)
  ACC_label <- "= (TN + TP)/(TN + FN + FP + TP) = (TN + TP)/(P + N)"
  
  # Balanced Accuracy (BACC):
  BACC <- (TN/(TN + FP) + TP/(FN + TP))/2
  BACC_label <- "= (TN/(TN + FP) + TP/(FN + TP))/2"
  
  # Area Under Curve (AUC):
  AUC <- Metrics::auc(actual = actual, predicted = probability)
  AUC_label <- "= Area Under ROC Curve"
  # Bias:
  BIAS <- base::mean(base::as.numeric(actual)) - base::mean(base::as.numeric(predicted))
  BIAS_label <- "= mean(actual) - mean(predicted)"
  # Classification Error (CE):
  CE <- (FN + FP)/(TN + FN + FP + TP)
  CE_label <- "= (FN + FP)/(TN + FN + FP + TP) = 1 - (TN + TP)/(TN + FN + FP + TP)"
  # Recall, Sensitivity, hit rate, True Positive Rate (TPR):
  TPR <- TP/(TP + FN)
  TPR_label <- "= TP/(TP + FN) = TP/P = 1 - FNR"
  # Specifity, selectivity, True Negative Rate (TNR):
  TNR <- TN/(TN + FP)
  TNR_label <- "= TN/(TN + FP) = TN/N = 1 - FPR"
  # Precision, Positive Prediction Value (PPV):
  PPV <- TP/(TP + FP)
  PPV_label <- "= TP/(TP + FP) = 1 - FDR"
  # Negative Predictive Value (NPV):
  NPV <- TN/(TN + FN)
  NPV_label <- "= TN/(TN + FN) = 1 - FOR"
  # False Negative Rate (FNR), miss rate:
  FNR <- FN/(FN + TP)
  FNR_label <- "= FN/(FN + TP) = FN/P = 1 - TPR"
  # False Positive Rate (FPR), fall-out:
  FPR <- FP/(FP + TN)
  FPR_label <- "= FP/(FP + TN) = FP/N = 1 - TNR"
  # False Discovery Rate (FDR):
  FDR <- FP/(FP + TP)
  FDR_label <- "= FP/(FP + TP) = 1 - PPV"
  # False Omission Rate (FOR):
  FOR <- FN/(FN + TN)
  FOR_label <- "= FN/(FN + TN) = 1 - NPV"
  # Threat Score (TS), Critical Success Index (CSI):
  TS <- TP/(TP + FN + FP)
  TS_label <- "= TP/(TP + FN + FP)"
  # F1:
  F1 <- (2 * PPV * TPR)/(PPV + TPR)
  F1_label <- "= (2 * PPV * TPR)/(PPV + TPR) = 2 * TP/(2 * TP + FP + FN)"
  # Informedness, Bookmaker Informedness (BM):
  BM <- TPR + TNR - 1
  BM_label <- "= TPR + TNR - 1"
  # Markedness (MK):
  MK <- PPV + NPV - 1
  MK_label <- "= PPV + NPV - 1"
  # Gini Index:
  GINI <- 2 * AUC - 1
  GINI_label <- "= 2 * AUC - 1"
  # Cost:
  COST <- FN * FN_cost + FP * FP_cost + TN * TN_cost + TP * TP_cost
  COST_label <- "= FN * FN_cost + FP * FP_cost + TN * TN_cost + TP * TP_cost"
  
  result_3 <- tibble::tibble(Metric = base::c("Number of Observations", "True Negative", "False Positive", "False Negative", "True Positive",
                                              "Condition Negative", "Condition Positive", "Accuracy", "Balanced Accuracy", "Area Under ROC Curve",
                                              "Bias", "Classification Error", "True Positive Rate", "True Negative Rate",
                                              "Positive Prediction Value", "Negative Predictive Value", "False Negative Rate", "False Positive Rate",
                                              "False Discovery Rate", "False Omission Rate", "Threat Score", "F1 Score",
                                              "Bookmaker Informedness", "Markedness", "Gini Index", "Cost"),
                             `Metric Abb` = base::c("RECORDS", "TN", "FP", "FN", "TP",
                                                    "N", "P", "ACC", "BACC", "AUC",
                                                    "BIAS", "CE", "TPR", "TNR", 
                                                    "PPV", "NPV", "FNR", "FPR",
                                                    "FDR", "FOR", "TS", "F1",
                                                    "BM", "MK", "GINI", "COST"),
                             `Metric Name` = base::c("-", "-", "Type I Error", "Type II Error", "-",
                                                     "-", "-", "-", "-", "-",
                                                     "-", "-", "Sensitivity, Recall, Hit Rate", "Specifity, Selectivity",
                                                     "Precision", "-", "Miss Rate", "Fall-Out",
                                                     "-", "-", "Critical Success Index", "-",
                                                     "-", "-", "-", "-"),
                             Score = base::round(base::c(OBS, TN, FP, FN, TP,
                                                         N, P, ACC, BACC, AUC,
                                                         BIAS, CE, TPR, TNR,
                                                         PPV, NPV, FNR, FPR,
                                                         FDR, FOR, TS, F1,
                                                         BM, MK, GINI, COST), digits = 6),
                             Calculation = base::c(OBS_label, TN_label, FP_label, FN_label, TP_label,
                                                  N_label, P_label, ACC_label, BACC_label, AUC_label,
                                                   BIAS_label, CE_label, TPR_label, TNR_label,
                                                   PPV_label, NPV_label, FNR_label, FPR_label,
                                                   FDR_label, FOR_label, TS_label, F1_label,
                                                   BM_label, MK_label, GINI_label, COST_label),
                             ID = base::c(1:7, 1:19)) %>%
    dplyr::select(ID, Metric, `Metric Abb`, `Metric Name`, Score, Calculation)
  
  result_3_label <- result_3 %>% knitr::kable(.)
  
  base::print(base::list("Confusion_Matrix_Explanation" = result_1_label,
                         "Confusion_Matrix_Result" = result_2_label,
                         "Assessment_of_Classifier_Effectiveness" = result_3_label))
  
  result_3 %>%
    dplyr::mutate(Group = base::ifelse(Metric %in% base::c("Number of Observations", "True Negative",
                                                           "False Positive", "False Negative",
                                                           "True Positive", "Condition Positive",
                                                           "Condition Negative"), 
                                       "Confusion Matrix Result", "Assessment of Classifier Effectiveness")) %>%
    gt::gt(rowname_col = "ID", groupname_col = "Group") %>%
    gt::tab_header(title = gt::md(base::paste("Model's evaluation metrics", sys_time)),
                   subtitle = gt::md("Binary classification model")) %>%
    gt::tab_source_note(gt::md(base::paste0("**Options**: ",
                                            "**cutoff** = ", cutoff,
                                            ", **TN_cost** = ", TN_cost,
                                            ", **FP_cost** = ", FP_cost,
                                            ", **FN_cost** = ", FN_cost,
                                            ", **TP_cost** = ", TP_cost))) %>%
    gt::tab_source_note(gt::md("More information available at: **https://github.com/ForesightAdamNowacki/DeepNeuralNetworksRepoR**.")) %>%
    gt::tab_spanner(label = "Metrics section",
                    columns = dplyr::vars(Metric, `Metric Abb`, `Metric Name`)) %>%
    gt::tab_spanner(label = "Performance section",
                    columns = dplyr::vars(Score, Calculation)) %>%
    gt::fmt_number(columns = dplyr::vars(Score),
                   decimals = 4,
                   use_seps = FALSE) %>%
    gt::cols_align(align = "left", columns = dplyr::vars(Metric, `Metric Abb`, `Metric Name`, Calculation)) %>%
    gt::tab_options(heading.background.color = "black",
                    table.background.color = "grey",
                    column_labels.background.color = "black",
                    row_group.background.color = "black",
                    source_notes.background.color = "black",
                    table.border.top.color = "black",
                    table.border.top.width = gt::px(3),
                    table.border.bottom.color = "black",
                    table.border.bottom.width = gt::px(3),
                    heading.title.font.size = 16,
                    table.font.size = 12,
                    source_notes.font.size = 10,
                    table.width = gt::pct(100),
                    data_row.padding = gt::px(5),
                    row_group.padding = gt::px(10),
                    source_notes.padding = gt::px(5)) %>% 
    gt::opt_table_outline(width = gt::px(3), color = "black") %>%
    gt::opt_table_lines() -> gt_table
  
  if (save == TRUE){
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste0(sys_time, " Binary model evaluation metrics ", type_info, ".png"), ":", "-"),
               vwidth = 900,
               vheight = 1600,
               expand = 5)
    if (open == TRUE){
    rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Binary model evaluation metrics ", type_info, ".png"), ":", "-"))
    }
  }

  gt_table %>% base::print(.)
  
  base::invisible(base::list("Confusion_Matrix_Explanation" = result_1,
                             "Confusion_Matrix_Result" = result_2,
                             "Assessment_of_Classifier_Effectiveness" = result_3))
}

# ------------------------------------------------------------------------------
# CUT-OFF OPTIMIZATION
# Function to optimize the cut-off level in relation to many evaluation metrics.
Binary_Classifier_Cutoff_Optimization <- function(actual,
                                                  predicted, 
                                                  type_info = "",
                                                  cuts = 25,
                                                  TN_cost = 0, 
                                                  FP_cost = 1, 
                                                  FN_cost = 1, 
                                                  TP_cost = 0,
                                                  text_size = 8,
                                                  title_size = 10,
                                                  key_metric = CUTOFF,
                                                  ascending = TRUE,
                                                  seed_value = 42,
                                                  top = 25,
                                                  save = FALSE,
                                                  open = TRUE){
  
  sys_time <- base::Sys.time()
  
  # Libraries:
  if (!require(Metrics)){utils::install.packages('Metrics'); require('Metrics')}  
  if (!require(tidyverse)){utils::install.packages('tidyverse'); require('tidyverse')}  
  if (!require(tibble)){utils::install.packages('tibble'); require('tibble')}  
  if (!require(gridExtra)){utils::install.packages('gridExtra'); require('gridExtra')}  
  if (!require(gt)){utils::install.packages('gt'); require('gt')}  
  if (!require(webshot)){utils::install.packages('webshot'); require('webshot')} 
  if (!require(stringr)){utils::install.packages('stringr'); require('stringr')} 
    
  key_metric <- dplyr::enquo(key_metric) 
  key_metric_name <- dplyr::quo_name(key_metric)
  
  base::set.seed(seed = seed_value)
  cuts_values <- stats::runif(n = cuts, min = 0, max = 1)
  cuts_values <- base::sort(x = cuts_values, decreasing = FALSE)
  
  df <- tibble::tibble(ID = 1:cuts,
                       CUTOFF = cuts_values,
                       RECORDS = base::numeric(cuts),
                       TN = base::numeric(cuts),
                       FP = base::numeric(cuts),
                       FN = base::numeric(cuts),
                       TP = base::numeric(cuts),
                       P = base::numeric(cuts),
                       N = base::numeric(cuts),
                       ACC = base::numeric(cuts),
                       BACC = base::numeric(cuts),
                       AUC = base::numeric(cuts),
                       BIAS = base::numeric(cuts),
                       CE = base::numeric(cuts),
                       TPR = base::numeric(cuts),
                       TNR = base::numeric(cuts),
                       PPV = base::numeric(cuts),
                       NPV = base::numeric(cuts),
                       FNR = base::numeric(cuts),
                       FPR = base::numeric(cuts),
                       FDR = base::numeric(cuts),
                       FOR = base::numeric(cuts),
                       TS = base::numeric(cuts),
                       F1 = base::numeric(cuts),
                       BM = base::numeric(cuts),
                       MK = base::numeric(cuts),
                       GINI = base::numeric(cuts),
                       COST = base::numeric(cuts))
  
  actual = base::factor(actual, levels = base::c(0, 1))
  
  for (i in 1:cuts){
    
    predicted_class <- base::factor(base::ifelse(predicted < df$CUTOFF[i], 0, 1), levels = c(0, 1))
    confusion_matrix <- base::table(actual, predicted_class)
    
    df$RECORDS <- base::sum(confusion_matrix)
    df$TN[i] <- confusion_matrix[1, 1]
    df$FP[i] <- confusion_matrix[1, 2]
    df$FN[i] <- confusion_matrix[2, 1]
    df$TP[i] <- confusion_matrix[2, 2]
    df$N <- df$TN[i] + df$FP[i]
    df$P <- df$FN[i] + df$TP[i]
    df$ACC[i] <- (df$TN[i] + df$TP[i])/(df$TN[i] + df$FN[i] + df$FP[i] + df$TP[i])
    df$BACC[i] <- (df$TN[i]/(df$TN[i] + df$FP[i]) + df$TP[i]/(df$FN[i] + df$TP[i]))/2
    df$AUC[i] <- Metrics::auc(actual = actual, predicted = predicted)
    df$BIAS[i] <- base::mean(base::as.numeric(actual)) - base::mean(base::as.numeric(predicted_class))
    df$CE[i] <- (df$FN[i] + df$FP[i])/(df$TN[i] + df$FN[i] + df$FP[i] + df$TP[i])
    df$TPR[i] <- df$TP[i]/(df$TP[i] + df$FN[i])
    df$TNR[i] <- df$TN[i]/(df$TN[i] + df$FP[i])
    df$PPV[i] <- df$TP[i]/(df$TP[i] + df$FP[i])
    df$NPV[i] <- df$TN[i]/(df$TN[i] + df$FN[i])
    df$FNR[i] <- df$FN[i]/(df$FN[i] + df$TP[i])
    df$FPR[i] <- df$FP[i]/(df$FP[i] + df$TN[i])
    df$FDR[i] <- df$FP[i]/(df$FP[i] + df$TP[i])
    df$FOR[i] <- df$FN[i]/(df$FN[i] + df$TN[i])
    df$TS[i] <- df$TP[i]/(df$TP[i] + df$FN[i] + df$FP[i])
    df$F1[i] <- (2 * df$PPV[i] * df$TPR[i])/(df$PPV[i] + df$TPR[i])
    df$BM[i] <- df$TPR[i] + df$TNR[i] - 1
    df$MK[i] <- df$PPV[i] + df$NPV[i] - 1
    df$GINI[i] <- 2 * df$AUC[i] - 1
    df$COST[i] <- TN_cost * df$TN[i] + FP_cost * df$FP[i] + FN_cost * df$FN[i] + TP_cost * df$TP[i]
  }
  
  if(ascending == TRUE){
      df %>%
        dplyr::arrange(!!key_metric) -> df
    } else {
      df %>%
        dplyr::arrange(dplyr::desc(!!key_metric)) -> df
  }

  df %>%
    utils::head(., top) %>%
    dplyr::mutate(ID = dplyr::row_number()) %>%
    gt::gt(rowname_col = "ID") %>%
    gt::tab_header(title = gt::md(base::paste("Cut-off value optimization", sys_time)),
                   subtitle = gt::md("Binary classification model")) %>%
    gt::fmt_number(columns = dplyr::vars(CUTOFF, ACC, BACC, AUC, BIAS, CE, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, F1, BM, MK, GINI),
                   decimals = 4,
                   use_seps = FALSE) %>%
    gt::tab_spanner(label = "Confusion Matrix Result",
                    columns = dplyr::vars(RECORDS, TN, FP, FN, TP, N, P)) %>%
    gt::tab_spanner(label = "Assessment of Classifier Effectiveness",
                    columns = dplyr::vars(ACC, BACC, AUC, BIAS, CE, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, F1, BM, MK, GINI, COST)) %>%
    gt::tab_options(heading.background.color = "black",
                    table.background.color = "grey",
                    column_labels.background.color = "black",
                    row_group.background.color = "black",
                    source_notes.background.color = "black",
                    table.border.top.color = "black",
                    table.border.top.width = gt::px(3),
                    table.border.bottom.color = "black",
                    table.border.bottom.width = gt::px(3),
                    heading.title.font.size = 16,
                    table.font.size = 12,
                    source_notes.font.size = 10,
                    table.width = gt::pct(100),
                    data_row.padding = gt::px(5),
                    source_notes.padding = gt::px(5),
                    grand_summary_row.border.color = "black",
                    grand_summary_row.border.width = gt::px(3),
                    grand_summary_row.padding = gt::px(5)) %>% 
    gt::opt_table_outline(width = gt::px(3), color = "black") %>%
    gt::grand_summary_rows(fns = list("Mean value" = "mean",
                                      "Median value" = "median",
                                      "Minimum value" = "min",
                                      "Maximum value" = "max"),
                           formatter = fmt_number,
                           columns = dplyr::vars(CUTOFF, ACC, BACC, AUC, BIAS, CE, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, F1, BM, MK, GINI),
                           decimals = 4,
                           use_seps = FALSE) %>%
    gt::grand_summary_rows(fns = list("Mean value" = "mean",
                                      "Median value" = "median",
                                      "Minimum value" = "min",
                                      "Maximum value" = "max"),
                           formatter = fmt_number,
                           columns = dplyr::vars(RECORDS, TN, FP, FN, TP, N, P, COST),
                           decimals = 0,
                           use_seps = FALSE) %>%
    gt::tab_source_note(gt::md(base::paste0("**Options**: ",
                                            "**cuts** = ", cuts,
                                            ", **TN_cost** = ", TN_cost,
                                            ", **FP_cost** = ", FP_cost,
                                            ", **FN_cost** = ", FN_cost,
                                            ", **TP_cost** = ", TP_cost,
                                            ", **key_metric** = ", key_metric_name, 
                                            ", **ascending** = ", ascending,
                                            ", **seed_value** = ", seed_value,
                                            ", **top** = ", top))) %>%
    gt::tab_source_note(gt::md("More information available at: **https://github.com/ForesightAdamNowacki/DeepNeuralNetworksRepoR**.")) %>%
    gt::opt_table_lines() -> gt_table
  
  if (save == TRUE){
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste0(sys_time, " Binary model cut-off value optimization ", type_info, ".png"), ":", "-"),
               vwidth = 1600,
               vheight = 900,
               expand = 5)
    if (open == TRUE){
    rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Binary model cut-off value optimization ", type_info, ".png"), ":", "-"))
    }
  }
  
  gt_table %>% base::print(.)
  base::invisible(df)
}

# ------------------------------------------------------------------------------
# Categorical model evaluation:
Categorical_model_evaluation <- function(actual,
                                         probabilities,
                                         labels,
                                         type_info = "",
                                         cutoff = 0.5,
                                         FN_cost = 1,
                                         FP_cost = 1,
                                         TN_cost = 0,
                                         TP_cost = 0,
                                         save = FALSE,
                                         open = TRUE){
  
  sys_time = base::Sys.time()
  
  # Libraries:
  if (!require(Metrics)){utils::install.packages('Metrics'); require('Metrics')}  
  if (!require(tidyverse)){utils::install.packages('tidyverse'); require('tidyverse')}  
  if (!require(tibble)){utils::install.packages('tibble'); require('tibble')}  
  if (!require(gridExtra)){utils::install.packages('gridExtra'); require('gridExtra')}  
  if (!require(gt)){utils::install.packages('gt'); require('gt')}  
  if (!require(webshot)){utils::install.packages('webshot'); require('webshot')} 
  if (!require(stringr)){utils::install.packages('stringr'); require('stringr')} 
  
  predicted <- base::max.col(probabilities)
  
  base::table(base::factor(actual, levels = 1:base::length(labels), labels = labels), 
              base::factor(predicted, levels = 1:base::length(labels), labels = labels)) %>%
    base::as.data.frame() %>%
    tibble::as_tibble() %>%
    dplyr::rename(actual = Var1,
                  predicted = Var2,
                  count = Freq) %>%
    tidyr::complete(actual, predicted, fill = base::list(count = 0)) %>%
    dplyr::arrange(actual, predicted) %>%
    dplyr::group_by(actual) %>%
    dplyr::mutate(records = base::sum(count),
                  freq = count/records) %>%
    dplyr::filter(actual == predicted) %>%
    dplyr::mutate(predicted = NULL) %>%
    dplyr::rename(correct = count,
                  accuracy = freq) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(incorrect = records - correct,
                  inaccuracy = 1 - accuracy) %>%
    dplyr::select(actual, records, correct, incorrect, accuracy, inaccuracy) -> stats; stats
  
  base::table(base::factor(actual, levels = 1:base::length(labels), labels = labels), 
              base::factor(predicted, levels = 1:base::length(labels), labels = labels)) %>%
    base::as.data.frame() %>%
    tibble::as_tibble() %>%
    dplyr::rename(actual = Var1,
                  predicted = Var2,
                  count = Freq) %>%
    tidyr::complete(actual, predicted, fill = base::list(count = 0)) %>%
    dplyr::group_by(predicted) %>%
    dplyr::mutate(records = base::sum(count),
                  precision = count/records) %>%
    dplyr::filter(actual == predicted) %>%
    dplyr::ungroup() %>%
    dplyr::select(actual, precision) -> precision; precision
  
  base::table(base::factor(actual, levels = 1:base::length(labels), labels = labels), 
              base::factor(predicted, levels = 1:base::length(labels), labels = labels)) %>%
    base::as.data.frame() %>%
    tibble::as_tibble() %>%
    dplyr::rename(actual = Var1,
                  predicted = Var2,
                  count = Freq) %>%
    tidyr::complete(actual, predicted, fill = base::list(count = 0)) %>%
    dplyr::group_by(actual) %>%
    dplyr::mutate(records = base::sum(count),
                  recall = count/records) %>%
    dplyr::filter(actual == predicted) %>%
    dplyr::ungroup() %>%
    dplyr::select(actual, recall) -> recall; recall
  
  stats %>%
    dplyr::left_join(precision, by = "actual") %>%
    dplyr::left_join(recall, by = "actual") %>%
    dplyr::mutate(f1 = (2 * precision * recall)/(precision + recall)) %>%
    tidyr::replace_na(base::list(f1 = 0)) %>%
    dplyr::mutate(type = "Split into categories",
                  actual = base::as.character(actual)) %>%
    dplyr::rename(class = actual) -> stats_2
  
  stats_2 %>%
    dplyr::mutate(class = "overall") %>%
    dplyr::group_by(class) %>%
    dplyr::summarise(records = base::sum(records),
                     correct = base::sum(correct),
                     incorrect = records - correct,
                     accuracy = correct/records,
                     inaccuracy = incorrect/records,
                     type = "All categories") -> stats_3
  
  dplyr::bind_rows(stats_2, stats_3) %>%
    dplyr::rename(Class = class,
                  Records = records,
                  Correct = correct,
                  Incorrect = incorrect, 
                  Accuracy = accuracy,
                  Inaccuracy = inaccuracy,
                  Precision = precision,
                  Recall = recall,
                  F1 = f1) -> stats_4
  
  probabilities %>%
    base::as.data.frame() %>%
    dplyr::mutate(id = dplyr::row_number()) %>%
    tidyr::pivot_longer(cols = dplyr::starts_with("V"), names_to = "predicted_class", values_to = "probability") %>%
    dplyr::mutate(predicted_class = base::as.numeric(stringr::str_sub(predicted_class, 2, -1))) %>%
    dplyr::left_join(tibble::tibble(id = 1:base::length(actual),
                                    actual = actual), by = base::c("id")) %>%
    dplyr::mutate(ground_true = base::ifelse(predicted_class == actual, 1, 0)) %>%
    dplyr::select(ground_true, probability) -> stats_5
  
  stats_4 %>%
    dplyr::ungroup() %>%
    gt::gt(rowname_col = "Class", groupname_col = "type") %>%
    gt::tab_header(title = gt::md(base::paste("Model's evaluation metrics", sys_time)),
                   subtitle = gt::md("Categorical classification model")) %>%
    gt::tab_source_note(gt::md("More information available at: **https://github.com/ForesightAdamNowacki/DeepNeuralNetworksRepoR**.")) %>%
    gt::fmt_number(columns = dplyr::vars(Accuracy, Inaccuracy, Precision, Recall, F1),
                   decimals = 4,
                   use_seps = FALSE) %>%
    gt::fmt_number(columns = dplyr::vars(Records, Correct, Incorrect),
                   decimals = 0,
                   use_seps = FALSE) %>%
    gt::tab_spanner(label = "Counts",
                    columns = dplyr::vars(Records, Correct, Incorrect)) %>% 
    gt::tab_spanner(label = "Metrics",
                    columns = dplyr::vars(Accuracy, Inaccuracy, Precision, Recall, F1)) %>%
    gt::tab_options(heading.background.color = "black",
                    table.background.color = "grey",
                    column_labels.background.color = "black",
                    row_group.background.color = "black",
                    source_notes.background.color = "black",
                    table.border.top.color = "black",
                    table.border.top.width = gt::px(3),
                    table.border.bottom.color = "black",
                    table.border.bottom.width = gt::px(3),
                    heading.title.font.size = 16,
                    table.font.size = 12,
                    source_notes.font.size = 10,
                    table.width = gt::pct(100),
                    data_row.padding = gt::px(5),
                    row_group.padding = gt::px(10),
                    source_notes.padding = gt::px(5)) %>% 
    gt::opt_table_outline(width = gt::px(3), color = "black") %>%
    gt::opt_table_lines() -> gt_table
  
  Binary_Classifier_Verification_2 <- function(actual,
                                               predicted,
                                               type_info = "",
                                               cutoff = 0.5,
                                               FN_cost = 1,
                                               FP_cost = 1,
                                               TN_cost = 0,
                                               TP_cost = 0,
                                               save = FALSE,
                                               open = TRUE){
    
    # Confusion matrix explanation:
    result_1 <- tibble::tibble("Confusion Matrix" = base::c("Actual Negative (0)", "Actual Positive (1)"),
                               "Predicted Negative (0)" = base::c("True Negative (TN)", "False Negative (FN)"),
                               "Predicted Positive (1)" = base::c("False Positive (FP)", "True Positive (TP)"))
    
    probability <- predicted
    if(base::length(base::unique(predicted)) > 2){predicted <- base::ifelse(predicted < cutoff, 0, 1)}
    
    # Confusion matrix result:
    confusion_matrix <- base::table(actual, predicted)
    result_2 <- tibble::tibble("Confusion Matrix" = base::c("Actual Negative (0)", "Actual Positive (1)"),
                               "Predicted Negative (0)" = base::c(confusion_matrix[1, 1], confusion_matrix[2, 1]),
                               "Predicted Positive (1)" = base::c(confusion_matrix[1, 2], confusion_matrix[2, 2])) 
    
    # Assessment of classifier effectiveness:
    OBS <- base::sum(confusion_matrix); OBS_label <- "= TN + FP + FN + TP"
    TN <- confusion_matrix[1, 1]; TN_label <- "= TN"
    FP <- confusion_matrix[1, 2]; FP_label <- "= FP"
    FN <- confusion_matrix[2, 1]; FN_label <- "= FN"
    TP <- confusion_matrix[2, 2]; TP_label <- "= TP"
    P <- FN + TP; P_label <- "= FN + TP"
    N <- TN + FP; N_label <- "= TN + FP"
    
    # Accuracy (ACC):
    ACC <- (TN + TP)/(TN + FN + FP + TP)
    ACC_label <- "= (TN + TP)/(TN + FN + FP + TP) = (TN + TP)/(P + N)"
    
    # Balanced Accuracy (BACC):
    BACC <- (TN/(TN + FP) + TP/(FN + TP))/2
    BACC_label <- "= (TN/(TN + FP) + TP/(FN + TP))/2"
    
    # Area Under Curve (AUC):
    AUC <- Metrics::auc(actual = actual, predicted = probability)
    AUC_label <- "= Area Under ROC Curve"
    # Bias:
    BIAS <- base::mean(base::as.numeric(actual)) - base::mean(base::as.numeric(predicted))
    BIAS_label <- "= mean(actual) - mean(predicted)"
    # Classification Error (CE):
    CE <- (FN + FP)/(TN + FN + FP + TP)
    CE_label <- "= (FN + FP)/(TN + FN + FP + TP) = 1 - (TN + TP)/(TN + FN + FP + TP)"
    # Recall, Sensitivity, hit rate, True Positive Rate (TPR):
    TPR <- TP/(TP + FN)
    TPR_label <- "= TP/(TP + FN) = TP/P = 1 - FNR"
    # Specifity, selectivity, True Negative Rate (TNR):
    TNR <- TN/(TN + FP)
    TNR_label <- "= TN/(TN + FP) = TN/N = 1 - FPR"
    # Precision, Positive Prediction Value (PPV):
    PPV <- TP/(TP + FP)
    PPV_label <- "= TP/(TP + FP) = 1 - FDR"
    # Negative Predictive Value (NPV):
    NPV <- TN/(TN + FN)
    NPV_label <- "= TN/(TN + FN) = 1 - FOR"
    # False Negative Rate (FNR), miss rate:
    FNR <- FN/(FN + TP)
    FNR_label <- "= FN/(FN + TP) = FN/P = 1 - TPR"
    # False Positive Rate (FPR), fall-out:
    FPR <- FP/(FP + TN)
    FPR_label <- "= FP/(FP + TN) = FP/N = 1 - TNR"
    # False Discovery Rate (FDR):
    FDR <- FP/(FP + TP)
    FDR_label <- "= FP/(FP + TP) = 1 - PPV"
    # False Omission Rate (FOR):
    FOR <- FN/(FN + TN)
    FOR_label <- "= FN/(FN + TN) = 1 - NPV"
    # Threat Score (TS), Critical Success Index (CSI):
    TS <- TP/(TP + FN + FP)
    TS_label <- "= TP/(TP + FN + FP)"
    # F1:
    F1 <- (2 * PPV * TPR)/(PPV + TPR)
    F1_label <- "= (2 * PPV * TPR)/(PPV + TPR) = 2 * TP/(2 * TP + FP + FN)"
    # Informedness, Bookmaker Informedness (BM):
    BM <- TPR + TNR - 1
    BM_label <- "= TPR + TNR - 1"
    # Markedness (MK):
    MK <- PPV + NPV - 1
    MK_label <- "= PPV + NPV - 1"
    # Gini Index:
    GINI <- 2 * AUC - 1
    GINI_label <- "= 2 * AUC - 1"
    # Cost:
    COST <- FN * FN_cost + FP * FP_cost + TN * TN_cost + TP * TP_cost
    COST_label <- "= FN * FN_cost + FP * FP_cost + TN * TN_cost + TP * TP_cost"
    
    result_3 <- tibble::tibble(Metric = base::c("Number of Observations", "True Negative", "False Positive", "False Negative", "True Positive",
                                                "Condition Negative", "Condition Positive", "Accuracy", "Balanced Accuracy", "Area Under ROC Curve",
                                                "Bias", "Classification Error", "True Positive Rate", "True Negative Rate",
                                                "Positive Prediction Value", "Negative Predictive Value", "False Negative Rate", "False Positive Rate",
                                                "False Discovery Rate", "False Omission Rate", "Threat Score", "F1 Score",
                                                "Bookmaker Informedness", "Markedness", "Gini Index", "Cost"),
                               `Metric Abb` = base::c("RECORDS", "TN", "FP", "FN", "TP",
                                                      "N", "P", "ACC", "BACC", "AUC",
                                                      "BIAS", "CE", "TPR", "TNR", 
                                                      "PPV", "NPV", "FNR", "FPR",
                                                      "FDR", "FOR", "TS", "F1",
                                                      "BM", "MK", "GINI", "COST"),
                               `Metric Name` = base::c("-", "-", "Type I Error", "Type II Error", "-",
                                                       "-", "-", "-", "-", "-",
                                                       "-", "-", "Sensitivity, Recall, Hit Rate", "Specifity, Selectivity",
                                                       "Precision", "-", "Miss Rate", "Fall-Out",
                                                       "-", "-", "Critical Success Index", "-",
                                                       "-", "-", "-", "-"),
                               Score = base::round(base::c(OBS, TN, FP, FN, TP,
                                                           N, P, ACC, BACC, AUC,
                                                           BIAS, CE, TPR, TNR,
                                                           PPV, NPV, FNR, FPR,
                                                           FDR, FOR, TS, F1,
                                                           BM, MK, GINI, COST), digits = 6),
                               Calculation = base::c(OBS_label, TN_label, FP_label, FN_label, TP_label,
                                                     N_label, P_label, ACC_label, BACC_label, AUC_label,
                                                     BIAS_label, CE_label, TPR_label, TNR_label,
                                                     PPV_label, NPV_label, FNR_label, FPR_label,
                                                     FDR_label, FOR_label, TS_label, F1_label,
                                                     BM_label, MK_label, GINI_label, COST_label),
                               ID = base::c(1:7, 1:19)) %>%
      dplyr::select(ID, Metric, `Metric Abb`, `Metric Name`, Score, Calculation)
    
    result_3 %>%
      dplyr::mutate(Group = base::ifelse(Metric %in% base::c("Number of Observations", "True Negative",
                                                             "False Positive", "False Negative",
                                                             "True Positive", "Condition Positive",
                                                             "Condition Negative"), 
                                         "Confusion Matrix Result", "Assessment of Classifier Effectiveness")) %>%
      gt::gt(rowname_col = "ID", groupname_col = "Group") %>%
      gt::tab_header(title = gt::md(base::paste("Model's evaluation metrics", sys_time)),
                     subtitle = gt::md("Binary-Categorical classification model")) %>%
      gt::tab_source_note(gt::md(base::paste0("**Options**: ",
                                              "**cutoff** = ", cutoff,
                                              ", **TN_cost** = ", TN_cost,
                                              ", **FP_cost** = ", FP_cost,
                                              ", **FN_cost** = ", FN_cost,
                                              ", **TP_cost** = ", TP_cost))) %>%
      gt::tab_source_note(gt::md("More information available at: **https://github.com/ForesightAdamNowacki/DeepNeuralNetworksRepoR**.")) %>%
      gt::tab_spanner(label = "Metrics section",
                      columns = dplyr::vars(Metric, `Metric Abb`, `Metric Name`)) %>%
      gt::tab_spanner(label = "Performance section",
                      columns = dplyr::vars(Score, Calculation)) %>%
      gt::fmt_number(columns = dplyr::vars(Score),
                     decimals = 4,
                     use_seps = FALSE) %>%
      gt::cols_align(align = "left", columns = dplyr::vars(Metric, `Metric Abb`, `Metric Name`, Calculation)) %>%
      gt::tab_options(heading.background.color = "black",
                      table.background.color = "grey",
                      column_labels.background.color = "black",
                      row_group.background.color = "black",
                      source_notes.background.color = "black",
                      table.border.top.color = "black",
                      table.border.top.width = gt::px(3),
                      table.border.bottom.color = "black",
                      table.border.bottom.width = gt::px(3),
                      heading.title.font.size = 16,
                      table.font.size = 12,
                      source_notes.font.size = 10,
                      table.width = gt::pct(100),
                      data_row.padding = gt::px(5),
                      row_group.padding = gt::px(10),
                      source_notes.padding = gt::px(5)) %>% 
      gt::opt_table_outline(width = gt::px(3), color = "black") %>%
      gt::opt_table_lines() -> gt_table
    
    base::print(gt_table)
    
    if (save == TRUE){
      gt::gtsave(data = gt_table,
                 filename = stringr::str_replace_all(base::paste0(sys_time, " Binary-categorical model evaluation metrics ", type_info, ".png"), ":", "-"),
                 vwidth = 900,
                 vheight = 1600,
                 expand = 5)
      if (open == TRUE){
        rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Binary-categorical model evaluation metrics ", type_info, ".png"), ":", "-"))
      }
    }
    
    base::invisible(base::list("Confusion_Matrix_Explanation" = result_1,
                               "Confusion_Matrix_Result" = result_2,
                               "Assessment_of_Classifier_Effectiveness" = result_3))
  }
  
  base::print(gt_table)
  
  if (save == TRUE){
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste0(sys_time, " Categorical model evaluation metrics ", type_info, ".png"), ":", "-"),
               vwidth = 900,
               vheight = 600,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Categorical model evaluation metrics ", type_info, ".png"), ":", "-"))
    }
  }
  
  binary_results <- Binary_Classifier_Verification_2(actual = stats_5$ground_true,
                                                     predicted = stats_5$probability,
                                                     type_info = type_info,
                                                     save = save,
                                                     open = open,
                                                     cutoff = cutoff,
                                                     FN_cost = FN_cost,
                                                     FP_cost = FP_cost,
                                                     TN_cost = TN_cost,
                                                     TP_cost = TP_cost)
  
  base::invisible(base::list("Categorical classification" = stats_4,
                             "Binary classification" = binary_results))
}
