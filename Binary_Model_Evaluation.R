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
# ------------------------------------------------------------------------------
Binary_Classifier_Verification <- function(actual,
                                           predicted,
                                           cutoff = 0.5,
                                           FN_cost = 1,
                                           FP_cost = 1,
                                           TN_cost = 0,
                                           TP_cost = 0){
  
  # Libraries:
  if (!base::require(Metrics)){utils::install.packages('Metrics'); base::require('Metrics')}  
  if (!base::require(tidyverse)){utils::install.packages('tidyverse'); base::require('tidyverse')}  
  if (!base::require(tibble)){utils::install.packages('tibble'); base::require('tibble')}  
  if (!base::require(knitr)){utils::install.packages('knitr'); base::require('knitr')}  
  
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
  # Area Under Curve (AUC):
  AUC <- Metrics::auc(actual = actual, predicted = probability)
  AUC_label <- "= Area Under ROC Curve"
  # Bias:
  BIAS <- base::mean(base::as.numeric(actual), na.rm = TRUE) - base::mean(base::as.numeric(predicted), na.rm = TRUE)
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
  
  result_3 <- tibble::tibble(Metric = base::c("Number of Observations", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)", "True Positive (TP)",
                                       "Condition Positive (P)", "Condition Negative (N)", "Accuracy (ACC)", "Area Under ROC Curve (AUC)",
                                       "Bias", "Classification Error", "True Positive Rate (TPR)", "True Negative Rate (TNR)",
                                       "Positive Prediction Value (PPV)", "Negative Predictive Value (NPV)", "False Negative Rate (FNR)", "False Positive Rate (FPR)",
                                       "False Discovery Rate (FDR)", "False Omission Rate (FOR)", "Threat Score (TS)", "F1 Score",
                                       "Bookmaker Informedness (BM)", "Markedness (MK)", "Gini Index", "Cost"),
                            Score = base::round(base::c(OBS, TN, FP, FN, TP,
                                                  P, N, ACC, AUC,
                                                  BIAS, CE, TPR, TNR,
                                                  PPV, NPV, FNR, FPR,
                                                  FDR, FOR, TS, F1,
                                                  BM, MK, GINI, COST), digits = 6),
                            Metric_Calculation = base::c(OBS_label, TN_label, FP_label, FN_label, TP_label,
                                                     P_label, N_label, ACC_label, AUC_label,
                                                     BIAS_label, CE_label, TPR_label, TNR_label,
                                                     PPV_label, NPV_label, FNR_label, FPR_label,
                                                     FDR_label, FOR_label, TS_label, F1_label,
                                                     BM_label, MK_label, GINI_label, COST_label),
                            Metric_AKA = base::c("-", "-", "Type I Error", "Type II Error", "-",
                                              "-", "-", "-", "-",
                                              "-", "-", "Sensitivity, Recall, Hit Rate", "Specifity, Selectivity",
                                              "Precision", "-", "Miss Rate", "Fall-Out",
                                              "-", "-", "Critical Success Index (CSI)", "-",
                                              "-", "-", "-", "-"),
                            ID = 1:base::length(Metric)) %>%
    dplyr::select(ID, Metric, Score, Metric_Calculation, Metric_AKA)
  
  result_3_label <- result_3 %>% knitr::kable(.)
  
  base::print(base::list("Confusion_Matrix_Explanation" = result_1_label,
                         "Confusion_Matrix_Result" = result_2_label,
                         "Assessment_of_Classifier_Effectiveness" = result_3_label))
  
  base::invisible(base::list("Confusion_Matrix_Explanation" = result_1,
                             "Confusion_Matrix_Result" = result_2,
                             "Assessment_of_Classifier_Effectiveness" = result_3))
} 
# ------------------------------------------------------------------------------
# Data: Johns Hopkins University Ionosphere database
# Description: Predict high-energy structures in the atmosphere from antenna data.
# Type: Classification
# Dimensions: 351 instances, 35 attributes
# Inputs: Numeric
# Output: Categorical, 2 class labels
# UCI Machine Learning Repository:
utils::browseURL(url = "https://archive.ics.uci.edu/ml/datasets/Ionosphere")
base::library(mlbench)
base::library(randomForest)
utils::data(Ionosphere)
data <- Ionosphere
data %>%
  tibble::as_tibble(.) %>%
  dplyr::mutate(V1 = NULL,
                V2 = NULL,
                Class = base::factor(base::ifelse(Class == "good", 1, 0))) -> data; data

model <- randomForest::randomForest(Class ~ ., data = data, ntree = 100)
predicted <- stats::predict(model, data, "prob")[,2]
actual <- base::as.numeric(data$Class) - 1

Binary_Classifier_Verification(actual = actual, predicted = predicted, cutoff = 0.25)
# ------------------------------------------------------------------------------
# CUT-OFF OPTIMIZATION
# Function to optimize the cut-off level in relation to many evaluation metrics.
Binary_Classifier_Cutoff_Optimization <- function(actual,
                                                  predicted, 
                                                  cuts = 25,
                                                  TN_cost = 0, 
                                                  FP_cost = 1, 
                                                  FN_cost = 1, 
                                                  TP_cost = 0,
                                                  text_size = 8,
                                                  title_size = 10){
  
  if (!require(Metrics)){utils::install.packages('Metrics'); require('Metrics')}  
  if (!require(tidyverse)){utils::install.packages('tidyverse'); require('tidyverse')}  
  if (!require(tibble)){utils::install.packages('tibble'); require('tibble')}  
  if (!require(gridExtra)){utils::install.packages('gridExtra'); require('gridExtra')}  
  
  cuts_values <- stats::runif(n = cuts, min = 0, max = 1)
  cuts_values <- base::sort(x = cuts_values, decreasing = FALSE)
  
  df <- tibble::tibble(ID = 1:cuts,
                        CUTOFF = cuts_values,
                        TN = base::numeric(cuts),
                        FP = base::numeric(cuts),
                        FN = base::numeric(cuts),
                        TP = base::numeric(cuts),
                        ACC = base::numeric(cuts),
                        COST = base::numeric(cuts),
                        TPR = base::numeric(cuts),
                        TNR = base::numeric(cuts),
                        PPV = base::numeric(cuts),
                        NPV = base::numeric(cuts),
                        FNR = base::numeric(cuts),
                        FPR = base::numeric(cuts),
                        FDR = base::numeric(cuts),
                        FOR = base::numeric(cuts),
                        TS = base::numeric(cuts),
                        F1 = base::numeric(cuts))
  
  actual = base::factor(actual, levels = base::c(0, 1))
  
  for (i in 1:cuts){
    predicted_classified = base::factor(base::ifelse(predicted < df$CUTOFF[i], 0, 1), levels = c(0, 1))
    confusion_matrix = base::table(actual, predicted_classified)
    
    TN = confusion_matrix[1, 1]
    FP = confusion_matrix[1, 2]
    FN = confusion_matrix[2, 1]
    TP = confusion_matrix[2, 2]
    PPV = TP/(TP + FP)
    TPR = TP/(TP + FN)
    
    df$TN[i] = confusion_matrix[1, 1]
    df$FP[i] = confusion_matrix[1, 2]
    df$FN[i] = confusion_matrix[2, 1]
    df$TP[i] = confusion_matrix[2, 2]
    df$ACC[i] = (TN + TP)/(TN + FN + FP + TP)
    df$COST[i] = TN_cost * TN + FP_cost * FP + FN_cost * FN + TP_cost * TP
    df$TPR[i] = TP/(TP + FN)
    df$TNR[i] = TN/(TN + FP)
    df$PPV[i] = TP/(TP + FP)
    df$NPV[i] = TN/(TN + FN)
    df$FNR[i] = FN/(FN + TP)
    df$FPR[i] = FP/(FP + TN)
    df$FDR[i] = FP/(FP + TP)
    df$FOR[i] = FN/(FN + TN)
    df$TS[i] = TP/(TP + FN + FP)
    df$F1[i] = (2 * PPV * TPR)/(PPV + TPR)
  }
  
  theme <- ggplot2::theme(plot.title = element_text(size = text_size, color = "black", face = "bold", hjust = 0.5, vjust = 0.5),
                          axis.text.y = element_text(size = text_size, color = "black", face = "plain"),
                          axis.text.x = element_text(size = text_size, color = "black", face = "plain"),
                          axis.title.y = element_text(size = text_size, color = "black", face = "bold"),
                          axis.title.x = element_text(size = text_size, color = "black", face = "bold"),
                          axis.ticks = element_line(size = 1, color = "black", linetype = "solid"),
                          axis.ticks.length = unit(0.1, "cm"),
                          plot.background = element_rect(fill = "gray80", color = "black", size = 1, linetype = "solid"),
                          panel.background = element_rect(fill = "gray90", color = "black", size = 0.5, linetype = "solid"),
                          panel.border = element_rect(fill = NA, color = "black", size = 0.5, linetype = "solid"),
                          panel.grid.major.x = element_line(color = "black", linetype = "dotted"),
                          panel.grid.major.y = element_line(color = "black", linetype = "dotted"),
                          panel.grid.minor.x = element_line(linetype = "blank"),
                          panel.grid.minor.y = element_line(linetype = "blank"),
                          plot.caption = element_text(size = text_size, color = "black", face = "bold", hjust = 1),
                          legend.position = "none")
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = TN)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point(size = 2) + theme +
    ggplot2::labs(title = "TRUE NEGATIVE OPTIMIZATION", x = "CUTOFF", y = "TRUE NEGATIVE COUNT") -> plot1
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = FP)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "FALSE POSITIVE OPTIMIZATION", x = "CUTOFF", y = "FALSE POSITIVE COUNT") -> plot2
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = FN)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "FALSE NEGATIVE OPTIMIZATION", x = "CUTOFF", y = "FALSE NEGATIVE COUNT") -> plot3
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = TP)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "TRUE POSITIVE OPTIMIZATION", x = "CUTOFF", y = "TRUE POSITIVE Count") -> plot4
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = ACC)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "ACCURACY OPTIMIZATION", x = "CUTOFF", y = "ACCURACY SCORE") -> plot5
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = COST)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "COST OPTIMIZATION", x = "CUTOFF", y = "COST SCORE") -> plot6
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = TPR)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "TOTAL POSITIVE RATE OPTIMIZATION", x = "CUTOFF", y = "TOTAL POSITIVE RATE SCORE") -> plot7
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = TNR)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "TOTAL NEGATIVE RATE OPTIMIZATION", x = "CUTOFF", y = "TOTAL NEGATIVE RATE SCORE") -> plot8
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = PPV)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "POSITIVE PREDICTION VALUE OPTIMIZATION", x = "CUTOFF", y = "POSITIVE PREDICTION VALUE SCORE") -> plot9
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = NPV)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "NEGATIVE PREDICTION VALUE OPTIMIZATION", x = "CUTOFF", y = "NEGATIVE PREDICTION VALUE SCORE") -> plot10
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = FNR)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "FALSE NEGATIVE RATE OPTIMIZATION", x = "CUTOFF", y = "FALSE NEGATIVE RATE SCORE") -> plot11
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = FPR)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "FALSE POSITIVE RATE OPTIMIZATION", x = "CUTOFF", y = "FALSE POSITIVE RATE SCORE") -> plot12
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = FDR)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "FALSE DISCOVERY RATE OPTIMIZATION", x = "CUTOFF", y = "FALSE DISCOVERY RATE SCORE") -> plot13
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = FOR)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "FALSE OMISSION RATE OPTIMIZATION", x = "CUTOFF", y = "FALSE OMISSION RATE SCORE") -> plot14
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = TS)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "THREAT OPTIMIZATION", x = "CUTOFF", y = "THREAT SCORE") -> plot15
  
  ggplot2::ggplot(data = df, mapping = aes(x = CUTOFF, y = F1)) +
    ggplot2::geom_line(lwd = 0.5, lty = 2) + ggplot2::geom_point() + theme +
    ggplot2::labs(title = "F1 OPTIMIZATION", x = "CUTOFF", y = "F1 SCORE") -> plot16
  
  plots <- gridExtra::grid.arrange(gridExtra::arrangeGrob(plot1, plot2, plot3, plot4,
                                                          plot5, plot6, plot7, plot8,
                                                          plot9, plot10, plot11, plot12,
                                                          plot13, plot14, plot15, plot16,
                                                          nrow = 4, ncol = 4))
  knitr::kable(df) %>% base::print(.)
}
# ------------------------------------------------------------------------------
# Data: Johns Hopkins University Ionosphere database
# Description: Predict high-energy structures in the atmosphere from antenna data.
# Type: Classification
# Dimensions: 351 instances, 35 attributes
# Inputs: Numeric
# Output: Categorical, 2 class labels
# UCI Machine Learning Repository:
utils::browseURL(url = "https://archive.ics.uci.edu/ml/datasets/Ionosphere")
base::library(mlbench)
base::library(randomForest)
utils::data(Ionosphere)
data <- Ionosphere
data %>%
  tibble::as_tibble(.) %>%
  dplyr::mutate(V1 = NULL,
                V2 = NULL,
                Class = base::factor(base::ifelse(Class == "good", 1, 0))) -> data; data

model <- randomForest::randomForest(Class ~ ., data = data, ntree = 100)
predicted <- stats::predict(model, data, type = "prob")[,2]
actual <- base::as.numeric(data$Class) - 1

Binary_Classifier_Cutoff_Optimization(actual = actual, predicted = predicted, cuts = 50)
# ------------------------------------------------------------------------------