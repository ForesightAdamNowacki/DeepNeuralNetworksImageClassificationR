# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
# keras::install_keras(tensorflow = "gpu")
base::library(tidyverse)
base::library(deepviz)
base::source("D:/GitHub/DeepNeuralNetworksImageClassificationR/Useful_Functions.R")


# Create k folds



data_dir <- "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\data"
data_to <- "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds"
folds <- 5
seed = 42

Create_KFolds_Directories(data_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\data",
                          target_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds",
                          seed = 42)

Create_KFolds_Directories <- function(data_dir,
                                      target_dir,
                                      folds = 5,
                                      seed = 42){
  base::set.seed(seed = seed)
  sys_time <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  class_dirs <- base::list.dirs(data_dir)[2:base::length(base::list.dirs(data_dir))]
  classes <- base::basename(class_dirs)
  folders <- base::paste("Fold", 1:folds, sep = "_")
  
  files <- base::list()
  for (i in base::seq_along(class_dirs)){
    files[[i]] <- base::list.files(class_dirs[i])}
  names(files) <- base::basename(class_dirs)
  
  files <- tibble::tibble(file = base::do.call(c, files),
                          class = base::rep(names(base::sapply(files, length)), times = base::sapply(files, length))) %>%
    dplyr::mutate(original_file_path = base::paste(data_dir, class, file, sep = "/"),
                  fold = caret::createFolds(class, k = folds, list = FALSE))
  
  base::dir.create(path = target_dir, showWarnings = FALSE, recursive = TRUE)
  
  for (i in 1:folds){
    
    files_2 <- files %>%
      dplyr::filter(fold == i)
    fold <- base::paste("fold", i, sep = "_")
    fold_dir <- base::paste(target_dir, fold, sep = "/")
    base::dir.create(path = fold_dir, showWarnings = FALSE, recursive = TRUE)
    
    for (j in base::seq_along(classes)){
      
      class_dir <- base::paste(target_dir, fold, classes[j], sep = "/")
      base::dir.create(path = class_dir, showWarnings = FALSE, recursive = TRUE)
      files_3 <- files_2 %>%
        dplyr::filter(class == classes[j])
      base::file.copy(from = files_3$original_file_path,
                      to = base::paste(class_dir, files_3$file, sep = "/"))
    }
  }

  folds <- base::list.files(target_dir)[base::grepl("fold_", list.files(target_dir))]
  steps <- base::paste("step", 1:base::length(folds), sep = "_")
  
  '%!in%' <- Negate('%in%')
  
  for (i in base::seq_along(steps)){
    
    step_dir <- base::paste(target_dir, steps[i], sep = "/")
    train_dir <- base::paste(step_dir, "train", sep = "/")
    validation_dir <- base::paste(step_dir, "validation", sep = "/")
    set_dirs <- base::c(train_dir, validation_dir)
    
    base::dir.create(path = step_dir, showWarnings = FALSE, recursive = TRUE)
    base::dir.create(path = train_dir, showWarnings = FALSE, recursive = TRUE)
    base::dir.create(path = validation_dir, showWarnings = FALSE, recursive = TRUE)
    
    for (j in base::seq_along(set_dirs)){
      
      base::setwd(set_dirs[j])
      
      for (k in base::seq_along(classes)){
        
        if (j == 1){
          
          class_dir <- base::paste(base::getwd(), classes[k], sep = "/")
          base::dir.create(path = class_dir, showWarnings = FALSE, recursive = TRUE)
          train_files_from <- files %>%
            dplyr::filter(fold %!in% i) %>%
            dplyr::filter(class == classes[k]) 
          
          base::file.copy(from = train_files_from$original_file_path,
                          to = base::paste(class_dir, train_files_from$file, sep = "/"))}
        
        if (j == 2){
          
          class_dir <- base::paste(base::getwd(), classes[k], sep = "/")
          base::dir.create(path = class_dir, showWarnings = FALSE, recursive = TRUE)
          validation_files_from <- files %>%
            dplyr::filter(fold %in% i) %>%
            dplyr::filter(class == classes[k]) 
          base::file.copy(from = validation_files_from$original_file_path,
                          to = base::paste(class_dir, validation_files_from$file, sep = "/"))}
      }
    }
  }
  
  base::setwd(target_dir)
  
  for (i in base::seq_along(steps)){
  
    col_name <- steps[i]
    files <- files %>%
      dplyr::mutate(!!rlang::sym(col_name) := base::ifelse(fold == i, "Validation", "Train"))}
  
  readr::write_csv(files, "Cross_Validation_Splits.csv")
}

Create_KFolds_Directories(data_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\data",
                          target_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds",
                          seed = 42)






data_dir <- "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds"

setwd("..")
getwd()

#folds <- base::list.files(data_dir)[base::grepl("fold_", list.files(data_dir))]
#steps <- base::paste("step", 1:base::length(folds), sep = "_")
#classes <- base::c("cats", "dogs") # from above function

i <- 1
j <- train_dir
j <- validation_dir
k <- 1

for (i in base::seq_along(steps)){
  
  step_dir <- base::paste(data_dir, steps[i], sep = "/")
  train_dir <- base::paste(step_dir, "train", sep = "/")
  validation_dir <- base::paste(step_dir, "validation", sep = "/")
  base::dir.create(path = step_dir, showWarnings = FALSE, recursive = TRUE)
  base::dir.create(path = train_dir, showWarnings = FALSE, recursive = TRUE)
  base::dir.create(path = validation_dir, showWarnings = FALSE, recursive = TRUE)
  
  
}

classes
step
folds




directory <- "D:\\GitHub\\Datasets\\Cifar10\\test"
directories <- base::list.dirs(directory)[-1]; directories

for (i in base::seq_along(directories)){
 class <- base::basename(directories[i])
 old <- base::list.files(base::paste(directory, class, sep = "/"), full.names = TRUE)
 new <- base::paste(class, base::list.files(base::paste(directory, class, sep = "/"), full.names = FALSE), sep = "_")
 new <- base::paste(directory, class, new, sep = "/")
 base::file.copy(from = old,
                 to = new)
 
}






###
Categorical_Classifier_Verification <- function(actual,
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
                                               type_info,
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
                 filename = stringr::str_replace_all(base::paste(sys_time, type_info, "binary_categorical_model_evaluation_metrics.png", sep = "_"), ":", "-"),
                 vwidth = 900,
                 vheight = 1600,
                 expand = 5)
      if (open == TRUE){
        rstudioapi::viewer(stringr::str_replace_all(base::paste(sys_time, type_info, "binary_categorical_model_evaluation_metrics.png", sep = "_"), ":", "-"))
      }
    }
    
    base::invisible(base::list("Confusion_Matrix_Explanation" = result_1,
                               "Confusion_Matrix_Result" = result_2,
                               "Assessment_of_Classifier_Effectiveness" = result_3))
  }
  
  base::print(gt_table)
  
  if (save == TRUE){
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste(sys_time, type_info, "categorical_model_evaluation_metrics.png", sep = "_"), ":", "-"),
               vwidth = 900,
               vheight = 600,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste(sys_time, type_info, "categorical_model_evaluation_metrics.png", sep = "_"), ":", "-"))
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
                             "Binary classification" = binary_results))}


