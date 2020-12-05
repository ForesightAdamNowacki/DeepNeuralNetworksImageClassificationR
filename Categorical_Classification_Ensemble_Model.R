# ------------------------------------------------------------------------------
# ENSEMBLE CATEGORICAL MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
# browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model:
model_name <- "Ensemble_Model"
model_type <- "Categorical"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
# 2. Create 'model_name' folder in cwd:
if (dir.exists(paste(getwd(), model_name, sep = "/")) == FALSE){dir.create(path = paste(getwd(), model_name, sep = "/"))}
# 3. Create 'model_type' subfolder in 'model_name' main folder:
if (dir.exists(paste(getwd(), model_name, model_type, sep = "/")) == FALSE){dir.create(path = paste(getwd(), model_name, model_type, sep = "/"))}

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
library(tensorflow)
library(keras)
library(tidyverse)
library(deepviz)
source("D:/GitHub/DeepNeuralNetworksImageClassificationR/Useful_Functions.R")

# Directories:
train_dir <- "D:/GitHub/Datasets/Cifar10/train"
validation_dir <- "D:/GitHub/Datasets/Cifar10/validation"
test_dir <- "D:/GitHub/Datasets/Cifar10/test"
models_store_dir <- paste(getwd(), model_name, model_type, sep = "/")
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"

train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# ------------------------------------------------------------------------------
models_vector <- c("Xception", "MobileNet_V2", "Inception_ResNet_V2")
labels <- as.character(train_files$category)

# Build ensembel model with weights and cutoff optimization on train dataset:
ensemble_1 <- Categorical_Ensemble_Model(models_vector = models_vector,
                                         labels = labels,
                                         optimization_dataset = "train",
                                         save_option = TRUE,
                                         weights = 50,
                                         key_metric = "Accuracy", 
                                         key_metric_as_string = TRUE,
                                         ascending = FALSE,
                                         top = 10,
                                         seed = 1,
                                         summary_type = "mean",
                                         cwd = models_store_dir,
                                         n = 3)

# ------------------------------------------------------------------------------
# Build ensembel model with weights and cutoff optimization on validation dataset:
ensemble_2 <- Categorical_Ensemble_Model(models_vector = models_vector,
                                         labels = labels,
                                         optimization_dataset = "validation",
                                         save_option = TRUE,
                                         weights = 50,
                                         key_metric = "Accuracy", 
                                         key_metric_as_string = TRUE,
                                         ascending = FALSE,
                                         top = 10,
                                         seed = 2,
                                         summary_type = "mean",
                                         cwd = models_store_dir,
                                         n = 3)

# ------------------------------------------------------------------------------
# Build ensembel model with weights and cutoff optimization on combined train and validation dataset:
ensemble_3 <- Categorical_Ensemble_Model(models_vector = models_vector,
                                         labels = labels,
                                         optimization_dataset = "train+validation",
                                         save_option = TRUE,
                                         weights = 50,
                                         key_metric = "Accuracy", 
                                         key_metric_as_string = TRUE,
                                         ascending = FALSE,
                                         top = 10,
                                         seed = 3,
                                         summary_type = "mean",
                                         cwd = models_store_dir,
                                         n = 3)

# ------------------------------------------------------------------------------
# Train dataset results - ensemble models and partial models summary and comparison:
ensemble_1$train_dataset_results %>%
  dplyr::rename(Ensemble_Model_1 = Ensemble_Model_Score) %>%
  dplyr::mutate(Ensemble_Model_2 = ensemble_2$train_dataset_results$Ensemble_Model_Score,
                Ensemble_Model_3 = ensemble_3$train_dataset_results$Ensemble_Model_Score) %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Validation dataset results - ensemble models and partial models summary and comparison:
ensemble_1$validation_dataset_results %>%
  dplyr::rename(Ensemble_Model_1 = Ensemble_Model_Score) %>%
  dplyr::mutate(Ensemble_Model_2 = ensemble_2$validation_dataset_results$Ensemble_Model_Score,
                Ensemble_Model_3 = ensemble_3$validation_dataset_results$Ensemble_Model_Score) %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Test dataset results - ensemble models and partial models summary and comparison:
ensemble_1$test_dataset_results %>%
  dplyr::rename(Ensemble_Model_1 = Ensemble_Model_Score) %>%
  dplyr::mutate(Ensemble_Model_2 = ensemble_2$test_dataset_results$Ensemble_Model_Score,
                Ensemble_Model_3 = ensemble_3$test_dataset_results$Ensemble_Model_Score) %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Partial models weights:
list(ensemble_1 = ensemble_1$optimal_weights,
           ensemble_2 = ensemble_2$optimal_weights,
           ensemble_3 = ensemble_3$optimal_weights)

list(ensemble_1 = ensemble_1$optimal_weights,
           ensemble_2 = ensemble_2$optimal_weights,
           ensemble_3 = ensemble_3$optimal_weights) %>%
  lapply(., cbind) %>%
  do.call(cbind, .) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(Partial_Model = models_vector) %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                      names_to = "Ensemble_Model",
                      values_to = "Weights") %>%
  dplyr::mutate(Ensemble_Model = stringr::str_replace(Ensemble_Model, "V", "Ensemble_Model_"),
                Partial_Model = factor(Partial_Model, levels = models_vector, labels = models_vector, ordered = TRUE)) %>%
  ggplot2::ggplot(data = ., mapping = ggplot2::aes(x = Partial_Model, y = Weights, fill = Ensemble_Model)) +
  ggplot2::geom_bar(stat = "identity", color = "black", position = ggplot2::position_dodge()) +
  ggplot2::theme(plot.title = ggplot2::element_text(size = 9, color = "black", face = "bold", hjust = 0.5, vjust = 0.5),
                 axis.text.y = ggplot2::element_text(size = 7, color = "black", face = "plain"),
                 axis.text.x = ggplot2::element_text(size = 7, color = "black", face = "plain"),
                 axis.title.y = ggplot2::element_text(size = 7, color = "black", face = "bold"),
                 axis.title.x = ggplot2::element_text(size = 7, color = "black", face = "bold"),
                 axis.ticks = ggplot2::element_line(size = 1, color = "black", linetype = "solid"),
                 axis.ticks.length = ggplot2::unit(0.1, "cm"),
                 plot.background = ggplot2::element_rect(fill = "gray80", color = "black", size = 1, linetype = "solid"),
                 panel.background = ggplot2::element_rect(fill = "gray90", color = "black", size = 0.5, linetype = "solid"),
                 panel.border = ggplot2::element_rect(fill = NA, color = "black", size = 0.5, linetype = "solid"),
                 panel.grid.major.x = ggplot2::element_line(color = "black", linetype = "dotted"),
                 panel.grid.major.y = ggplot2::element_line(color = "black", linetype = "dotted"),
                 panel.grid.minor.x = ggplot2::element_line(linetype = "blank"),
                 panel.grid.minor.y = ggplot2::element_line(linetype = "blank"),
                 legend.box.background = ggplot2::element_rect(color = "black", size = 0.5, linetype = "solid"),
                 legend.background = ggplot2::element_rect(fill = "gray90", size = 0.5, linetype = "solid", color = "black"),
                 legend.position = "bottom",
                 legend.box.spacing = ggplot2::unit(0.25, "cm"),
                 legend.text = ggplot2::element_text(size = 7, color = "black", face = "plain"),
                 legend.title = ggplot2::element_text(size = 7, color = "black", face = "bold")) +
  ggplot2::guides(fill = guide_legend("Ensemble model:", nrow = 1)) +
  ggplot2::labs(x = "Partial Model",
                y = "Partial Model Contributions/Weights",
                title = "Ensemble models")

# ------------------------------------------------------------------------------
# Save correct and incorrect predictions:
save_summary_files <- TRUE
save_correct_images <- FALSE
save_incorrect_images <- FALSE

# Train:
Train_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = train_dir,
                                                                                                         actual_classes = ensemble_3$train_actual_class,
                                                                                                         predicted = ensemble_3$train_ensemble_model_prediction,
                                                                                                         type_info = model_name,
                                                                                                         cwd = models_store_dir,
                                                                                                         save_summary_files = save_summary_files,
                                                                                                         save_correct_images = save_correct_images,
                                                                                                         save_incorrect_images = save_incorrect_images)

# Validation:
Validation_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = validation_dir,
                                                                                                              actual_classes = ensemble_3$validation_actual_class,
                                                                                                              predicted = ensemble_3$validation_ensemble_model_prediction,
                                                                                                              type_info = model_name,
                                                                                                              cwd = models_store_dir,
                                                                                                              save_summary_files = save_summary_files,
                                                                                                              save_correct_images = save_correct_images,
                                                                                                              save_incorrect_images = save_incorrect_images)

# Test:
Test_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = test_dir,
                                                                                                        actual_classes = ensemble_3$test_actual_class,
                                                                                                        predicted = ensemble_3$test_ensemble_model_prediction,
                                                                                                        type_info = model_name,
                                                                                                        cwd = models_store_dir,
                                                                                                        save_summary_files = save_summary_files,
                                                                                                        save_correct_images = save_correct_images,
                                                                                                        save_incorrect_images = save_incorrect_images)

# ------------------------------------------------------------------------------
# Visualize predictions distribution:
save_plot <- TRUE

train_probabilities <- as.matrix(ensemble_3$train_ensemble_model_prediction)
train_classes <- ensemble_3$train_actual_class
train_predicted_2 <- train_probabilities[matrix(data = c(1:nrow(train_probabilities), train_classes), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = train_classes,
                                              predicted = train_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = paste(model_name, "train", sep = "_"),
                                              save_plot = save_plot)

validation_probabilities <- as.matrix(ensemble_3$validation_ensemble_model_prediction)
validation_classes <- ensemble_3$validation_actual_class
validation_predicted_2 <- validation_probabilities[matrix(data = c(1:nrow(validation_probabilities), validation_classes), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = validation_classes,
                                              predicted = validation_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = paste(model_name, "validation", sep = "_"),
                                              save_plot = save_plot)

test_probabilities <- as.matrix(ensemble_3$test_ensemble_model_prediction)
test_classes <- ensemble_3$test_actual_class
test_predicted_2 <- test_probabilities[matrix(data = c(1:nrow(test_probabilities), test_classes), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = test_classes,
                                              predicted = test_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = paste(model_name, "test", sep = "_"),
                                              save_plot = save_plot)

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to all classes:
save_plot <- TRUE

Display_All_Classes_Predictions_Distribution(actual = train_classes,
                                             predicted = train_probabilities,
                                             labels = labels,
                                             bins = 4,
                                             type_info = paste(model_name, "train", sep = "_"),
                                             save_plot = save_plot,
                                             plot_size = 30)

Display_All_Classes_Predictions_Distribution(actual = validation_classes,
                                             predicted = validation_probabilities,
                                             labels = labels,
                                             bins = 4,
                                             type_info = paste(model_name, "validation", sep = "_"),
                                             save_plot = save_plot,
                                             plot_size = 30)

Display_All_Classes_Predictions_Distribution(actual = test_classes,
                                             predicted = test_probabilities,
                                             labels = labels,
                                             bins = 4,
                                             type_info = paste(model_name, "test", sep = "_"),
                                             save_plot = save_plot,
                                             plot_size = 30)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki