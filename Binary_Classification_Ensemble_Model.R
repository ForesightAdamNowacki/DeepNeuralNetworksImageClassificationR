# ------------------------------------------------------------------------------
# ENSEMBLE BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model:
model_name <- "Ensemble_Model"
model_type <- "Binary"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
# 2. Create 'model_name' folder in cwd:
if (base::dir.exists(base::paste(base::getwd(), model_name, sep = "/")) == FALSE){base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))}
# 3. Create 'model_type' subfolder in 'model_name' main folder:
if (base::dir.exists(base::paste(base::getwd(), model_name, model_type, sep = "/")) == FALSE){base::dir.create(path = base::paste(base::getwd(), model_name, model_type, sep = "/"))}

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::source("D:/GitHub/DeepNeuralNetworksImageClassificationR/Useful_Functions.R")

# Directories:
train_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/train"
validation_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/validation"
test_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/test"
models_store_dir <- base::paste(base::getwd(), model_name, model_type, sep = "/")
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"

train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# ------------------------------------------------------------------------------
models_vector <- base::c("ResNet50", "Xception", "MobileNet_V2", "Inception_V3", "Inception_ResNet_V2")

# Build ensembel model with weights and cutoff optimization on train dataset:
ensemble_1 <- Binary_Ensemble_Model(models_vector = models_vector,
                                    optimization_dataset = "train",
                                    save_option = FALSE,
                                    default_cutoff = 0.5,
                                    cuts = 50,
                                    weights = 50,
                                    key_metric = "ACC",
                                    key_metric_as_string = TRUE,
                                    ascending = FALSE,
                                    top = 10,
                                    seed = 1,
                                    summary_type = "mean",
                                    n = 5)

# ------------------------------------------------------------------------------
# Build ensembel model with weights and cutoff optimization on validation dataset:
ensemble_2 <- Binary_Ensemble_Model(models_vector = models_vector,
                                    optimization_dataset = "validation",
                                    save_option = FALSE,
                                    default_cutoff = 0.5,
                                    cuts = 50,
                                    weights = 50,
                                    key_metric = "ACC",
                                    key_metric_as_string = TRUE,
                                    ascending = FALSE,
                                    top = 10,
                                    seed = 1,
                                    summary_type = "mean",
                                    n = 5)

# ------------------------------------------------------------------------------
# Build ensembel model with weights and cutoff optimization on combined train and validation dataset:
ensemble_3 <- Binary_Ensemble_Model(models_vector = models_vector,
                                    optimization_dataset = "train+validation",
                                    save_option = FALSE,
                                    default_cutoff = 0.5,
                                    cuts = 50,
                                    weights = 50,
                                    key_metric = "ACC",
                                    key_metric_as_string = TRUE,
                                    ascending = FALSE,
                                    top = 10,
                                    seed = 1,
                                    summary_type = "mean",
                                    n = 5)

# ------------------------------------------------------------------------------
# Train dataset results - ensemble models and partial models summary and comparison:
ensemble_1$train_dataset_results %>%
  dplyr::rename(Ensemble_Model_1 = Ensemble_Model) %>%
  dplyr::mutate(Ensemble_Model_2 = ensemble_2$train_dataset_results$Ensemble_Model,
                Ensemble_Model_3 = ensemble_3$train_dataset_results$Ensemble_Model) %>%
  knitr::kable()
  
# ------------------------------------------------------------------------------
# Validation dataset results - ensemble models and partial models summary and comparison:
ensemble_1$validation_dataset_results %>%
  dplyr::rename(Ensemble_Model_1 = Ensemble_Model) %>%
  dplyr::mutate(Ensemble_Model_2 = ensemble_2$validation_dataset_results$Ensemble_Model,
                Ensemble_Model_3 = ensemble_3$validation_dataset_results$Ensemble_Model) %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Test dataset results - ensemble models and partial models summary and comparison:
ensemble_1$test_dataset_results %>%
  dplyr::rename(Ensemble_Model_1 = Ensemble_Model) %>%
  dplyr::mutate(Ensemble_Model_2 = ensemble_2$test_dataset_results$Ensemble_Model,
                Ensemble_Model_3 = ensemble_3$test_dataset_results$Ensemble_Model) %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Cutoffs comparison:
base::list(ensemble_1 = ensemble_1$optimal_cutoff,
           ensemble_2 = ensemble_2$optimal_cutoff,
           ensemble_3 = ensemble_3$optimal_cutoff) 

# ------------------------------------------------------------------------------
# Partial models weights:
base::list(ensemble_1 = ensemble_1$optimal_weights,
           ensemble_2 = ensemble_2$optimal_weights,
           ensemble_3 = ensemble_3$optimal_weights)

base::list(ensemble_1 = ensemble_1$optimal_weights,
           ensemble_2 = ensemble_2$optimal_weights,
           ensemble_3 = ensemble_3$optimal_weights) %>%
  base::lapply(., base::cbind) %>%
  base::do.call(base::cbind, .) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(Partial_Model = models_vector) %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                      names_to = "Ensemble_Model",
                      values_to = "Weights") %>%
  dplyr::mutate(Ensemble_Model = stringr::str_replace(Ensemble_Model, "V", "Ensemble_Model_"),
                Partial_Model = base::factor(Partial_Model, levels = models_vector, labels = models_vector, ordered = TRUE)) %>%
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
# Organize correct and incorrect ensemble model predictions on train dataset:
save_summary_files <- TRUE
save_correct_images <- FALSE
save_incorrect_images <- FALSE

Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = train_dir,
                                                  actual_classes = base::rep(base::c(0, 1), train_files$category_obs),
                                                  predicted = ensemble_3$train_models_predictions$Ensemble_Model,
                                                  cutoff = ensemble_3$optimal_cutoff,
                                                  type_info = model_name,
                                                  save_summary_files = save_summary_files,
                                                  save_correct_images = save_correct_images,
                                                  save_incorrect_images = save_incorrect_images)

# Organize correct and incorrect ensemble model predictions on validation dataset:
Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = validation_dir,
                                                  actual_classes = base::rep(base::c(0, 1), validation_files$category_obs),
                                                  predicted = ensemble_3$validation_models_predictions$Ensemble_Model,
                                                  cutoff = ensemble_3$optimal_cutoff,
                                                  type_info = model_name,
                                                  save_summary_files = save_summary_files,
                                                  save_correct_images = save_correct_images,
                                                  save_incorrect_images = save_incorrect_images)

# Organize correct and incorrect ensemble model predictions on test dataset:
Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = test_dir,
                                                  actual_classes = base::rep(base::c(0, 1), test_files$category_obs),
                                                  predicted = ensemble_3$test_models_predictions$Ensemble_Model,
                                                  cutoff = ensemble_3$optimal_cutoff,
                                                  type_info = model_name,
                                                  save_summary_files = save_summary_files,
                                                  save_correct_images = save_correct_images,
                                                  save_incorrect_images = save_incorrect_images)

# ------------------------------------------------------------------------------
# Visualize predictions distribution:
save_plot <- FALSE
labels <- base::sort(base::as.character(train_files$category)); labels

# Train:
train_probabilities <- base::matrix(data = base::cbind(1 - ensemble_3$train_models_predictions$Ensemble_Model,
                                                       ensemble_3$train_models_predictions$Ensemble_Model), ncol = 2)
train_actual <- base::rep(base::c(0, 1), train_files$category_obs)
train_predicted_2 <- train_probabilities[base::matrix(data = base::cbind(1:base::nrow(train_probabilities), train_actual + 1), byrow = FALSE, ncol = 2)]

Display_Target_Class_Predictions_Distribution(actual = train_actual,
                                              predicted = train_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = base::paste(model_name, "train", sep = "_"),
                                              save_plot = save_plot)

# Validation:
validation_probabilities <- base::matrix(data = base::cbind(1 - ensemble_3$validation_models_predictions$Ensemble_Model,
                                                       ensemble_3$validation_models_predictions$Ensemble_Model), ncol = 2)
validation_actual <- base::rep(base::c(0, 1), validation_files$category_obs)
validation_predicted_2 <- validation_probabilities[base::matrix(data = base::cbind(1:base::nrow(validation_probabilities), validation_actual + 1), byrow = FALSE, ncol = 2)]

Display_Target_Class_Predictions_Distribution(actual = validation_actual,
                                              predicted = validation_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = base::paste(model_name, "validation", sep = "_"),
                                              save_plot = save_plot)

# Test:
test_probabilities <- base::matrix(data = base::cbind(1 - ensemble_3$test_models_predictions$Ensemble_Model,
                                                            ensemble_3$test_models_predictions$Ensemble_Model), ncol = 2)
test_actual <- base::rep(base::c(0, 1), test_files$category_obs)
test_predicted_2 <- test_probabilities[base::matrix(data = base::cbind(1:base::nrow(test_probabilities), test_actual + 1), byrow = FALSE, ncol = 2)]

Display_Target_Class_Predictions_Distribution(actual = test_actual,
                                              predicted = test_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = base::paste(model_name, "test", sep = "_"),
                                              save_plot = save_plot)

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to all classes:
save_plot <- FALSE

Display_All_Classes_Predictions_Distribution(actual = train_actual + 1,
                                             predicted = train_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = base::paste(model_name, "train", sep = "_"),
                                             save_plot = save_plot)

Display_All_Classes_Predictions_Distribution(actual = validation_actual + 1,
                                             predicted = validation_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = base::paste(model_name, "validation", sep = "_"),
                                             save_plot = save_plot)

Display_All_Classes_Predictions_Distribution(actual = test_actual + 1,
                                             predicted = test_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = base::paste(model_name, "test", sep = "_"),
                                             save_plot = save_plot)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki