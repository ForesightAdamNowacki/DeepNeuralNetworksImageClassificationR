# ------------------------------------------------------------------------------
# ENSEMBLE BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Script variables:
model_name <- "Ensemble_Model"
model_type <- "Binary"
dataset_types <- base::c("train", "validation", "test")
ensemble_split_types <- base::c("train", "validation", "train_validation")
script_seed <- 2020

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'CNN' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))
# 3. Create 'Binary' subfolder in 'CNN' main folder
base::dir.create(path = base::paste(base::getwd(), model_name, model_type, sep = "/"))

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Useful_Functions.R")

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
# Models predictions:
train_pattern <- "train_binary_probabilities"
validation_pattern <- "validation_binary_probabilities"
test_dir <- "test_binary_probabilities"

models_vector <- base::c("ResNet50", "Xception", "MobileNet_V2", "Inception_V3", "Inception_ResNet_V2")
all_predictions <- base::list()
for (i in base::seq_along(models_vector)){
  model <- base::list()
  model[[1]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = train_pattern, full.names = TRUE))
  model[[2]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = validation_pattern, full.names = TRUE))
  model[[3]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = test_dir, full.names = TRUE))
  all_predictions[[i]] <- model}

Display_List_Structure(all_predictions, n = 5)

# ------------------------------------------------------------------------------
# Predictions:
# Train:
train_predictions <- base::list()
for (i in 1:base::length(all_predictions)){train_predictions[[i]] <- all_predictions[[i]][[1]]$V2}
train_predictions <- base::do.call(base::cbind, train_predictions)
base::colnames(train_predictions) <- models_vector; utils::head(train_predictions); base::cat(base::dim(train_predictions))

# Validation:
validation_predictions <- base::list()
for (i in 1:base::length(all_predictions)){validation_predictions[[i]] <- all_predictions[[i]][[2]]$V2}
validation_predictions <- base::do.call(base::cbind, validation_predictions)
base::colnames(validation_predictions) <- models_vector; utils::head(validation_predictions); base::cat(base::dim(validation_predictions))

# Test:
test_predictions <- base::list()
for (i in 1:base::length(all_predictions)){test_predictions[[i]] <- all_predictions[[i]][[3]]$V2}
test_predictions <- base::do.call(base::cbind, test_predictions)
base::colnames(test_predictions) <- models_vector; utils::head(test_predictions); base::cat(base::dim(test_predictions))

# ------------------------------------------------------------------------------
# Actual:
train_actual <- all_predictions[[1]][[1]]$actual_class; base::cat(base::length(train_actual), "\n"); base::print(base::table(train_actual))
validation_actual <- all_predictions[[1]][[2]]$actual_class; base::cat(base::length(validation_actual), "\n"); base::print(base::table(validation_actual))
test_actual <- all_predictions[[1]][[3]]$actual_class; base::cat(base::length(test_actual), "\n"); base::print(base::table(test_actual))

# ------------------------------------------------------------------------------
# Train results for single component models:
base::setwd(models_store_dir)
save_option <- TRUE
default_cutoff <- 0.5

train_default <- base::list()
for (i in 1:base::length(models_vector)){
  Assessment_of_Classifier_Effectiveness <- Binary_Classifier_Verification(actual = train_actual,
                                                                           predicted = train_predictions[,i],
                                                                           cutoff = default_cutoff,
                                                                           type_info = base::paste(models_vector[i], "default_cutoff", dataset_types[1], sep = "_"),
                                                                           save = save_option,
                                                                           open = FALSE)[[3]]
  train_default[[i]] <- Assessment_of_Classifier_Effectiveness}

train_default_summary <- base::data.frame(Metric = train_default[[1]]$Metric)
for (i in 1:base::length(models_vector)){train_default_summary <- dplyr::bind_cols(train_default_summary, train_default[[i]][5])}
base::colnames(train_default_summary) <- base::c("Metric", models_vector)
train_default_summary %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Validation results for single component models:
base::setwd(models_store_dir)
save_option <- TRUE
default_cutoff <- 0.5

validation_default <- base::list()
for (i in 1:base::length(models_vector)){
  Assessment_of_Classifier_Effectiveness <- Binary_Classifier_Verification(actual = validation_actual,
                                                                           predicted = validation_predictions[,i],
                                                                           cutoff = default_cutoff,
                                                                           type_info = base::paste(models_vector[i], "default_cutoff", dataset_types[2], sep = "_"),
                                                                           save = save_option,
                                                                           open = FALSE)[[3]]
  validation_default[[i]] <- Assessment_of_Classifier_Effectiveness}

validation_default_summary <- base::data.frame(Metric = validation_default[[1]]$Metric)
for (i in 1:base::length(models_vector)){validation_default_summary <- dplyr::bind_cols(validation_default_summary, validation_default[[i]][5])}
base::colnames(validation_default_summary) <- base::c("Metric", models_vector)
validation_default_summary %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Test results for single component models:
base::setwd(models_store_dir)
save_option <- TRUE
default_cutoff <- 0.5

test_default <- base::list()
for (i in 1:base::length(models_vector)){
  Assessment_of_Classifier_Effectiveness <- Binary_Classifier_Verification(actual = test_actual,
                                                                           predicted = test_predictions[,i],
                                                                           cutoff = default_cutoff,
                                                                           type_info = base::paste(models_vector[i], "default_cutoff", dataset_types[3], sep = "_"),
                                                                           save = save_option,
                                                                           open = FALSE)[[3]]
  test_default[[i]] <- Assessment_of_Classifier_Effectiveness}

test_default_summary <- base::data.frame(Metric = test_default[[1]]$Metric)
for (i in 1:base::length(models_vector)){test_default_summary <- dplyr::bind_cols(test_default_summary, test_default[[i]][5])}
base::colnames(test_default_summary) <- base::c("Metric", models_vector)
test_default_summary %>%
  knitr::kable()

# ------------------------------------------------------------------------------
# Optimize cutoff and weights in ensemble model on train data using simulation approach:
train_ensemble_optimization <- Optimize_Ensemble_Cutoff_Model(actual = train_actual,
                                                              predictions = train_predictions,
                                                              cuts = 50,
                                                              weights = 50,
                                                              top = 10,
                                                              seed = script_seed,
                                                              summary_type = "median")
train_ensemble_optimization_cutoff <- train_ensemble_optimization[[3]] %>%
  dplyr::pull()

train_ensemble_optimization_weights <- train_ensemble_optimization[[4]] %>%
  tidyr::pivot_longer(cols = dplyr::everything()) %>%
  dplyr::select(value) %>%
  dplyr::pull()

train_result_train_optimization <- mapply("*", base::as.data.frame(train_predictions), train_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

validation_result_train_optimization <- mapply("*", base::as.data.frame(validation_predictions), train_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

test_result_train_optimization <- mapply("*", base::as.data.frame(test_predictions), train_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

train_ensemble_optimization_predictions <- base::list(train_result_train_optimization,
                                                      validation_result_train_optimization,
                                                      test_result_train_optimization)

# ------------------------------------------------------------------------------
# Optimize cutoff and weights in ensemble model on validation data using simulation approach:
validation_ensemble_optimization <- Optimize_Ensemble_Cutoff_Model(actual = validation_actual,
                                                              predictions = validation_predictions,
                                                              cuts = 50,
                                                              weights = 50,
                                                              top = 10,
                                                              seed = script_seed,
                                                              summary_type = "median")
validation_ensemble_optimization_cutoff <- validation_ensemble_optimization[[3]] %>%
  dplyr::pull()

validation_ensemble_optimization_weights <- validation_ensemble_optimization[[4]] %>%
  tidyr::pivot_longer(cols = dplyr::everything()) %>%
  dplyr::select(value) %>%
  dplyr::pull()

train_result_validation_optimization <- mapply("*", base::as.data.frame(train_predictions), validation_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

validation_result_validation_optimization <- mapply("*", base::as.data.frame(validation_predictions), validation_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

test_result_validation_optimization <- mapply("*", base::as.data.frame(test_predictions), validation_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

validation_ensemble_optimization_predictions <- base::list(train_result_validation_optimization,
                                                           validation_result_validation_optimization,
                                                           test_result_validation_optimization)

# ------------------------------------------------------------------------------
# Optimize cutoff and weights in ensemble model on train and validation data combined using simulation approach:
train_validation_ensemble_optimization <- Optimize_Ensemble_Cutoff_Model(actual = base::c(train_actual, validation_actual),
                                                                         predictions = base::rbind(train_predictions, validation_predictions),
                                                                         cuts = 50,
                                                                         weights = 50,
                                                                         top = 10,
                                                                         seed = script_seed,
                                                                         summary_type = "median")
train_validation_ensemble_optimization_cutoff <- validation_ensemble_optimization[[3]] %>%
  dplyr::pull()

train_validation_ensemble_optimization_weights <- validation_ensemble_optimization[[4]] %>%
  tidyr::pivot_longer(cols = dplyr::everything()) %>%
  dplyr::select(value) %>%
  dplyr::pull()

train_result_train_validation_optimization <- mapply("*", base::as.data.frame(train_predictions), train_validation_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

validation_result_train_validation_optimization <- mapply("*", base::as.data.frame(validation_predictions), train_validation_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

test_result_train_validation_optimization <- mapply("*", base::as.data.frame(test_predictions), train_validation_ensemble_optimization_weights) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(prediction = base::rowSums(.)) %>%
  dplyr::select(prediction) %>%
  dplyr::pull()

train_validation_ensemble_optimization_predictions <- base::list(train_result_train_validation_optimization,
                                                                 validation_result_train_validation_optimization,
                                                                 test_result_train_validation_optimization)

# ------------------------------------------------------------------------------
# Ensemble model results on train data:
base::setwd(models_store_dir)
save_option <- TRUE
optimized_cutoff <- base::c(train_ensemble_optimization_cutoff, validation_ensemble_optimization_cutoff, train_validation_ensemble_optimization_cutoff)
ensemble_model_train_predictions <- base::list(train_result_train_optimization,
                                               train_result_validation_optimization,
                                               train_result_train_validation_optimization)

train_dataset_ensemble_model_results <- base::list()
for (i in 1:base::length(ensemble_model_train_predictions)){
  train_dataset_ensemble_model_results[[i]] <- Binary_Classifier_Verification(actual = train_actual,
                                                                         predicted = ensemble_model_train_predictions[[i]],
                                                                         cutoff = optimized_cutoff[i],
                                                                         type_info = base::paste(ensemble_split_types[i], "optimized_cutoff", dataset_types[1], sep = "_"),
                                                                         save = save_option,
                                                                         open = FALSE)[[3]]}

train_optimized_ensemble_summary <- base::data.frame(Metric = train_dataset_ensemble_model_results[[1]]$Metric)
for (i in 1:base::length(train_dataset_ensemble_model_results)){train_optimized_ensemble_summary <- dplyr::bind_cols(train_optimized_ensemble_summary, train_dataset_ensemble_model_results[[i]][5])}
base::colnames(train_optimized_ensemble_summary) <- base::c("Metric", "train_optimized", "validation_optimized", "train_validation_optimized")
train_optimized_ensemble_summary %>%
  knitr::kable()

train_default_summary %>%
  dplyr::left_join(train_optimized_ensemble_summary, by = "Metric") %>%
  knitr::kable()

datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
train_default_summary %>%
  dplyr::left_join(train_optimized_ensemble_summary, by = "Metric") %>%
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, model_name, "train_dataset_.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Ensemble model results on validation data:
base::setwd(models_store_dir)
save_option <- TRUE
optimized_cutoff <- base::c(train_ensemble_optimization_cutoff, validation_ensemble_optimization_cutoff, train_validation_ensemble_optimization_cutoff)
ensemble_model_validation_predictions <- base::list(validation_result_train_optimization, 
                                                    validation_result_validation_optimization, 
                                                    validation_result_train_validation_optimization)

validation_dataset_ensemble_model_results <- base::list()
for (i in 1:base::length(ensemble_model_train_predictions)){
  validation_dataset_ensemble_model_results[[i]] <- Binary_Classifier_Verification(actual = validation_actual,
                                                                              predicted = ensemble_model_validation_predictions[[i]],
                                                                              cutoff = optimized_cutoff[i],
                                                                              type_info = base::paste(ensemble_split_types[i], "optimized_cutoff", dataset_types[2], sep = "_"),
                                                                              save = save_option,
                                                                              open = FALSE)[[3]]}

validation_optimized_ensemble_summary <- base::data.frame(Metric = validation_dataset_ensemble_model_results[[1]]$Metric)
for (i in 1:base::length(validation_dataset_ensemble_model_results)){validation_optimized_ensemble_summary <- dplyr::bind_cols(validation_optimized_ensemble_summary, validation_dataset_ensemble_model_results[[i]][5])}
base::colnames(validation_optimized_ensemble_summary) <- base::c("Metric", "train_optimized", "validation_optimized", "train_validation_optimized")
validation_optimized_ensemble_summary %>%
  knitr::kable()

validation_default_summary %>%
  dplyr::left_join(validation_optimized_ensemble_summary, by = "Metric") %>%
  knitr::kable()

datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
validation_default_summary %>%
  dplyr::left_join(validation_optimized_ensemble_summary, by = "Metric") %>%
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, model_name, "validation_dataset_.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Ensemble model results on test data:
base::setwd(models_store_dir)
save_option <- TRUE
optimized_cutoff <- base::c(train_ensemble_optimization_cutoff, validation_ensemble_optimization_cutoff, train_validation_ensemble_optimization_cutoff)
ensemble_model_test_predictions <- base::list(test_result_train_optimization, 
                                                    test_result_validation_optimization, 
                                                    test_result_train_validation_optimization)

test_dataset_ensemble_model_results <- base::list()
for (i in 1:base::length(ensemble_model_test_predictions)){
  test_dataset_ensemble_model_results[[i]] <- Binary_Classifier_Verification(actual = test_actual,
                                                                                   predicted = ensemble_model_test_predictions[[i]],
                                                                                   cutoff = optimized_cutoff[i],
                                                                                   type_info = base::paste(ensemble_split_types[i], "optimized_cutoff", dataset_types[3], sep = "_"),
                                                                                   save = save_option,
                                                                                   open = FALSE)[[3]]}

test_optimized_ensemble_summary <- base::data.frame(Metric = test_dataset_ensemble_model_results[[1]]$Metric)
for (i in 1:base::length(test_dataset_ensemble_model_results)){test_optimized_ensemble_summary <- dplyr::bind_cols(test_optimized_ensemble_summary, test_dataset_ensemble_model_results[[i]][5])}
base::colnames(test_optimized_ensemble_summary) <- base::c("Metric", "train_optimized", "validation_optimized", "train_validation_optimized")
test_optimized_ensemble_summary %>%
  knitr::kable()

test_default_summary %>%
  dplyr::left_join(test_optimized_ensemble_summary, by = "Metric") %>%
  knitr::kable()

datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
test_default_summary %>%
  dplyr::left_join(test_optimized_ensemble_summary, by = "Metric") %>%
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, model_name, "test_dataset_.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Final summary of ensemble model results:
ensemble_model_summary <- base::list(train_dataset = train_default_summary %>%
                                       dplyr::left_join(train_optimized_ensemble_summary, by = "Metric"),
                                     validation_dataset = validation_default_summary %>%
                                       dplyr::left_join(validation_optimized_ensemble_summary, by = "Metric"),
                                     test_dataset = test_default_summary %>%
                                       dplyr::left_join(test_optimized_ensemble_summary, by = "Metric"))

ensemble_model_summary %>%
  base::lapply(., knitr::kable)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki