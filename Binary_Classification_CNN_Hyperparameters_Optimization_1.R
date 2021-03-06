# ------------------------------------------------------------------------------
# CNN BINARY MODEL IMPLEMENTATION - HYPERPARAMETERS OPTIMIZATION (1)
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
# browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model:
model_name <- "CNN_Hyperparameters_Optimization"
model_type <- "Binary"

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
train_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/train"
validation_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/validation"
test_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/test"
models_store_dir <- paste(getwd(), model_name, model_type, sep = "/")
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"

train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Optimize model's hyperparameters:
Hyperparametrization_Optimization_Results <- Hyperparametrization_Optimization(hyperparameters_list = list(filters_1 = c(64),
                                                                                                                 filters_2 = c(64, 128),
                                                                                                                 filters_3 = c(128),
                                                                                                                 filters_4 = c(128, 256),
                                                                                                                 dense_units_1 = c(256),
                                                                                                                 dense_units_2 = c(256)),
                                  script_directory = "Binary_Classification_CNN_Hyperparameters_Optimization_2.R")

Hyperparametrization_Optimization_Results %>%
  dplyr::arrange(dplyr::desc(val_acc)) %>%
  dplyr::slice(1) %>%
  dplyr::glimpse() 

Hyperparametrization_Optimization_Results %>%
  dplyr::arrange(dplyr::desc(val_acc)) %>%
  dplyr::slice(1) %>%
  dplyr::select(Id) %>%
  dplyr::pull() %>%
  paste("logs", ., sep = "_") -> key; key

# ------------------------------------------------------------------------------
# Remove not optimal models:
setwd(models_store_dir)
saved_models <- sort(list.files(pattern = ".hdf5"))
saved_models <- saved_models[!grepl(key, saved_models)]
if (length(saved_models) > 1){
  for (j in 1:(length(saved_models))){
    cat("Remove .hdf5 file:", saved_models[j], "\n")
    unlink(saved_models[j], recursive = TRUE, force = TRUE)}}

# ------------------------------------------------------------------------------
# Remove not optimal models; logger callbacks:
saved_loggers <- sort(list.files(pattern = ".csv"))
saved_loggers <- saved_loggers[!grepl(paste(key, model_name, sep = "_"), saved_loggers)]
if (length(saved_loggers) > 1){
  for (j in 1:(length(saved_loggers))){
    cat("Remove .csv file:", saved_loggers[j], "\n")
    unlink(saved_loggers[j], recursive = TRUE, force = TRUE)}}

# ------------------------------------------------------------------------------
# Remove logs folders:
logs_folder <- paste(getwd(), list.dirs(), sep = "/")
if (length(logs_folder) > 1){
  for (j in 2:(length(logs_folder))){
    cat("Remove folder:", logs_folder[j], "\n")
    unlink(logs_folder[j], recursive = TRUE, force = TRUE)}}

# ------------------------------------------------------------------------------
# Save optimal model in local models repository: 
optimal_model <- paste(getwd(), list.files(pattern = ".hdf5"), sep = "/")
optimal_model_repo_dir <- paste(models_repo_store_dir, paste(model_type, model_name, "Model.hdf5", sep = "_"), sep = "/")
file.copy(from = optimal_model,
                to = optimal_model_repo_dir, 
                overwrite = TRUE); cat("Optimal model directory:", optimal_model_repo_dir, "\n")
unlink(optimal_model, recursive = TRUE, force = TRUE)

# ------------------------------------------------------------------------------
# Clear session and import the best trained model:
keras::k_clear_session()
optimal_model_repo_dir <- paste(models_repo_store_dir, paste(model_type, model_name, "Model.hdf5", sep = "_"), sep = "/")
model <- keras::load_model_hdf5(filepath = optimal_model_repo_dir, compile = TRUE)

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% summary()

# ------------------------------------------------------------------------------
# Model predictions using generators:
train_datagen <- keras::image_data_generator(rescale = 1/255)
train_generator <- keras::flow_images_from_directory(directory = train_dir,
                                                     generator = train_datagen, 
                                                     target_size = c(image_size, image_size),
                                                     batch_size = batch_size,
                                                     class_mode = class_mode,
                                                     classes = levels(validation_files$category),
                                                     shuffle = FALSE)

validation_datagen <- keras::image_data_generator(rescale = 1/255)
validation_generator <- keras::flow_images_from_directory(directory = validation_dir,
                                                          generator = validation_datagen,
                                                          target_size = c(image_size, image_size),
                                                          batch_size = batch_size,
                                                          class_mode = class_mode,
                                                          classes = levels(validation_files$category),
                                                          shuffle = FALSE)

test_datagen <- keras::image_data_generator(rescale = 1/255)
test_generator <- keras::flow_images_from_directory(directory = test_dir,
                                                    generator = test_datagen,
                                                    target_size = c(image_size, image_size),
                                                    batch_size = batch_size,
                                                    class_mode = class_mode,
                                                    shuffle = FALSE)

train_evaluation <- keras::evaluate_generator(model, train_generator, steps = ceiling(train_generator$n/train_generator$batch_size)); train_evaluation
validation_evaluation <- keras::evaluate_generator(model, validation_generator, steps = ceiling(validation_generator$n/validation_generator$batch_size)); validation_evaluation
test_evaluation <- keras::evaluate_generator(model, test_generator, steps = ceiling(test_generator$n/test_generator$batch_size)); test_evaluation 

train_probabilities <- keras::predict_generator(model, train_generator, steps = ceiling(train_generator$n/train_generator$batch_size), verbose = 1)
validation_probabilities <- keras::predict_generator(model, validation_generator, steps = ceiling(validation_generator$n/validation_generator$batch_size), verbose = 1)
test_probabilities <- keras::predict_generator(model, test_generator, steps = ceiling(test_generator$n/test_generator$batch_size), verbose = 1)

setwd(models_store_dir)
readr::write_csv2(tibble::as_tibble(train_probabilities) %>%
                    dplyr::mutate(filepath = train_generator$filepaths,
                                  actual_class = train_generator$classes,
                                  model = model_name),
                  paste(stringr::str_replace_all(Sys.time(), ":", "-"), model_name, "train_binary_probabilities.csv", sep = "_"))
readr::write_csv2(tibble::as_tibble(validation_probabilities) %>%
                    dplyr::mutate(filepath = validation_generator$filepaths,
                                  actual_class = validation_generator$classes,
                                  model = model_name),
                  paste(stringr::str_replace_all(Sys.time(), ":", "-"), model_name, "validation_binary_probabilities.csv", sep = "_"))
readr::write_csv2(tibble::as_tibble(test_probabilities) %>%
                    dplyr::mutate(filepath = test_generator$filepaths,
                                  actual_class = test_generator$classes,
                                  model = model_name), 
                  paste(stringr::str_replace_all(Sys.time(), ":", "-"), model_name, "test_binary_probabilities.csv", sep = "_"))

# ------------------------------------------------------------------------------
# Model verification - default cutoff:
default_cutoff <- 0.5
save_option <- TRUE

train_verification_1 <- Binary_Classifier_Verification(actual = train_generator$classes,
                                                       predicted = train_probabilities[,2],
                                                       cutoff = default_cutoff,
                                                       type_info = paste(model_name, "default_cutoff", "train", sep = "_"),
                                                       save = save_option,
                                                       open = FALSE)

validation_verification_1 <- Binary_Classifier_Verification(actual = validation_generator$classes,
                                                            predicted = validation_probabilities[,2],
                                                            cutoff = default_cutoff,
                                                            type_info = paste(model_name, "default_cutoff", "validation", sep = "_"),
                                                            save = save_option,
                                                            open = FALSE)

test_verification_1 <- Binary_Classifier_Verification(actual = test_generator$classes,
                                                      predicted = test_probabilities[,2],
                                                      cutoff = default_cutoff,
                                                      type_info = paste(model_name, "default_cutoff", "test", sep = "_"),
                                                      save = save_option,
                                                      open = FALSE)

final_score_1 <- train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  knitr::kable(.); final_score_1

datetime <- stringr::str_replace_all(Sys.time(), ":", "-")
train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  readr::write_csv2(path = paste(models_store_dir, paste(datetime, model_name, "binary_classification_summary_default_cutoff.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Model verification - cutoff optimization:
save_option <- TRUE
train_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = train_generator$classes,
                                                                   predicted = train_probabilities[,2],
                                                                   type_info = paste(model_name, "train", sep = "_"),
                                                                   seed_value = 42,
                                                                   top = 10,
                                                                   cuts = 100,
                                                                   key_metric = ACC,
                                                                   ascending = FALSE,
                                                                   save = save_option,
                                                                   open = FALSE)
train_cutoff_optimization$top_cutoffs %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  mean() -> train_optimal_cutoff; train_optimal_cutoff

validation_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = validation_generator$classes,
                                                                        predicted = validation_probabilities[,2],
                                                                        type_info = paste(model_name, "validation", sep = "_"),
                                                                        seed_value = 42,
                                                                        top = 10,
                                                                        cuts = 100,
                                                                        key_metric = ACC,
                                                                        ascending = FALSE,
                                                                        save = save_option,
                                                                        open = FALSE)
validation_cutoff_optimization$top_cutoffs %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  mean() -> validation_optimal_cutoff; validation_optimal_cutoff

train_validation_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = c(train_generator$classes, validation_generator$classes),
                                                                              predicted = c(train_probabilities[,2], validation_probabilities[,2]),
                                                                              type_info = paste(model_name, "train", "validation", sep = "_"),
                                                                              seed_value = 42,
                                                                              top = 10,
                                                                              cuts = 100,
                                                                              key_metric = ACC,
                                                                              ascending = FALSE,
                                                                              save = save_option,
                                                                              open = FALSE)

train_validation_cutoff_optimization$top_cutoffs %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  mean() -> train_validation_optimal_cutoff; train_validation_optimal_cutoff

# Select cutoff:
# * train_optimal_cutoff
# * validation_optimal_cutoff
# * train_validation_optimal_cutoff
selected_cutoff <- validation_optimal_cutoff
save_option <- TRUE

train_verification_2 <- Binary_Classifier_Verification(actual = train_generator$classes,
                                                       predicted = train_probabilities[,2],
                                                       cutoff = selected_cutoff,
                                                       type_info = paste(model_name, "optimized_cutoff", "train", sep = "_"),
                                                       save = save_option,
                                                       open = FALSE)

validation_verification_2 <- Binary_Classifier_Verification(actual = validation_generator$classes,
                                                            predicted = validation_probabilities[,2],
                                                            cutoff = selected_cutoff,
                                                            type_info = paste(model_name, "optimized_cutoff", "validation", sep = "_"),
                                                            save = save_option,
                                                            open = FALSE)

test_verification_2 <- Binary_Classifier_Verification(actual = test_generator$classes,
                                                      predicted = test_probabilities[,2],
                                                      cutoff = selected_cutoff,
                                                      type_info = paste(model_name, "optimized_cutoff", "test", sep = "_"),
                                                      save = save_option,
                                                      open = FALSE)

final_score_2 <- train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = selected_cutoff) %>%
  knitr::kable(.); final_score_2

datetime <- stringr::str_replace_all(Sys.time(), ":", "-")
train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = selected_cutoff) %>%
  readr::write_csv2(path = paste(models_store_dir, paste(datetime, model_name, "binary_classification_summary_optimized_cutoff.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Final summary - cutoff summary comparison:
final_score_1_summary <- train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score); final_score_1_summary

final_score_2_summary <- train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score); final_score_2_summary

`%!in%` = Negate(`%in%`)

final_score_1_summary %>%
  dplyr::left_join(final_score_2_summary, by = "Metric") %>%
  dplyr::rename(Train_default = Score_train.x,
                Validation_default = Score_validation.x,
                Test_default = Score_test.x,
                Train_optimized = Score_train.y,
                Validation_optimized = Score_validation.y,
                Test_optimized = Score_test.y) %>%
  dplyr::mutate(Train_diff = Train_optimized - Train_default,
                Validation_diff = Validation_optimized - Validation_default,
                Test_diff = Test_optimized - Test_default,
                Cutoff = selected_cutoff) %>%
  knitr::kable(.)

datetime <- stringr::str_replace_all(Sys.time(), ":", "-")
final_score_1_summary %>%
  dplyr::left_join(final_score_2_summary, by = "Metric") %>%
  dplyr::rename(Train_default = Score_train.x,
                Validation_default = Score_validation.x,
                Test_default = Score_test.x,
                Train_optimized = Score_train.y,
                Validation_optimized = Score_validation.y,
                Test_optimized = Score_test.y) %>%
  dplyr::mutate(Train_diff = Train_optimized - Train_default,
                Validation_diff = Validation_optimized - Validation_default,
                Test_diff = Test_optimized - Test_default,
                Cutoff = selected_cutoff) %>%
  readr::write_csv2(path = paste(models_store_dir, paste(datetime, model_name, "binary_classification_cutoff_summary_comparison.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Predict indicated image:
labels <- sort(as.character(train_files$category)); labels
set <- "train"
category <- "dogs"  
id <- 1

Predict_Image(image_path = paste("D:/GitHub/Datasets/Cats_And_Dogs", set, category, list.files(paste("D:/GitHub/Datasets/Cats_And_Dogs", set, category, sep = "/")), sep = "/")[id],
              model = model,
              classes = labels,
              plot_image = TRUE)

# ------------------------------------------------------------------------------
# Save correct and incorrect predictions:
save_summary_files <- TRUE
save_correct_images <- FALSE
save_incorrect_images <- FALSE

# Train:
Train_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = train_dir,
                                                                                                    actual = train_generator$classes,
                                                                                                    predicted = train_probabilities[,2],
                                                                                                    type_info = model_name,
                                                                                                    cwd = models_store_dir,
                                                                                                    cutoff = 0.5,
                                                                                                    save_summary_files = save_summary_files,
                                                                                                    save_correct_images = save_correct_images,
                                                                                                    save_incorrect_images = save_incorrect_images)

# Validation:
Validation_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = validation_dir,
                                                                                                         actual = validation_generator$classes,
                                                                                                         predicted = validation_probabilities[,2],
                                                                                                         type_info = model_name,
                                                                                                         cwd = models_store_dir,
                                                                                                         cutoff = 0.5,
                                                                                                         save_summary_files = save_summary_files,
                                                                                                         save_correct_images = save_correct_images,
                                                                                                         save_incorrect_images = save_incorrect_images)

# Test:
Test_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = test_dir,
                                                                                                   actual = test_generator$classes,
                                                                                                   predicted = test_probabilities[,2],
                                                                                                   type_info = model_name,
                                                                                                   cwd = models_store_dir,
                                                                                                   cutoff = 0.5,
                                                                                                   save_summary_files = save_summary_files,
                                                                                                   save_correct_images = save_correct_images,
                                                                                                   save_incorrect_images = save_incorrect_images)

# ------------------------------------------------------------------------------
# Visualize predictions distribution:
save_plot <- TRUE
labels <- sort(as.character(train_files$category)); labels

train_predicted_2 <- train_probabilities[matrix(data = c(1:nrow(train_probabilities), train_generator$classes + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = train_generator$classes,
                                              predicted = train_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = paste(model_name, "train", sep = "_"),
                                              save_plot = save_plot)

validation_predicted_2 <- validation_probabilities[matrix(data = c(1:nrow(validation_probabilities), validation_generator$classes + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = validation_generator$classes,
                                              predicted = validation_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = paste(model_name, "validation", sep = "_"),
                                              save_plot = save_plot)

test_predicted_2 <- test_probabilities[matrix(data = c(1:nrow(test_probabilities), test_generator$classes + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = test_generator$classes,
                                              predicted = test_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = paste(model_name, "test", sep = "_"),
                                              save_plot = save_plot)

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to all classes:
save_plot <- TRUE

Display_All_Classes_Predictions_Distribution(actual = train_generator$classes + 1,
                                             predicted = train_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = paste(model_name, "train", sep = "_"),
                                             save_plot = save_plot)

Display_All_Classes_Predictions_Distribution(actual = validation_generator$classes + 1,
                                             predicted = validation_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = paste(model_name, "validation", sep = "_"),
                                             save_plot = save_plot)

Display_All_Classes_Predictions_Distribution(actual = test_generator$classes + 1,
                                             predicted = test_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = paste(model_name, "test", sep = "_"),
                                             save_plot = save_plot)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki