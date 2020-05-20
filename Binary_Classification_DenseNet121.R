# ------------------------------------------------------------------------------
# DENSENET121 BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "DenseNet121"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'DenseNet121' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))
# 3. Create 'Binary' subfolder in 'DenseNet121' main folder
base::dir.create(path = base::paste(base::getwd(), model_name, "Binary", sep = "/"))

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
models_store_dir <- base::paste(base::getwd(), model_name, "Binary", sep = "/")
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"
callback_model_checkpoint_path <- base::paste(models_store_dir, "keras_model.weights.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5", sep = "/")
callback_tensorboard_path <- base::paste(models_store_dir, "logs", sep = "/")
callback_csv_logger_path <- base::paste(models_store_dir, base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, "model_optimization_logger.csv", sep = "_"), sep = "/")

train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Setting pipeline parameters values: 
# Image:
image_size <- 224
channels <- 3

# Model structure:
weights <- "imagenet"
include_top <- FALSE
activation <- "softmax"

# Model compilation:
loss <- "categorical_crossentropy"
optimizer <- keras::optimizer_adam()
metrics <- base::c("acc")

# Training:
batch_size <- 16
class_mode <- "categorical"
shuffle <- TRUE
epochs <- 1
early_stopping_patience <- 10
reduce_lr_on_plateu_patience <- 5
monitor <- "val_acc"
save_best_only <- TRUE
if (monitor == "val_loss"){mode <- "min"} else {mode <- "max"}
verbose <- 1
write_graph <- TRUE
write_grads <- TRUE
write_images <- TRUE
restore_best_weights <- FALSE
histogram_freq <- 1
min_delta <- 0

# ------------------------------------------------------------------------------
# DenseNet121 model architecture:
model <- keras::application_densenet121(include_top = include_top,
                                        weights = weights,
                                        input_shape = base::c(image_size, image_size, channels))

input_tensor <- keras::layer_input(shape = base::c(image_size, image_size, channels))
output_tensor <- input_tensor %>%
  model %>%
  keras::layer_global_average_pooling_2d() %>%
  keras::layer_dense(units = base::length(base::levels(validation_files$category)), activation = activation) 

model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)

# ------------------------------------------------------------------------------
# Upload pre-trained model for training:
# last_model <- base::list.files(path = models_store_dir, pattern = ".hdf5")[base::length(base::list.files(path = models_store_dir, pattern = ".hdf5"))]
# model <- keras::load_model_hdf5(filepath = paste(models_store_dir, last_model, sep = "/"), compile = FALSE)

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Model compilation:
model %>% keras::compile(loss = loss,
                         optimizer = optimizer, 
                         metrics = metrics)

# ------------------------------------------------------------------------------
# Generators:
train_datagen <- keras::image_data_generator(featurewise_center = FALSE,
                                             samplewise_center = FALSE,
                                             featurewise_std_normalization = FALSE,
                                             samplewise_std_normalization = FALSE,
                                             zca_whitening = FALSE,
                                             zca_epsilon = 1e-06,
                                             rotation_range = 0,
                                             width_shift_range = 0,
                                             height_shift_range = 0,
                                             brightness_range = base::c(1, 1),
                                             shear_range = 0,
                                             zoom_range = 0,
                                             channel_shift_range = 0,
                                             fill_mode = "nearest",
                                             cval = 0,
                                             horizontal_flip = FALSE,
                                             vertical_flip = FALSE,
                                             rescale = 1/255,
                                             preprocessing_function = NULL,
                                             data_format = NULL,
                                             validation_split = 0)
train_generator <- keras::flow_images_from_directory(directory = train_dir,
                                                     generator = train_datagen, 
                                                     target_size = base::c(image_size, image_size),
                                                     batch_size = batch_size,
                                                     class_mode = class_mode,
                                                     classes = base::levels(validation_files$category),
                                                     shuffle = shuffle)

validation_datagen <- keras::image_data_generator(rescale = 1/255) 
validation_generator <- keras::flow_images_from_directory(directory = validation_dir,
                                                          generator = validation_datagen,
                                                          target_size = base::c(image_size, image_size),
                                                          batch_size = batch_size,
                                                          class_mode = class_mode,
                                                          classes = base::levels(validation_files$category),
                                                          shuffle = shuffle)

# ------------------------------------------------------------------------------
# Tensorboard:
base::dir.create(path = callback_tensorboard_path)
keras::tensorboard(log_dir = callback_tensorboard_path, host = "127.0.0.1")
# If 'ERROR: invalid version specification':
# 1. Anaconda Prompt
# 2. conda activate GPU_ML_2
# 3. cd D:/GitHub/DeepNeuralNetworksRepoR/DenseNet121/Binary
# 4. tensorboard --logdir=logs --host=127.0.0.1
# 5. http://127.0.0.1:6006/
# 6. Start model optimization
# 7. F5 http://127.0.0.1:6006/ to examine the latest results

# ------------------------------------------------------------------------------
# Model optimization:
history <- model %>% keras::fit_generator(generator = train_generator,
                                          steps_per_epoch = base::ceiling(base::sum(train_files$category_obs)/train_generator$batch_size), 
                                          epochs = epochs,
                                          validation_data = validation_generator,
                                          validation_steps = base::ceiling(base::sum(validation_files$category_obs)/train_generator$batch_size), 
                                          callbacks = base::list(keras::callback_model_checkpoint(filepath = callback_model_checkpoint_path,
                                                                                                  monitor = monitor,
                                                                                                  verbose = verbose,
                                                                                                  save_best_only = save_best_only),
                                                                 keras::callback_early_stopping(monitor = monitor,
                                                                                                min_delta = min_delta,
                                                                                                verbose = verbose,
                                                                                                patience = early_stopping_patience,
                                                                                                restore_best_weights = restore_best_weights),
                                                                 keras::callback_reduce_lr_on_plateau(monitor = monitor,
                                                                                                      factor = 0.1,
                                                                                                      patience = reduce_lr_on_plateu_patience,
                                                                                                      verbose = verbose),
                                                                 keras::callback_tensorboard(log_dir = callback_tensorboard_path,
                                                                                             histogram_freq = histogram_freq,
                                                                                             write_graph = write_graph,
                                                                                             write_grads = write_grads,
                                                                                             write_images = write_images),
                                                                 keras::callback_csv_logger(filename = callback_csv_logger_path,
                                                                                            separator = ";",
                                                                                            append = TRUE)))
history$metrics %>%
  tibble::as_tibble() %>%
  dplyr::mutate(epoch = dplyr::row_number()) %>%
  base::as.data.frame() %>%
  knitr::kable(.)

# ------------------------------------------------------------------------------
# Remove not optimal models:
base::setwd(models_store_dir)
saved_models <- base::sort(base::list.files(pattern = ".hdf5"))
if (length(saved_models) > 1){
  for (j in 1:(base::length(saved_models) - 1)){
    base::cat("Remove .hdf5 file:", saved_models[j], "\n")
    base::unlink(saved_models[j], recursive = TRUE, force = TRUE)}}

# ------------------------------------------------------------------------------
# Remove logs folder:
logs_folder <- base::paste(base::getwd(), base::list.files(pattern = "logs"), sep = "/")
base::unlink(logs_folder, force = TRUE, recursive = TRUE)

# ------------------------------------------------------------------------------
# Save optimal model in local models repository: 
optimal_model <- base::paste(base::getwd(), base::list.files(pattern = ".hdf5"), sep = "/")
optimal_model_repo_dir <- base::paste(models_repo_store_dir, base::paste("Binary", model_name, "Model.hdf5", sep = "_"), sep = "/")
base::file.copy(from = optimal_model,
                to = optimal_model_repo_dir, 
                overwrite = TRUE); base::cat("Optimal model directory:", optimal_model_repo_dir, "\n")
base::unlink(optimal_model, recursive = TRUE, force = TRUE)

# ------------------------------------------------------------------------------
# Clear session and import the best trained model:
keras::k_clear_session()
optimal_model_repo_dir <- base::paste(models_repo_store_dir, base::paste("Binary", model_name, "Model.hdf5", sep = "_"), sep = "/")
model <- keras::load_model_hdf5(filepath = optimal_model_repo_dir, compile = FALSE)
model %>% keras::compile(loss = loss,
                         optimizer = optimizer, 
                         metrics = metrics)

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Model predictions using generators:
train_datagen <- keras::image_data_generator(rescale = 1/255)
train_generator <- keras::flow_images_from_directory(directory = train_dir,
                                                     generator = train_datagen, 
                                                     target_size = base::c(image_size, image_size),
                                                     batch_size = batch_size,
                                                     class_mode = class_mode,
                                                     classes = base::levels(validation_files$category),
                                                     shuffle = FALSE)

validation_datagen <- keras::image_data_generator(rescale = 1/255)
validation_generator <- keras::flow_images_from_directory(directory = validation_dir,
                                                          generator = validation_datagen,
                                                          target_size = base::c(image_size, image_size),
                                                          batch_size = batch_size,
                                                          class_mode = class_mode,
                                                          classes = base::levels(validation_files$category),
                                                          shuffle = FALSE)

test_datagen <- keras::image_data_generator(rescale = 1/255)
test_generator <- keras::flow_images_from_directory(directory = test_dir,
                                                    generator = test_datagen,
                                                    target_size = base::c(image_size, image_size),
                                                    batch_size = batch_size,
                                                    class_mode = class_mode,
                                                    shuffle = FALSE)

train_evaluation <- keras::evaluate_generator(model, train_generator, steps = base::ceiling(base::sum(train_files$category_obs)/train_generator$batch_size)); train_evaluation
validation_evaluation <- keras::evaluate_generator(model, validation_generator, steps = base::ceiling(base::sum(validation_files$category_obs)/validation_generator$batch_size)); validation_evaluation
test_evaluation <- keras::evaluate_generator(model, test_generator, steps = base::ceiling(base::sum(test_files$category_obs)/test_generator$batch_size)); test_evaluation

train_probabilities <- keras::predict_generator(model, train_generator, steps = base::ceiling(base::sum(train_files$category_obs)/train_generator$batch_size), verbose = 1)
validation_probabilities <- keras::predict_generator(model, validation_generator, steps = base::ceiling(base::sum(validation_files$category_obs)/validation_generator$batch_size), verbose = 1)
test_probabilities <- keras::predict_generator(model, test_generator, steps = base::ceiling(base::sum(test_files$category_obs)/test_generator$batch_size), verbose = 1)

base::setwd(models_store_dir)
readr::write_csv2(tibble::as_tibble(train_probabilities) %>%
                    dplyr::mutate(filepath = train_generator$filepaths,
                                  actual_class = train_generator$classes,
                                  model = model_name),
                  base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, "train_binary_probabilities.csv", sep = "_"))
readr::write_csv2(tibble::as_tibble(validation_probabilities) %>%
                    dplyr::mutate(filepath = validation_generator$filepaths,
                                  actual_class = validation_generator$classes,
                                  model = model_name),
                  base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, "validation_binary_probabilities.csv", sep = "_"))
readr::write_csv2(tibble::as_tibble(test_probabilities) %>%
                    dplyr::mutate(filepath = test_generator$filepaths,
                                  actual_class = test_generator$classes,
                                  model = model_name), 
                  base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, "test_binary_probabilities.csv", sep = "_"))

# ------------------------------------------------------------------------------
# Model verification - default cutoff:
default_cutoff <- 0.5
save_option <- TRUE

train_verification_1 <- Binary_Classifier_Verification(actual = train_generator$classes,
                                                       predicted = train_probabilities[,2],
                                                       cutoff = default_cutoff,
                                                       type_info = base::paste(model_name, "default_cutoff", "train", sep = "_"),
                                                       save = save_option,
                                                       open = FALSE)

validation_verification_1 <- Binary_Classifier_Verification(actual = validation_generator$classes,
                                                            predicted = validation_probabilities[,2],
                                                            cutoff = default_cutoff,
                                                            type_info = base::paste(model_name, "default_cutoff", "validation", sep = "_"),
                                                            save = save_option,
                                                            open = FALSE)

test_verification_1 <- Binary_Classifier_Verification(actual = test_generator$classes,
                                                      predicted = test_probabilities[,2],
                                                      cutoff = default_cutoff,
                                                      type_info = base::paste(model_name, "default_cutoff", "test", sep = "_"),
                                                      save = save_option,
                                                      open = FALSE)

final_score_1 <- train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  knitr::kable(.); final_score_1

datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, model_name, "binary_classification_summary_default_cutoff.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Model verification - cutoff optimization:
save_option <- TRUE
train_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = train_generator$classes,
                                                                   predicted = train_probabilities[,2],
                                                                   type_info = base::paste(model_name, "train", sep = "_"),
                                                                   seed_value = 42,
                                                                   top = 10,
                                                                   cuts = 100,
                                                                   key_metric = ACC,
                                                                   ascending = FALSE,
                                                                   save = save_option,
                                                                   open = FALSE)
train_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> train_optimal_cutoff; train_optimal_cutoff

validation_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = validation_generator$classes,
                                                                        predicted = validation_probabilities[,2],
                                                                        type_info = base::paste(model_name, "validation", sep = "_"),
                                                                        seed_value = 42,
                                                                        top = 10,
                                                                        cuts = 100,
                                                                        key_metric = ACC,
                                                                        ascending = FALSE,
                                                                        save = save_option,
                                                                        open = FALSE)
validation_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> validation_optimal_cutoff; validation_optimal_cutoff

train_validation_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = base::c(train_generator$classes, validation_generator$classes),
                                                                              predicted = base::c(train_probabilities[,2], validation_probabilities[,2]),
                                                                              type_info = base::paste(model_name, "train", "validation", sep = "_"),
                                                                              seed_value = 42,
                                                                              top = 10,
                                                                              cuts = 100,
                                                                              key_metric = ACC,
                                                                              ascending = FALSE,
                                                                              save = save_option,
                                                                              open = FALSE)

train_validation_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> train_validation_optimal_cutoff; train_validation_optimal_cutoff

# Select cutoff:
# * train_optimal_cutoff
# * validation_optimal_cutoff
# * train_validation_optimal_cutoff
selected_cutoff <- validation_optimal_cutoff
save_option <- TRUE

train_verification_2 <- Binary_Classifier_Verification(actual = train_generator$classes,
                                                       predicted = train_probabilities[,2],
                                                       cutoff = selected_cutoff,
                                                       type_info = base::paste(model_name, "optimized_cutoff", "train", sep = "_"),
                                                       save = save_option,
                                                       open = FALSE)

validation_verification_2 <- Binary_Classifier_Verification(actual = validation_generator$classes,
                                                            predicted = validation_probabilities[,2],
                                                            cutoff = selected_cutoff,
                                                            type_info = base::paste(model_name, "optimized_cutoff", "validation", sep = "_"),
                                                            save = save_option,
                                                            open = FALSE)

test_verification_2 <- Binary_Classifier_Verification(actual = test_generator$classes,
                                                      predicted = test_probabilities[,2],
                                                      cutoff = selected_cutoff,
                                                      type_info = base::paste(model_name, "optimized_cutoff", "test", sep = "_"),
                                                      save = save_option,
                                                      open = FALSE)

final_score_2 <- train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = selected_cutoff) %>%
  knitr::kable(.); final_score_2

datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = selected_cutoff) %>%
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, model_name, "binary_classification_summary_optimized_cutoff.csv", sep = "_"), sep = "/"))

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

`%!in%` = base::Negate(`%in%`)

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

datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
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
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, model_name, "binary_classification_cutoff_summary_comparison.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# Predict indicated image:
labels <- base::sort(base::as.character(train_files$category)); labels
set <- "train"
category <- "dogs"  
id <- 4

Predict_Image(image_path = base::paste("D:/GitHub/Datasets/Cats_And_Dogs", set, category, base::list.files(base::paste("D:/GitHub/Datasets/Cats_And_Dogs", set, category, sep = "/")), sep = "/")[id],
              model = model,
              classes = labels,
              plot_image = TRUE)

# ------------------------------------------------------------------------------
# Save correct and incorrect predictions:
save_summary_files <- TRUE
save_correct_images <- FALSE
save_incorrect_images <- FALSE

# Train:
Train_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/train",
                                                                                                    actual = train_generator$classes,
                                                                                                    predicted = train_probabilities[,2],
                                                                                                    type_info = model_name,
                                                                                                    cwd = models_store_dir,
                                                                                                    cutoff = 0.5,
                                                                                                    save_summary_files = save_summary_files,
                                                                                                    save_correct_images = save_correct_images,
                                                                                                    save_incorrect_images = save_incorrect_images)

# Validation:
Validation_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/validation",
                                                                                                         actual = validation_generator$classes,
                                                                                                         predicted = validation_probabilities[,2],
                                                                                                         type_info = model_name,
                                                                                                         cwd = models_store_dir,
                                                                                                         cutoff = 0.5,
                                                                                                         save_summary_files = save_summary_files,
                                                                                                         save_correct_images = save_correct_images,
                                                                                                         save_incorrect_images = save_incorrect_images)

# Test:
Test_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/test",
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
labels <- base::sort(base::as.character(train_files$category)); labels

train_predicted_2 <- train_probabilities[base::matrix(data = base::c(1:base::nrow(train_probabilities), train_generator$classes + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = train_generator$classes,
                                              predicted = train_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = base::paste(model_name, "train", sep = "_"),
                                              save_plot = save_plot)

validation_predicted_2 <- validation_probabilities[base::matrix(data = base::c(1:base::nrow(validation_probabilities), validation_generator$classes + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = validation_generator$classes,
                                              predicted = validation_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = base::paste(model_name, "validation", sep = "_"),
                                              save_plot = save_plot)

test_predicted_2 <- test_probabilities[base::matrix(data = base::c(1:base::nrow(test_probabilities), test_generator$classes + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = test_generator$classes,
                                              predicted = test_predicted_2,
                                              labels = labels,
                                              bins = 10,
                                              type_info = base::paste(model_name, "test", sep = "_"),
                                              save_plot = save_plot)

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to all classes:
save_plot <- TRUE

Display_All_Classes_Predictions_Distribution(actual = train_generator$classes + 1,
                                             predicted = train_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = base::paste(model_name, "train", sep = "_"),
                                             save_plot = save_plot)

Display_All_Classes_Predictions_Distribution(actual = validation_generator$classes + 1,
                                             predicted = validation_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = base::paste(model_name, "validation", sep = "_"),
                                             save_plot = save_plot)

Display_All_Classes_Predictions_Distribution(actual = test_generator$classes + 1,
                                             predicted = test_probabilities,
                                             labels = labels,
                                             bins = 10,
                                             type_info = base::paste(model_name, "test", sep = "_"),
                                             save_plot = save_plot)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki