# ------------------------------------------------------------------------------
# DENSENET169 CATEGORICAL MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/cifar-10/overview
utils::browseURL(url = "https://www.kaggle.com/c/cifar-10/overview")

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'DenseNet169' folder in cwd
base::dir.create(path = base::paste(base::getwd(), "DenseNet169", sep = "/"))
# 3. Create 'Categorical' subfolder in 'DenseNet169' main folder
base::dir.create(path = base::paste(base::getwd(), "DenseNet169", "Categorical", sep = "/"))

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
# keras::install_keras(tensorflow = "gpu")
base::library(tidyverse)
base::library(deepviz)
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Useful_Functions.R")

train_dir <- "D:/GitHub/Datasets/Cifar10/train"
validation_dir <- "D:/GitHub/Datasets/Cifar10/validation"
test_dir <- "D:/GitHub/Datasets/Cifar10/test"
models_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR/DenseNet169/Categorical"
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"
callback_model_checkpoint_path <- base::paste(models_store_dir, "keras_model.weights.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5", sep = "/")
callback_tensorboard_path <- base::paste(models_store_dir, "logs", sep = "/")
callback_csv_logger_path <- base::paste(models_store_dir, "Optimization_logger.csv", sep = "/")

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
# DenseNet169 model architecture:
model <- keras::application_densenet169(include_top = include_top,
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
# 3. cd D:\GitHub\DeepNeuralNetworksRepoR\DenseNet169\Categorical
# 4. tensorboard --logdir=logs --host=127.0.0.1
# 5. http://127.0.0.1:6006/
# 6. Start model optimization
# 7. F5 http://127.0.0.1:6006/ to examine the latest results

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
saved_models <- base::sort(base::list.files()[base::grepl(".hdf5", base::list.files())])
if (length(saved_models) > 1){
  for (j in 1:(base::length(saved_models) - 1)){
    base::cat("Remove .hdf5 file:", saved_models[j], "\n")
    base::unlink(saved_models[j], recursive = TRUE, force = TRUE)}}

# ------------------------------------------------------------------------------
# Remove logs folder:
logs_folder <- base::paste(base::getwd(), base::list.files()[base::grepl("logs", base::list.files())], sep = "/")
base::unlink(logs_folder, force = TRUE, recursive = TRUE)

# ------------------------------------------------------------------------------
# Save optimal model in local models repository: 
optimal_model <- base::paste(base::getwd(), base::list.files(pattern = ".hdf5"), sep = "/")
optimal_model_repo_dir <- base::paste(models_repo_store_dir, "Categorical_DenseNet169_Model.hdf5", sep = "/")
base::file.copy(from = optimal_model,
                to = optimal_model_repo_dir, 
                overwrite = TRUE); base::cat("Optimal model directory:", optimal_model_repo_dir, "\n")
base::unlink(optimal_model, recursive = TRUE, force = TRUE)

# ------------------------------------------------------------------------------
# Clear session and import the best trained model:
keras::k_clear_session()
optimal_model_repo_dir <- base::paste(models_repo_store_dir, "Categorical_DenseNet169_Model.hdf5", sep = "/")
model <- keras::load_model_hdf5(filepath = optimal_model_repo_dir, compile = FALSE)
model %>% keras::compile(loss = loss,
                         optimizer = optimizer, 
                         metrics = metrics)

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Model evaluation and predictions using generators:
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
datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
readr::write_csv2(tibble::as_tibble(train_probabilities) %>%
                    dplyr::mutate(filepath = train_generator$filepaths), base::paste(datetime, "DenseNet169_train_categorical_probabilities.csv"))
readr::write_csv2(tibble::as_tibble(validation_probabilities) %>%
                    dplyr::mutate(filepath = validation_generator$filepaths), base::paste(datetime, "DenseNet169_validation_categorical_probabilities.csv"))
readr::write_csv2(tibble::as_tibble(test_probabilities) %>%
                    dplyr::mutate(filepath = test_generator$filepaths), base::paste(datetime, "DenseNet169_test_categorical_probabilities.csv"))

# ------------------------------------------------------------------------------
# Model verification:
labels <- base::sort(base::as.character(train_files$category)); labels
train_actual <- base::rep(x = 1:base::length(train_files$category), times = train_files$category_obs); train_actual
validation_actual <- base::rep(x = 1:base::length(validation_files$category), times = validation_files$category_obs); validation_actual
test_actual <- base::rep(x = 1:base::length(test_files$category), times = test_files$category_obs); test_actual

Categorical_train_results <- Categorical_Model_Evaluation(actual = train_actual,
                                                          probabilities = train_probabilities,
                                                          labels = labels,
                                                          type_info = "Train DenseNet169",
                                                          save = TRUE,
                                                          open = FALSE)

Categorical_validation_results <- Categorical_Model_Evaluation(actual = validation_actual,
                                                               probabilities = validation_probabilities,
                                                               labels = labels,
                                                               type_info = "Validation DenseNet169",
                                                               save = TRUE,
                                                               open = FALSE)

Categorical_test_results <- Categorical_Model_Evaluation(actual = test_actual,
                                                         probabilities = test_probabilities,
                                                         labels = labels,
                                                         type_info = "Test DenseNet169",
                                                         save = TRUE,
                                                         open = FALSE)

# ------------------------------------------------------------------------------
# Predict indicated image:
labels <- base::sort(base::as.character(train_files$category)); labels
set <- "train"
category <- "automobile"  
id <- 3

Predict_Image(image_path = base::paste("D:/GitHub/Datasets/Cifar10", set, category, base::list.files(base::paste("D:/GitHub/Datasets/Cifar10", set, category, sep = "/")), sep = "/")[id],
              model = model,
              classes = labels,
              plot_image = TRUE)

# ------------------------------------------------------------------------------
# Save true and false predictions:
# Train:
Train_Correct_Incorrect_Categorical_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = "D:/GitHub/Datasets/Cifar10/train",
                                                                                                              actual_classes = train_actual,
                                                                                                              prediction = train_probabilities,
                                                                                                              cwd = models_store_dir,
                                                                                                              save_summary_files = TRUE,
                                                                                                              save_correct_images = FALSE,
                                                                                                              save_incorrect_images = FALSE)

# Validation:
Validation_Correct_Incorrect_Categorical_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = "D:/GitHub/Datasets/Cifar10/validation",
                                                                                                                   actual_classes = validation_actual,
                                                                                                                   prediction = validation_probabilities,
                                                                                                                   cwd = models_store_dir,
                                                                                                                   save_summary_files = TRUE,
                                                                                                                   save_correct_images = FALSE,
                                                                                                                   save_incorrect_images = FALSE)

# Test:
Test_Correct_Incorrect_Categorical_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = "D:/GitHub/Datasets/Cifar10/test",
                                                                                                             actual_classes = test_actual,
                                                                                                             prediction = test_probabilities,
                                                                                                             cwd = models_store_dir,
                                                                                                             save_summary_files = TRUE,
                                                                                                             save_correct_images = FALSE,
                                                                                                             save_incorrect_images = FALSE)

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to target classes:
train_predicted <- train_probabilities[base::matrix(data = base::c(1:base::nrow(train_probabilities), train_actual), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = train_actual,
                                              predicted = train_predicted,
                                              labels = labels,
                                              bins = 10)

validation_predicted <- validation_probabilities[base::matrix(data = base::c(1:base::nrow(validation_probabilities), validation_actual), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = validation_actual,
                                              predicted = validation_predicted,
                                              labels = labels,
                                              bins = 10)

test_predicted <- test_probabilities[base::matrix(data = base::c(1:base::nrow(test_probabilities), test_actual), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = test_actual,
                                              predicted = test_predicted,
                                              labels = labels,
                                              bins = 10)

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to all classes:
Display_All_Classes_Predictions_Distribution(actual = train_actual,
                                             predicted = train_probabilities,
                                             labels = labels,
                                             bins = 5)

Display_All_Classes_Predictions_Distribution(actual = validation_actual,
                                             predicted = validation_probabilities,
                                             labels = labels,
                                             bins = 5)

Display_All_Classes_Predictions_Distribution(actual = test_actual,
                                             predicted = test_probabilities,
                                             labels = labels,
                                             bins = 5)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki