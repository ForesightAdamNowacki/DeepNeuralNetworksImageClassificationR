# ------------------------------------------------------------------------------
# DENSENET201 CATEGORICAL MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/cifar-10/overview
utils::browseURL(url = "https://www.kaggle.com/c/cifar-10/overview")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "DenseNet201"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'DenseNet201' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))
# 3. Create 'Categorical' subfolder in 'DenseNet201' main folder
base::dir.create(path = base::paste(base::getwd(), model_name, "Categorical", sep = "/"))

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Useful_Functions.R")

train_dir <- "D:/GitHub/Datasets/Cifar10/train"
validation_dir <- "D:/GitHub/Datasets/Cifar10/validation"
test_dir <- "D:/GitHub/Datasets/Cifar10/test"
models_store_dir <- base::paste(base::getwd(), model_name, "Categorical", sep = "/")
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
batch_size <- 8
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
# DenseNet201 model architecture:
model <- keras::application_densenet201(include_top = include_top,
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
# 3. cd D:\GitHub\DeepNeuralNetworksRepoR\DenseNet201\Categorical
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
optimal_model_repo_dir <- base::paste(models_repo_store_dir, base::paste("Categorical", model_name, "Model.hdf5", sep = "_"), sep = "/")
base::file.copy(from = optimal_model,
                to = optimal_model_repo_dir, 
                overwrite = TRUE); base::cat("Optimal model directory:", optimal_model_repo_dir, "\n")
base::unlink(optimal_model, recursive = TRUE, force = TRUE)

# ------------------------------------------------------------------------------
# Clear session and import the best trained model:
keras::k_clear_session()
optimal_model_repo_dir <- base::paste(models_repo_store_dir, base::paste("Categorical", model_name, "Model.hdf5", sep = "_"), sep = "/")
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
readr::write_csv2(tibble::as_tibble(train_probabilities) %>%
                    dplyr::mutate(filepath = train_generator$filepaths,
                                  actual_class = train_generator$classes + 1,
                                  model = model_name),
                  base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, "train_binary_probabilities.csv", sep = "_"))
readr::write_csv2(tibble::as_tibble(validation_probabilities) %>%
                    dplyr::mutate(filepath = validation_generator$filepaths,
                                  actual_class = validation_generator$classes + 1,
                                  model = model_name),
                  base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, "validation_binary_probabilities.csv", sep = "_"))
readr::write_csv2(tibble::as_tibble(test_probabilities) %>%
                    dplyr::mutate(filepath = test_generator$filepaths,
                                  actual_class = test_generator$classes + 1,
                                  model = model_name), 
                  base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, "test_binary_probabilities.csv", sep = "_"))

# ------------------------------------------------------------------------------
# Model verification:
labels <- base::sort(base::as.character(train_files$category)); labels
save_option <- TRUE

Categorical_train_results <- Categorical_Classifier_Verification(actual = train_generator$classes + 1,
                                                                 probabilities = train_probabilities,
                                                                 labels = labels,
                                                                 type_info = base::paste(model_name, "train", sep = "_"),
                                                                 save = save_option,
                                                                 open = FALSE)

Categorical_validation_results <- Categorical_Classifier_Verification(actual = validation_generator$classes + 1,
                                                                      probabilities = validation_probabilities,
                                                                      labels = labels,
                                                                      type_info = base::paste(model_name, "validation", sep = "_"),
                                                                      save = save_option,
                                                                      open = FALSE)

Categorical_test_results <- Categorical_Classifier_Verification(actual = test_generator$classes + 1,
                                                                probabilities = test_probabilities,
                                                                labels = labels,
                                                                type_info = base::paste(model_name, "test", sep = "_"),
                                                                save = save_option,
                                                                open = FALSE)

# ------------------------------------------------------------------------------
# Predict indicated image:
labels <- base::sort(base::as.character(train_files$category)); labels
set <- "train"
category <- "automobile"  
id <- 1

Predict_Image(image_path = base::paste("D:/GitHub/Datasets/Cifar10", set, category, base::list.files(base::paste("D:/GitHub/Datasets/Cifar10", set, category, sep = "/")), sep = "/")[id],
              model = model,
              classes = labels,
              plot_image = TRUE)

# ------------------------------------------------------------------------------
# Save true and false predictions:
save_summary_files <- TRUE
save_correct_images <- FALSE
save_incorrect_images <- FALSE

# Train:
Train_Correct_Incorrect_Categorical_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = "D:/GitHub/Datasets/Cifar10/train",
                                                                                                              actual_classes = train_generator$classes + 1,
                                                                                                              predicted = train_probabilities,
                                                                                                              type_info = model_name,
                                                                                                              cwd = models_store_dir,
                                                                                                              save_summary_files = save_summary_files,
                                                                                                              save_correct_images = save_correct_images,
                                                                                                              save_incorrect_images = save_incorrect_images)

# Validation:
Validation_Correct_Incorrect_Categorical_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = "D:/GitHub/Datasets/Cifar10/validation",
                                                                                                                   actual_classes = validation_generator$classes + 1,
                                                                                                                   predicted = validation_probabilities,
                                                                                                                   type_info = model_name,
                                                                                                                   cwd = models_store_dir,
                                                                                                                   save_summary_files = save_summary_files,
                                                                                                                   save_correct_images = save_correct_images,
                                                                                                                   save_incorrect_images = save_incorrect_images)

# Test:
Test_Correct_Incorrect_Categorical_Classifications <- Organize_Correct_Incorrect_Categorical_Classifications(dataset_dir = "D:/GitHub/Datasets/Cifar10/test",
                                                                                                             actual_classes = test_generator$classes + 1,
                                                                                                             predicted = test_probabilities,
                                                                                                             type_info = model_name,
                                                                                                             cwd = models_store_dir,
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
                                             bins = 4,
                                             type_info = base::paste(model_name, "train", sep = "_"),
                                             save_plot = save_plot,
                                             plot_size = 30)

Display_All_Classes_Predictions_Distribution(actual = validation_generator$classes + 1,
                                             predicted = validation_probabilities,
                                             labels = labels,
                                             bins = 4,
                                             type_info = base::paste(model_name, "validation", sep = "_"),
                                             save_plot = save_plot,
                                             plot_size = 30)

Display_All_Classes_Predictions_Distribution(actual = test_generator$classes + 1,
                                             predicted = test_probabilities,
                                             labels = labels,
                                             bins = 4,
                                             type_info = base::paste(model_name, "test", sep = "_"),
                                             save_plot = save_plot,
                                             plot_size = 30)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki