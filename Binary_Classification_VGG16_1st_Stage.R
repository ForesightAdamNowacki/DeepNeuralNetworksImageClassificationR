# ------------------------------------------------------------------------------
# VGG16 BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "VGG16"
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
callback_model_checkpoint_path <- base::paste(models_store_dir, "keras_model.1st_stage_weights.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5", sep = "/")
callback_tensorboard_path <- base::paste(models_store_dir, "logs_freeze_weights", sep = "/")
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
activation_1 <- "linear"
activation_2 <- "relu"
activation_3 <- "softmax"

# Model compilation:
loss <- "categorical_crossentropy"
optimizer <- keras::optimizer_adam()
metrics <- base::c("acc")

# Training:
batch_size <- 16
class_mode <- "categorical"
shuffle <- TRUE
epochs <- 2
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
# VGG16 model architecture:
model <- keras::application_vgg16(include_top = include_top,
                                  weights = weights,
                                  input_shape = base::c(image_size, image_size, channels))

input_tensor <- keras::layer_input(shape = base::c(image_size, image_size, channels))
output_tensor <- input_tensor %>%
  model %>%
  keras::layer_flatten() %>%
  keras::layer_dense(units = 4096, activation = activation_1) %>%
  keras::layer_activation(activation = activation_2) %>%
  keras::layer_dense(units = 4096, activation = activation_1) %>%
  keras::layer_activation(activation = activation_2) %>%
  keras::layer_dense(units = base::length(base::levels(train_files$category)), activation = activation_3)

model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)

base::cat("Trainable layers before freezing:", base::length(model$trainable_weights), "\n")
model %>% keras::freeze_weights(from = "input_2", to = "vgg16") # from including, to including
base::cat("Trainable layers after freezing:", base::length(model$trainable_weights), "\n")

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
# keras::tensorboard(log_dir = callback_tensorboard_path, host = "127.0.0.1")
# If 'ERROR: invalid version specification':
# 1. Anaconda Prompt
# 2. conda activate GPU_ML_2
# 3. cd D:/GitHub/DeepNeuralNetworksRepoR/VGG16/Binary
# 4. tensorboard --logdir=logs_freeze_weights --host=127.0.0.1
# 5. http://127.0.0.1:6006/
# 6. Start model optimization
# 7. F5 http://127.0.0.1:6006/ to examine the latest results

# ------------------------------------------------------------------------------
# Model optimization:
history <- model %>% keras::fit_generator(generator = train_generator,
                                          steps_per_epoch = base::ceiling(train_generator$n/train_generator$batch_size),
                                          epochs = epochs,
                                          validation_data = validation_generator,
                                          validation_steps = base::ceiling(validation_generator$n/validation_generator$batch_size),
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
# https://github.com/ForesightAdamNowacki