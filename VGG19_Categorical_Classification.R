# ------------------------------------------------------------------------------
# VGG19 MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/cifar-10/overview
utils::browseURL(url = "https://www.kaggle.com/c/cifar-10/overview")

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
# keras::install_keras(tensorflow = "gpu")
base::library(tidyverse)
base::library(deepviz)
base::source("D:\\GitHub\\DeepNeuralNetworksRepoR\\Binary_Categorical_Model_Evaluation.R")

train_dir <- "D:\\GitHub\\Datasets\\Cifar10\\train"
validation_dir <- "D:\\GitHub\\Datasets\\Cifar10\\validation"
test_dir <- "D:\\GitHub\\Datasets\\Cifar10\\test"
callback_model_checkpoint_path <- "D:\\GitHub\\DeepNeuralNetworksRepoR\\VGG19\\Categorical\\keras_model.weights.{epoch:02d}-{val_acc:.2f}.hdf5"
callback_tensorboard_path <- "D:\\GitHub\\DeepNeuralNetworksRepoR\\VGG19\\Categorical\\logs"
callback_csv_logger_path <- "D:\\GitHub\\DeepNeuralNetworksRepoR\\VGG19\\Categorical\\Optimization_logger.csv"
models_store <- "D:\\GitHub\\DeepNeuralNetworksRepoR\\VGG19\\Categorical"

count_files = function(path){
  dirs <- base::list.dirs(path = path)
  dirs <- dirs[2:base::length(dirs)]
  files <- base::integer(base::length(dirs))
  folder <- base::character(base::length(dirs))
  for (i in base::seq_along(dirs)){
    files[i] <- base::length(base::list.files(path = dirs[i]))
    folder[i] <- base::basename(path = dirs[i])}
  result = base::data.frame(category = folder, category_obs = files)
  return(result)}

train_files <- count_files(path = train_dir); train_files
validation_files <- count_files(path = validation_dir); validation_files
test_files <- count_files(path = test_dir); test_files

# ------------------------------------------------------------------------------
# Clear session
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Setting pipeline parameters values: 
# Image:
image_size <- 224
channels <- 3

# Model structure:
weights <- "imagenet"
include_top <- FALSE
activation_1 <- "relu"
activation_2 <- "softmax"
units <- 4096

# Model compilation:
loss <- "categorical_crossentropy"
optimizer <- keras::optimizer_adam()
metrics <- base::c("acc")

# Augmentation:
Augmentation <- TRUE
rescale <- 1/255
rotation_range <- 25
width_shift_range <- 0.1
height_shift_range <- 0.1
shear_range <- 0.1
zoom_range <- 0.1
brightness_range <- base::c(0.5, 1.5)
horizontal_flip <- TRUE
vertical_flip <- FALSE
fill_mode <- "nearest"
featurewise_center <- FALSE
samplewise_center <- FALSE
featurewise_std_normalization <- FALSE
samplewise_std_normalization <- FALSE
zca_whitening <- FALSE
zca_epsilon <- 1e-06
channel_shift_range <- 0
cval <- 0

# Training:
batch_size <- 16
class_mode <- "categorical"
shuffle <- TRUE
epochs <- 25
patience <- 10
monitor <- "val_acc"
save_best_only <- TRUE
mode <- "max"
verbose <- 1
write_graph <- TRUE
write_grads <- TRUE
write_images <- TRUE
restore_best_weights <- FALSE
histogram_freq <- 1
min_delta <- 0

# ------------------------------------------------------------------------------
# VGG19 model architecture:
model <- keras::application_vgg19(include_top = include_top,
                                  weights = weights,
                                  input_shape = base::c(image_size, image_size, channels))

input_tensor <- keras::layer_input(shape = c(image_size, image_size, channels))
output_tensor <- input_tensor %>%
  model %>%
  keras::layer_flatten() %>%
  keras::layer_dense(units = units) %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_dense(units = units) %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_dense(units = base::length(base::levels(validation_files$category)), activation = activation_2) 

model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)

# ------------------------------------------------------------------------------
# Upload pre-trained model for training:
# last_model <- base::list.files(path = models_store, pattern = ".hdf5")[base::length(base::list.files(path = models_store, pattern = ".hdf5"))]
# model <- keras::load_model_hdf5(filepath = paste(models_store, last_model, sep = "\\"), compile = FALSE)

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
if (Augmentation == TRUE){
  train_datagen <- keras::image_data_generator(rescale = rescale,
                                               rotation_range = rotation_range, 
                                               width_shift_range = width_shift_range,
                                               height_shift_range = height_shift_range,
                                               shear_range = shear_range,
                                               zoom_range = zoom_range,
                                               brightness_range = brightness_range,
                                               horizontal_flip = horizontal_flip,
                                               vertical_flip = vertical_flip,
                                               fill_mode = fill_mode,
                                               featurewise_center = featurewise_center,
                                               samplewise_center = samplewise_center,
                                               featurewise_std_normalization = featurewise_std_normalization,
                                               samplewise_std_normalization = samplewise_std_normalization,
                                               zca_whitening = zca_whitening,
                                               zca_epsilon = zca_epsilon,
                                               channel_shift_range = channel_shift_range,
                                               cval = cval)
  train_generator <- keras::flow_images_from_directory(directory = train_dir,
                                                       generator = train_datagen, 
                                                       target_size = base::c(image_size, image_size),
                                                       batch_size = batch_size,
                                                       class_mode = class_mode,
                                                       classes = base::levels(validation_files$category),
                                                       shuffle = shuffle)
}

if (Augmentation == FALSE){
  train_datagen <- keras::image_data_generator(rescale = rescale)
  train_generator <- keras::flow_images_from_directory(directory = train_dir,
                                                       generator = train_datagen, 
                                                       target_size = base::c(image_size, image_size),
                                                       batch_size = batch_size,
                                                       class_mode = class_mode,
                                                       classes = base::levels(validation_files$category),
                                                       shuffle = shuffle)
}

validation_datagen <- keras::image_data_generator(rescale = rescale) 
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
# 3. cd C:\Users\admin\Desktop\GitHub\Models_Store\ResNet50-Models
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
                                          callbacks = base::c(keras::callback_model_checkpoint(filepath = callback_model_checkpoint_path,
                                                                                               monitor = monitor,
                                                                                               verbose = verbose,
                                                                                               save_best_only = save_best_only,
                                                                                               mode = mode),
                                                              keras::callback_early_stopping(monitor = monitor,
                                                                                             min_delta = min_delta,
                                                                                             verbose = verbose,
                                                                                             patience = patience,
                                                                                             restore_best_weights = restore_best_weights),
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
# Clear session and import the best trained model:
keras::k_clear_session()
last_model <- base::list.files(path = models_store, pattern = ".hdf5")[base::length(base::list.files(path = models_store, pattern = ".hdf5"))]; last_model
model <- keras::load_model_hdf5(filepath = paste(models_store, last_model, sep = "\\"), compile = TRUE)

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Model evaluation and predictions using generators:
train_datagen <- keras::image_data_generator(rescale = rescale)
train_generator <- keras::flow_images_from_directory(directory = train_dir,
                                                     generator = train_datagen, 
                                                     target_size = base::c(image_size, image_size),
                                                     batch_size = batch_size,
                                                     class_mode = class_mode,
                                                     classes = base::levels(validation_files$category),
                                                     shuffle = FALSE)

validation_datagen <- keras::image_data_generator(rescale = rescale)
validation_generator <- keras::flow_images_from_directory(directory = validation_dir,
                                                          generator = validation_datagen,
                                                          target_size = base::c(image_size, image_size),
                                                          batch_size = batch_size,
                                                          class_mode = class_mode,
                                                          classes = base::levels(validation_files$category),
                                                          shuffle = FALSE)

test_datagen <- keras::image_data_generator(rescale = rescale)
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

# ------------------------------------------------------------------------------
# Model verification:
labels <- base::sort(base::as.character(train_files$category)); labels
actual_train <- base::rep(x = 1:base::length(train_files$category), times = train_files$category_obs); actual_train
actual_validation <- base::rep(x = 1:base::length(validation_files$category), times = validation_files$category_obs); actual_validation
actual_test <- base::rep(x = 1:base::length(test_files$category), times = test_files$category_obs); actual_test

Categorical_train_results <- Categorical_Model_Evaluation(actual = actual_train,
                                                          probabilities = train_probabilities,
                                                          labels = labels,
                                                          type_info = "Train ResNet50",
                                                          save = FALSE,
                                                          open = FALSE)

Categorical_validation_results <- Categorical_Model_Evaluation(actual = actual_validation,
                                                               probabilities = validation_probabilities,
                                                               labels = labels,
                                                               type_info = "Validation ResNet50",
                                                               save = FALSE,
                                                               open = FALSE)

Categorical_test_results <- Categorical_Model_Evaluation(actual = actual_test,
                                                         probabilities = test_probabilities,
                                                         labels = labels,
                                                         type_info = "Test ResNet50",
                                                         save = FALSE,
                                                         open = FALSE)
# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki