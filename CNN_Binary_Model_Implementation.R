# ------------------------------------------------------------------------------
# BINARY CNN MODEL IMPLEMENTATION FROM THE SCRATCH
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Environment:
base::rm(list = ls())
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::library(stringr)
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Binary_Categorical_Model_Evaluation.R")
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Useful_Functions.R")

train_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/train"
validation_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/validation"
test_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/test"
callback_model_checkpoint_path <- "D:/GitHub/DeepNeuralNetworksRepoR/CNN_Binary_Model\\keras_model.weights.{epoch:02d}-{val_acc:.2f}.hdf5"
callback_tensorboard_path <- "D:/GitHub/DeepNeuralNetworksRepoR/CNN_Binary_Model/logs"
callback_csv_logger_path <- "D:/GitHub/DeepNeuralNetworksRepoR/CNN_Binary_Model/Optimization_logger.csv"
models_store <- "D:/GitHub/DeepNeuralNetworksRepoR/CNN_Binary_Model"

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
image_size <- 64
channels <- 3

# Model structure:
activation_1 <- "relu"
activation_2 <- "linear"
activation_3 <- "softmax"
dense_units <- 256
dropout_rate <- 0.5

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
batch_size <- 64
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
# CNN model architecture:
# TRUE: "Christian_Szegedy_Original_Paper": Convolution/Dense Layer -> Linear Activation -> Batch normalization -> Non Linear Activation (e.g. ReLU) -> Dropout -> Convolution/Dense Layer...
# FALSE: "Christian_Szegedy_Modified_Architecture": Convolution/Dense Layer -> Linear Activation -> Non Linear Activation (e.g. ReLU) -> Batch normalization -> Dropout -> Convolution/Dense Layer...
keras::k_clear_session()
model_architecture <- FALSE

input_tensor <- keras::layer_input(shape = base::c(image_size, image_size, channels))

if (model_architecture == TRUE){

  output_tensor <- input_tensor %>%
    
    keras::layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%
    
    keras::layer_conv_2d(filters = 128, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_conv_2d(filters = 128, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%
    
    keras::layer_conv_2d(filters = 256, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_conv_2d(filters = 256, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%
    
    keras::layer_flatten() %>%
    
    keras::layer_dense(units = dense_units, activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_dropout(rate = dropout_rate) %>%
    
    keras::layer_dense(units = dense_units, activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_dropout(rate = dropout_rate) %>%
    
    keras::layer_dense(units = base::length(base::levels(validation_files$category)), activation = activation_3)
  
} else {
  
  output_tensor <- input_tensor %>%
    
    keras::layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%
    
    keras::layer_conv_2d(filters = 128, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 128, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%
    
    keras::layer_conv_2d(filters = 256, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 256, kernel_size = c(3, 3), strides = c(1, 1), padding = "same", activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2), padding = "valid") %>%
    
    keras::layer_flatten() %>%
    
    keras::layer_dense(units = dense_units, activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_dropout(rate = dropout_rate) %>%
    
    keras::layer_dense(units = dense_units, activation = activation_2) %>%
    keras::layer_activation(activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_dropout(rate = dropout_rate) %>%
    
    keras::layer_dense(units = base::length(base::levels(validation_files$category)), activation = activation_3)
  
}

model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)

# ------------------------------------------------------------------------------
# Upload pre-trained model for training:
# last_model <- base::list.files(path = models_store, pattern = ".hdf5")[base::length(base::list.files(path = models_store, pattern = ".hdf5"))]
# model <- keras::load_model_hdf5(filepath = paste(models_store, last_model, sep = "/"), compile = FALSE)

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
} else {
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
# Clear session and import the best trained model:
keras::k_clear_session()
last_model <- base::list.files(path = models_store, pattern = ".hdf5")[base::length(base::list.files(path = models_store, pattern = ".hdf5"))]; last_model
model <- keras::load_model_hdf5(filepath = paste(models_store, last_model, sep = "/"), compile = TRUE)

# ------------------------------------------------------------------------------
# Remove not optimal models:
base::setwd(models_store_dir)
saved_models <- base::sort(base::list.files()[base::grepl(".hdf5", base::list.files())])
for (j in 1:(base::length(saved_models) - 1)){
  base::cat("Remove .hdf5 file:", saved_models[j], "\n")
  base::unlink(saved_models[j], recursive = TRUE, force = TRUE)}

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Model predictions using generators:
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

train_probabilities <- keras::predict_generator(model, train_generator, steps = base::ceiling(base::sum(train_files$category_obs)/train_generator$batch_size), verbose = 1)
validation_probabilities <- keras::predict_generator(model, validation_generator, steps = base::ceiling(base::sum(validation_files$category_obs)/validation_generator$batch_size), verbose = 1)
test_probabilities <- keras::predict_generator(model, test_generator, steps = base::ceiling(base::sum(test_files$category_obs)/test_generator$batch_size), verbose = 1)

# ------------------------------------------------------------------------------
# Model verification - default cutoff:
train_actual <- base::rep(base::c(0, 1), times = train_files$category_obs)
train_predicted <- train_probabilities[,2]
train_verification_1 <- Binary_Classifier_Verification(actual = train_actual,
                                                       predicted = train_predicted,
                                                       cutoff = 0.5,
                                                       type_info = "Train default cutoff",
                                                       save = FALSE,
                                                       open = FALSE)

validation_actual <- base::rep(base::c(0, 1), times = validation_files$category_obs)
validation_predicted <- validation_probabilities[,2]
validation_verification_1 <- Binary_Classifier_Verification(actual = validation_actual,
                                                            predicted = validation_predicted,
                                                            cutoff = 0.5,
                                                            type_info = "Validation default cutoff",
                                                            save = FALSE,
                                                            open = FALSE)

test_actual <- base::c(base::rep(0, test_files$category_obs[1]/2), base::rep(1, test_files$category_obs[1]/2))
test_predicted <- test_probabilities[,2]
test_verification_1 <- Binary_Classifier_Verification(actual = test_actual,
                                                      predicted = test_predicted,
                                                      cutoff = 0.5,
                                                      type_info = "Test default cutoff",
                                                      save = FALSE,
                                                      open = FALSE)

final_score_1 <- train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score) %>%
  knitr::kable(.); final_score_1

# ------------------------------------------------------------------------------
# Model verification - cutoff optimization on validation set:
train_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = train_actual,
                                                                   predicted = train_predicted,
                                                                   type_info = "Train ResNet50",
                                                                   seed_value = 42,
                                                                   top = 10,
                                                                   cuts = 250,
                                                                   key_metric = ACC,
                                                                   ascending = FALSE,
                                                                   save = FALSE,
                                                                   open = FALSE)
train_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> train_optimal_cutoff; train_optimal_cutoff

validation_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = validation_actual,
                                                                        predicted = validation_predicted,
                                                                        type_info = "Validation ResNet50",
                                                                        seed_value = 42,
                                                                        top = 10,
                                                                        cuts = 250,
                                                                        key_metric = ACC,
                                                                        ascending = FALSE,
                                                                        save = FALSE,
                                                                        open = FALSE)
validation_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> validation_optimal_cutoff; validation_optimal_cutoff

selected_cutoff_level <- validation_optimal_cutoff

train_verification_2 <- Binary_Classifier_Verification(actual = train_actual,
                                                       predicted = train_predicted,
                                                       cutoff = selected_cutoff_level,
                                                       type_info = "Train optimized cutoff",
                                                       save = FALSE,
                                                       open = FALSE)

validation_verification_2 <- Binary_Classifier_Verification(actual = validation_actual,
                                                            predicted = validation_predicted,
                                                            cutoff = selected_cutoff_level,
                                                            type_info = "Validation optimized cutoff",
                                                            save = FALSE,
                                                            open = FALSE)

test_verification_2 <- Binary_Classifier_Verification(actual = test_actual,
                                                      predicted = test_predicted,
                                                      cutoff = selected_cutoff_level,
                                                      type_info = "Test optimized cutoff",
                                                      save = FALSE,
                                                      open = FALSE)

final_score_2 <- train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score) %>%
  knitr::kable(.); final_score_2

# ------------------------------------------------------------------------------
# Final summary:
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

final_score <- base::list("scores" = final_score_1_summary %>%
                            dplyr::left_join(final_score_2_summary, by = "Metric") %>%
                            dplyr::rename(Train_default = Score_train.x,
                                          Validation_default = Score_validation.x,
                                          Test_default = Score_test.x,
                                          Train_optimized = Score_train.y,
                                          Validation_optimized = Score_validation.y,
                                          Test_optimized = Score_test.y) %>%
                            dplyr::mutate(Train_diff = Train_optimized - Train_default,
                                          Validation_diff = Validation_optimized - Validation_default,
                                          Test_diff = Test_optimized - Test_default) %>%
                            dplyr::filter(Metric %!in% base::c('Number of Observations', 'Area Under ROC Curve', 'Condition Negative', 'Condition Positive', 'Gini Index')) %>%
                            knitr::kable(.),
                          "cutoff" = selected_cutoff_level); final_score
# ------------------------------------------------------------------------------
# Predict one selected image:
base::setwd("D:\\GitHub\\Datasets\\Cats_And_Dogs\\train\\cats")
i <- 1 # file index
Predict_Image(image_path = base::list.files(base::getwd())[i], model = model)

base::setwd("D:\\GitHub\\Datasets\\Cats_And_Dogs\\train\\dogs")
i <- 1 # file index
Predict_Image(image_path = base::list.files(base::getwd())[i], model = model)

# ------------------------------------------------------------------------------
# Correct and incorrect classifications:
base::setwd(models_store); getwd()
Organize_Correct_Incorrect_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/train",
                                           actual_classes = train_actual,
                                           prediction = train_predicted,
                                           cutoff = selected_cutoff_level)

Organize_Correct_Incorrect_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/validation",
                                           actual_classes = validation_actual,
                                           prediction = validation_predicted,
                                           cutoff = selected_cutoff_level)
  
# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki