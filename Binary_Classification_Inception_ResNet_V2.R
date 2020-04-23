# ------------------------------------------------------------------------------
# INCEPTION RESNET V2 BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Environment:
base::remove(list = base::ls())
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
models_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR/Inception_ResNet_V2/Binary"
callback_model_checkpoint_path <- base::paste(models_store_dir, "keras_model.weights.{epoch:02d}-{val_acc:.2f}.hdf5", sep = "/")
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
image_size <- 299
channels <- 3

# Model structure:
weights <- "imagenet"
include_top <- FALSE
activation <- "sigmoid"

# Model compilation:
loss <- "binary_crossentropy"
optimizer <- keras::optimizer_adam()
metrics <- base::c("acc")

# Training:
batch_size <- 16
class_mode <- "binary"
shuffle <- TRUE
epochs <- 1
patience <- 10
monitor <- "val_loss"
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
# XCEPTION model architecture:
model <- keras::application_inception_resnet_v2(include_top = include_top,
                                     weights = weights,
                                     input_shape = base::c(image_size, image_size, channels))

input_tensor <- keras::layer_input(shape = base::c(image_size, image_size, channels))
output_tensor <- input_tensor %>%
  model %>%
  keras::layer_global_average_pooling_2d() %>%
  keras::layer_dense(units = 1, activation = activation) 

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
                                                     shuffle = TRUE)

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
# 3. cd D:/GitHub/DeepNeuralNetworksRepoR/Inception_ResNet_V2/Binary
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
last_model <- base::list.files(path = models_store_dir, pattern = ".hdf5")[base::length(base::list.files(path = models_store_dir, pattern = ".hdf5"))]; last_model
model <- keras::load_model_hdf5(filepath = paste(models_store_dir, last_model, sep = "/"), compile = FALSE)
# model <- keras::load_model_hdf5(filepath = "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Xception_Binary_Cats_and_Dogs.hdf5")
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

train_probabilities <- keras::predict_generator(model, train_generator, steps = base::ceiling(base::sum(train_files$category_obs)/train_generator$batch_size), verbose = 1)
validation_probabilities <- keras::predict_generator(model, validation_generator, steps = base::ceiling(base::sum(validation_files$category_obs)/validation_generator$batch_size), verbose = 1)
test_probabilities <- keras::predict_generator(model, test_generator, steps = base::ceiling(base::sum(test_files$category_obs)/test_generator$batch_size), verbose = 1)

base::setwd(models_store_dir)
datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
readr::write_csv2(tibble::as_tibble(train_probabilities), base::paste(datetime, "Inception_ResNet_V2_train_binary_probabilities.csv"))
readr::write_csv2(tibble::as_tibble(validation_probabilities), base::paste(datetime, "Inception_ResNet_V2_validation_binary_probabilities.csv"))
readr::write_csv2(tibble::as_tibble(test_probabilities), base::paste(datetime, "Inception_ResNet_V2_test_binary_probabilities.csv"))

# ------------------------------------------------------------------------------
# Model verification - default cutoff:
default_cutoff <- 0.5

train_actual <- base::rep(base::c(0, 1), times = train_files$category_obs)
train_predicted <- train_probabilities[,1]
train_verification_1 <- Binary_Classifier_Verification(actual = train_actual,
                                                       predicted = train_predicted,
                                                       cutoff = default_cutoff,
                                                       type_info = "Train Inception ResNet V2 default cutoff",
                                                       save = FALSE,
                                                       open = FALSE)

validation_actual <- base::rep(base::c(0, 1), times = validation_files$category_obs)
validation_predicted <- validation_probabilities[,1]
validation_verification_1 <- Binary_Classifier_Verification(actual = validation_actual,
                                                            predicted = validation_predicted,
                                                            cutoff = default_cutoff,
                                                            type_info = "Validation Inception ResNet V2 default cutoff",
                                                            save = FALSE,
                                                            open = FALSE)

test_actual <- base::c(base::rep(0, test_files$category_obs[1]/2), base::rep(1, test_files$category_obs[1]/2))
test_predicted <- test_probabilities[,1]
test_verification_1 <- Binary_Classifier_Verification(actual = test_actual,
                                                      predicted = test_predicted,
                                                      cutoff = default_cutoff,
                                                      type_info = "Test Inception ResNet V2 default cutoff",
                                                      save = FALSE,
                                                      open = FALSE)

final_score_1 <- train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  knitr::kable(.); final_score_1

train_verification_1$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_1$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  readr::write_csv2(path = base::paste(models_store_dir, "Summary_Default_Cutoff_Inception_ResNet_V2.csv", sep = "/"))

# ------------------------------------------------------------------------------
# Model verification - cutoff optimization on validation set:
train_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = train_actual,
                                                                   predicted = train_predicted,
                                                                   type_info = "Train Inception ResNet V2",
                                                                   seed_value = 42,
                                                                   top = 10,
                                                                   cuts = 100,
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
                                                                        type_info = "Validation Inception ResNet V2",
                                                                        seed_value = 42,
                                                                        top = 10,
                                                                        cuts = 100,
                                                                        key_metric = ACC,
                                                                        ascending = FALSE,
                                                                        save = FALSE,
                                                                        open = FALSE)
validation_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> validation_optimal_cutoff; validation_optimal_cutoff

selected_cutoff <- validation_optimal_cutoff

train_verification_2 <- Binary_Classifier_Verification(actual = train_actual,
                                                       predicted = train_predicted,
                                                       cutoff = selected_cutoff,
                                                       type_info = "Train Inception ResNet V2 optimized cutoff",
                                                       save = FALSE,
                                                       open = FALSE)

validation_verification_2 <- Binary_Classifier_Verification(actual = validation_actual,
                                                            predicted = validation_predicted,
                                                            cutoff = selected_cutoff,
                                                            type_info = "Validation Inception ResNet V2 optimized cutoff",
                                                            save = FALSE,
                                                            open = FALSE)

test_verification_2 <- Binary_Classifier_Verification(actual = test_actual,
                                                      predicted = test_predicted,
                                                      cutoff = selected_cutoff,
                                                      type_info = "Test Inception ResNet V2 optimized cutoff",
                                                      save = FALSE,
                                                      open = FALSE)

final_score_2 <- train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = selected_cutoff) %>%
  knitr::kable(.); final_score_2

train_verification_2$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification_2$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = selected_cutoff) %>%
  readr::write_csv2(path = base::paste(models_store_dir, "Summary_Optimized_Cutoff_Inception_ResNet_V2.csv", sep = "/"))

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

# ------------------------------------------------------------------------------
# Predict indicated image:
# Train:
Predict_Image(image_path = "D:/GitHub/Datasets/Cats_And_Dogs/train/cats/cat1.jpg",
              model = model,
              plot_image = TRUE)
Predict_Image(image_path = "D:/GitHub/Datasets/Cats_And_Dogs/train/dogs/dog1.jpg",
              model = model,
              plot_image = TRUE)

# Validation:
Predict_Image(image_path = "D:/GitHub/Datasets/Cats_And_Dogs/validation/cats/cat4501.jpg",
              model = model,
              plot_image = TRUE)
Predict_Image(image_path = "D:/GitHub/Datasets/Cats_And_Dogs/validation/dogs/dog4501.jpg",
              model = model,
              plot_image = TRUE)

# Test:
Predict_Image(image_path = "D:/GitHub/Datasets/Cats_And_Dogs/test/test/cat2001.jpg",
              model = model,
              plot_image = TRUE)
Predict_Image(image_path = "D:/GitHub/Datasets/Cats_And_Dogs/test/test/dog2001.jpg",
              model = model,
              plot_image = TRUE)

# ------------------------------------------------------------------------------
# Save true and false predictions:
# Train:
Train_Correct_Incorrect_Classifications <- Organize_Correct_Incorrect_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/train",
                                                                                      actual_classes = train_actual,
                                                                                      prediction = train_predicted,
                                                                                      cwd = models_store_dir,
                                                                                      cutoff = 0.5,
                                                                                      save_summary_files = TRUE,
                                                                                      save_correct_images = TRUE,
                                                                                      save_incorrect_images = TRUE)

# Validation:
Validation_Correct_Incorrect_Classifications <- Organize_Correct_Incorrect_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/validation",
                                                                                           actual_classes = validation_actual,
                                                                                           prediction = validation_predicted,
                                                                                           cwd = models_store_dir,
                                                                                           cutoff = 0.5,
                                                                                           save_summary_files = TRUE,
                                                                                           save_correct_images = TRUE,
                                                                                           save_incorrect_images = TRUE)

# ------------------------------------------------------------------------------
# Visualize predictions distribution:
Visualize_Predictions_Distribution(actual = train_actual,
                                   predicted = train_predicted,
                                   bins = 10)

Visualize_Predictions_Distribution(actual = validation_actual,
                                   predicted = validation_predicted,
                                   bins = 10)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki