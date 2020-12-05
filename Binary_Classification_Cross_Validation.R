# ------------------------------------------------------------------------------
# K-FOLD CROSS VALIDATION BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
# browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model:
model_name <- "Cross_Validation"
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
cv_data_dir <- "D:/GitHub/Datasets/Cats_And_Dogs_Small/data"
models_store_dir <-  paste(getwd(), model_name, model_type, sep = "/")
repo_models_store_dir <- paste("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store", model_name, model_type, sep = "/")
dir.create(path = paste("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store", model_name, sep = "/"))
dir.create(path = paste("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store", model_name, model_type, sep = "/"))
folds_directory <- "D:/GitHub/Datasets/Cats_And_Dogs_Small/folds"

# ------------------------------------------------------------------------------
# Create folds:
folds <- 5
Create_KFolds_Directories(data_dir = cv_data_dir,
                          target_dir = folds_directory,
                          folds = folds,
                          seed = 42)

# Display number of observations per class in each fold:
folds_dirs <- paste(folds_directory, list.files(path = folds_directory, pattern = "fold"), sep = "/")
for (i in seq_along(folds_dirs)){
  cat(folds_dirs[i]); cat("\n")
  print(Count_Files(path = paste(folds_dirs[i], sep = "/")))
  cat("\n")}

# Display number of observations per dataset type (train, validation) and class in each step of cross validation:
steps_dirs <- expand.grid(paste(folds_directory, list.files(folds_directory)[grepl("step", list.files(folds_directory))], sep = "/"),
                                c("train", "validation")) %>%
  dplyr::mutate(steps_dirs = paste(Var1, Var2, sep = "/")) %>%
  dplyr::select(steps_dirs) %>%
  dplyr::pull() %>%
  sort()
for (i in seq_along(steps_dirs)){
  cat(steps_dirs[i]); cat("\n")
  print(Count_Files(path = paste(steps_dirs[i], sep = "/")))
  cat("\n")}

# Count train files:
train_steps_dirs <- steps_dirs[grepl("train", steps_dirs)]
train_steps_files <- list()
for (i in 1:folds){train_steps_files[[i]] <- Count_Files(path = train_steps_dirs[i])}
train_steps_files

# Count validation files:
validation_steps_dirs <- steps_dirs[grepl("validation", steps_dirs)]
validation_steps_files <- list()
for (i in 1:folds){validation_steps_files[[i]] <- Count_Files(path = validation_steps_dirs[i])}
validation_steps_files

# Main datasets:
train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# Set current working directory:
setwd(models_store_dir)

# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Model architecture:
build_model <- function(image_size = 150, channels = 3,
                        activation_1 = "linear", activation_2 = "relu", activation_3 = "softmax",
                        loss = "categorical_crossentropy", optimizer = keras::optimizer_adam(), metrics = c("acc")){
  
  input_tensor <- keras::layer_input(shape = c(image_size, image_size, channels))
  output_tensor <- input_tensor %>%
    keras::layer_conv_2d(filters = 32, kernel_size = c(3, 3), strides = c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 32, kernel_size = c(3, 3), strides = c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = 2) %>%
    
    keras::layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = 2) %>%
    
    keras::layer_conv_2d(filters = 128, kernel_size = c(3, 3), strides = c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 128, kernel_size = c(3, 3), strides = c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = 2) %>%
    
    keras::layer_flatten() %>%
    
    keras::layer_dense(units = 256, activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_dropout(rate = 0.5) %>%
    
    keras::layer_dense(units = 256, activation = activation_1) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_dropout(rate = 0.5) %>%
    
    keras::layer_dense(units = length(list.files(path = cv_data_dir)), activation = activation_1) %>%
    keras::layer_activation(activation = activation_3)
  
  model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)
  
  model %>% keras::compile(loss = loss,
                           optimizer = optimizer, 
                           metrics = metrics)
  return(model)}

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% summary()

# ------------------------------------------------------------------------------
# Cross validation algorithm:
Cross_Validation_Pipe_Runner <- function(epochs, folds_directory, models_store_dir,
                                         image_size = 150, batch_size = 32, 
                                         class_mode = "categorical", shuffle = TRUE,
                                         early_stopping_patience = 10, 
                                         reduce_lr_on_plateu_patience = 5, monitor = "val_acc",
                                         save_best_only = TRUE, verbose = 1, write_graph = TRUE,
                                         write_grads = TRUE, write_images = TRUE, min_delta = 0,
                                         restore_best_weights = FALSE, histogram_freq = 1){
  
  datetime <- stringr::str_replace_all(Sys.time(), ":", "-")
  if (monitor == "val_loss"){mode <- "min"} else {mode <- "max"}
  history_list <- list()
  
  for (i in 1:folds){
    
    keras::k_clear_session()
    
    train_dir <- paste0(folds_directory, "/step_", i, "/train")
    validation_dir <- paste0(folds_directory, "/step_", i, "/validation")
    callback_model_checkpoint_path <- paste(models_store_dir, paste("fold", i, "keras_model.weights.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5", sep = "_"), sep = "/")
    callback_tensorboard_path <- paste(models_store_dir,  paste("fold", i, "logs", sep = "_"), sep = "/")
    callback_csv_logger_path <- paste(models_store_dir, paste("fold", i, "optimization_logger.csv", sep = "_"), sep = "/")
    
    cat(paste("Fold:", i, "\n"))
    cat("Train directory:", train_dir, "\n")
    cat("Validation directory:", validation_dir, "\n")
    
    model <- build_model(image_size = image_size)
    
    train_datagen <- keras::image_data_generator(featurewise_center = FALSE,
                                                 samplewise_center = FALSE,
                                                 featurewise_std_normalization = FALSE,
                                                 samplewise_std_normalization = FALSE,
                                                 zca_whitening = FALSE,
                                                 zca_epsilon = 1e-06,
                                                 rotation_range = 0,
                                                 width_shift_range = 0,
                                                 height_shift_range = 0,
                                                 brightness_range = c(1, 1),
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
                                                         target_size = c(image_size, image_size),
                                                         batch_size = batch_size,
                                                         class_mode = class_mode,
                                                         classes = list.files(path = cv_data_dir),
                                                         shuffle = shuffle)
    
    validation_datagen <- keras::image_data_generator(rescale = 1/255) 
    validation_generator <- keras::flow_images_from_directory(directory = validation_dir,
                                                              generator = validation_datagen,
                                                              target_size = c(image_size, image_size),
                                                              batch_size = batch_size,
                                                              class_mode = class_mode,
                                                              classes = list.files(path = cv_data_dir),
                                                              shuffle = shuffle)
    
    history <- model %>% keras::fit_generator(generator = train_generator,
                                              steps_per_epoch = ceiling(train_generator$n/train_generator$batch_size), 
                                              epochs = epochs,
                                              validation_data = validation_generator,
                                              validation_steps = ceiling(validation_generator$n/validation_generator$batch_size), 
                                              callbacks = list(keras::callback_model_checkpoint(filepath = callback_model_checkpoint_path,
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
    history_list[[i]] <- history$metrics %>%
      tibble::as_tibble() %>%
      dplyr::mutate(epoch = dplyr::row_number(),
                    fold = i) %>%
      tibble::as_tibble()
    
    
    files <- paste(getwd(), list.files(path = getwd(), pattern = paste("fold", i, sep = "_")), sep = "/")
    files <- sort(files[grepl(".hdf5", files)])
    for (j in 1:(length(files) - 1)){
      cat("Remove .hdf5 file:", files[j], "\n")
      unlink(files[j], recursive = TRUE, force = TRUE)}
  }
  
  history_list %>%
    do.call(dplyr::bind_rows, .) %>%
    readr::write_csv(paste(datetime, "CV_results.csv"))
}

Cross_Validation_Pipe_Runner(epochs = 5,
                             folds_directory = folds_directory,
                             models_store_dir = models_store_dir)

# ------------------------------------------------------------------------------
# Remove logs folders if are not important:
logs_folders <- paste(getwd(), list.files(path = getwd(), pattern = "logs"), sep = "/")
for (i in seq_along(logs_folders)){
  cat("Remove logs folder:", logs_folders[i], "\n")
  unlink(logs_folders[i], force = TRUE, recursive = TRUE)}

# ------------------------------------------------------------------------------
# Save optimal models in local models repository: 
optimal_models <- paste(getwd(), list.files(path = getwd(), pattern = ".hdf5"), sep = "/")
file.copy(from = optimal_models,
                to = paste(repo_models_store_dir, basename(optimal_models), sep = "/"))
for (i in seq_along(optimal_models)){
  cat("Remove model:", optimal_models[i], "\n")
  unlink(optimal_models[i])}
optimal_models_repo_store <- paste(repo_models_store_dir, basename(optimal_models), sep = "/")
for (i in seq_along(optimal_models_repo_store)){
  cat("Fold", i, "optimal model directory:", optimal_models_repo_store[i], "\n")}

# ------------------------------------------------------------------------------
# Cross validation results:
results <- readr::read_csv(list.files(getwd(), pattern = "CV_results.csv")); results

# ------------------------------------------------------------------------------
# Display cross validation results:
title_size <- 9
text_size <- 7

# Accuracy:
results %>%
  dplyr::select(dplyr::contains("acc"), epoch, fold) %>%
  tidyr::pivot_longer(cols = c("acc", "val_acc"),
                      names_to = "type",
                      values_to = "value") %>%
  dplyr::mutate(type = dplyr::case_when(type == "acc" ~ "Train accuracy",
                                        type == "val_acc" ~ "Validation accuracy")) %>%
  dplyr::group_by(epoch, type) %>%
  dplyr::mutate(median = stats::median(value),
                mean = mean(value),
                fold = factor(fold)) -> results_accuracy; results_accuracy

results_accuracy %>%
  ggplot2::ggplot(data = .) +
  ggplot2::geom_point(mapping = ggplot2::aes(x = epoch, y = value)) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = value), stat = "smooth", col = "blue", lwd = 1.25) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = median), col = "red", lwd = 1.25) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = mean), col = "black", lwd = 1.25) +  
  ggplot2::facet_wrap(.~type) +
  ggplot2::labs(x = "Epoch",
                y = "Accuracy",
                title = "Accuracy distribution for Cross Validation") +
  ggplot2::theme(plot.title = element_text(size = title_size, color = "black", face = "bold", hjust = 0.5, vjust = 0.5),
                 axis.text.y = element_text(size = text_size, color = "black", face = "plain"),
                 axis.text.x = element_text(size = text_size, color = "black", face = "plain"),
                 axis.title.y = element_text(size = text_size, color = "black", face = "bold"),
                 axis.title.x = element_text(size = text_size, color = "black", face = "bold"),
                 axis.ticks = element_line(size = 1, color = "black", linetype = "solid"),
                 axis.ticks.length = unit(0.1, "cm"),
                 plot.background = element_rect(fill = "gray80", color = "black", size = 1, linetype = "solid"),
                 panel.background = element_rect(fill = "gray90", color = "black", size = 0.5, linetype = "solid"),
                 panel.border = element_rect(fill = NA, color = "black", size = 0.5, linetype = "solid"),
                 panel.grid.major.x = element_line(color = "black", linetype = "dashed"),
                 panel.grid.major.y = element_line(color = "black", linetype = "dashed"),
                 panel.grid.minor.x = element_line(color = "black", linetype = "dotted"),
                 panel.grid.minor.y = element_line(color = "black", linetype = "dotted"),
                 plot.caption = element_text(size = text_size, color = "black", face = "bold", hjust = 1),
                 legend.position = "right",
                 strip.background = element_rect(color = "black", fill = "gray80", size = 0.5, linetype = "solid"),
                 strip.text = element_text(size = text_size, face = "bold")) -> accuracy_plot; accuracy_plot
ggplot2::ggsave("Plot_accuracy_CV.png", accuracy_plot, units = "cm", width = 20, height = 20)

# Loss:
results %>%
  dplyr::select(dplyr::contains("loss"), epoch, fold) %>%
  tidyr::pivot_longer(cols = c("loss", "val_loss"),
                      names_to = "type",
                      values_to = "value") %>%
  dplyr::mutate(type = dplyr::case_when(type == "loss" ~ "Train loss",
                                        type == "val_loss" ~ "Validation loss")) %>%
  dplyr::group_by(epoch, type) %>%
  dplyr::mutate(median = stats::median(value),
                mean = mean(value),
                fold = factor(fold)) -> results_loss; results_loss

results_loss %>%
  ggplot2::ggplot(data = .) +
  ggplot2::geom_point(mapping = ggplot2::aes(x = epoch, y = value)) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = value), stat = "smooth", col = "blue", lwd = 1.25) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = median), col = "red", lwd = 1.25) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = mean), col = "black", lwd = 1.25) +  
  ggplot2::facet_wrap(.~type) +
  ggplot2::labs(x = "Epoch",
                y = "Loss",
                title = "Loss distribution for cross validation") +
  ggplot2::theme(plot.title = element_text(size = title_size, color = "black", face = "bold", hjust = 0.5, vjust = 0.5),
                 axis.text.y = element_text(size = text_size, color = "black", face = "plain"),
                 axis.text.x = element_text(size = text_size, color = "black", face = "plain"),
                 axis.title.y = element_text(size = text_size, color = "black", face = "bold"),
                 axis.title.x = element_text(size = text_size, color = "black", face = "bold"),
                 axis.ticks = element_line(size = 1, color = "black", linetype = "solid"),
                 axis.ticks.length = unit(0.1, "cm"),
                 plot.background = element_rect(fill = "gray80", color = "black", size = 1, linetype = "solid"),
                 panel.background = element_rect(fill = "gray90", color = "black", size = 0.5, linetype = "solid"),
                 panel.border = element_rect(fill = NA, color = "black", size = 0.5, linetype = "solid"),
                 panel.grid.major.x = element_line(color = "black", linetype = "dashed"),
                 panel.grid.major.y = element_line(color = "black", linetype = "dashed"),
                 panel.grid.minor.x = element_line(color = "black", linetype = "dotted"),
                 panel.grid.minor.y = element_line(color = "black", linetype = "dotted"),
                 plot.caption = element_text(size = text_size, color = "black", face = "bold", hjust = 1),
                 legend.position = "right",
                 strip.background = element_rect(color = "black", fill = "gray80", size = 0.5, linetype = "solid"),
                 strip.text = element_text(size = text_size, face = "bold")) -> loss_plot; loss_plot
ggplot2::ggsave("Plot_loss_CV.png", loss_plot, units = "cm", width = 20, height = 20)

# ------------------------------------------------------------------------------
# Save predictions from all folds models:
Predict_Folds_Models <- function(data_dir, type, batch_size = 16){
  
  for (i in seq_along(optimal_models_repo_store)){
    print(optimal_models_repo_store[i])
    model <- build_model() %>%
      keras::load_model_weights_hdf5(optimal_models_repo_store[i]) %>%
      keras::compile(loss = "categorical_crossentropy",
                     optimizer = keras::optimizer_adam(), 
                     metrics = c("acc"))
    
    datagen <- keras::image_data_generator(rescale = 1/255)
    generator <- keras::flow_images_from_directory(directory = data_dir,
                                                   generator = datagen,
                                                   target_size = c(model$input_shape[[2]], model$input_shape[[2]]),
                                                   batch_size = batch_size,
                                                   class_mode = "categorical",
                                                   shuffle = FALSE)
    prediction <- keras::predict_generator(model, generator, steps = ceiling(generator$n/generator$batch_size), verbose = 1)
    prediction <- prediction %>%
      tibble::as_tibble()
    
    filename <- paste(getwd(), paste(type, "prediction", i, "model.csv", sep = "_"), sep = "/")
    prediction %>% 
      dplyr::mutate(filepath = generator$filepaths,
                    actual_class = generator$classes) %>%
      readr::write_csv(filename)
    
    cat("Save file:", filename, "\n")}}

Predict_Folds_Models(data_dir = cv_data_dir, type = "cv_data")
Predict_Folds_Models(data_dir = train_dir, type = "train_data")
Predict_Folds_Models(data_dir = validation_dir, type = "validation_data")
Predict_Folds_Models(data_dir = test_dir, type = "test_data")

# ------------------------------------------------------------------------------
# Model verification - default cutoff:
default_cutoff <- 0.5

# Cross validation data:
cv_results <- list()
for (i in 1:folds){
  cv_results[[i]] <- readr::read_csv(list.files(pattern = "cv_data")[i])
  cv_results[[i]]$model <- paste("model", i, sep = "_")}
cv_results %>%
  do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  dplyr::summarise(V1 = mean(V1),
                   V2 = mean(V2),
                   actual_class = mean(actual_class)) -> cv_results; cv_results
  
cv_verification <- Binary_Classifier_Verification(actual = cv_results$actual_class,
                                                  predicted = cv_results$V2,
                                                  cutoff = default_cutoff,
                                                  type_info = "CV_data_default_cutoff",
                                                  save = TRUE,
                                                  open = FALSE)

# Train data:
train_results <- list()
for (i in 1:folds){
  train_results[[i]] <- readr::read_csv(list.files(pattern = "train_data")[i])
  train_results[[i]]$model <- paste("model", i, sep = "_")}
train_results %>%
  do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  dplyr::summarise(V1 = mean(V1),
                   V2 = mean(V2),
                   actual_class = mean(actual_class)) -> train_results; train_results

train_verification <- Binary_Classifier_Verification(actual = train_results$actual_class,
                                                     predicted = train_results$V2,
                                                     cutoff = default_cutoff,
                                                     type_info = "Train_data_default_cutoff",
                                                     save = TRUE,
                                                     open = FALSE)

# Validation data:
validation_results <- list()
for (i in 1:folds){
  validation_results[[i]] <- readr::read_csv(list.files(pattern = "validation_data")[i])
  validation_results[[i]]$model <- paste("model", i, sep = "_")}
validation_results %>%
  do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  dplyr::summarise(V1 = mean(V1),
                   V2 = mean(V2),
                   actual_class = mean(actual_class)) -> validation_results; validation_results

validation_verification <- Binary_Classifier_Verification(actual = validation_results$actual_class,
                                                          predicted = validation_results$V2,
                                                          cutoff = default_cutoff,
                                                          type_info = "Validation_data_default_cutoff",
                                                          save = TRUE,
                                                          open = FALSE)

# Test data:
test_results <- list()
for (i in 1:folds){
  test_results[[i]] <- readr::read_csv(list.files(pattern = "test_data")[i])
  test_results[[i]]$model <- paste("model", i, sep = "_")}
test_results %>%
  do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  dplyr::summarise(V1 = mean(V1),
                   V2 = mean(V2),
                   actual_class = mean(actual_class)) -> test_results; test_results

test_verification <- Binary_Classifier_Verification(actual = test_results$actual_class,
                                                    predicted = test_results$V2,
                                                    cutoff = default_cutoff,
                                                    type_info = "Test_data_default_cutoff",
                                                    save = TRUE,
                                                    open = FALSE)

# ------------------------------------------------------------------------------
# Results summary:
cv_verification$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_cv = Score) %>%
  dplyr::mutate(Score_train = train_verification$Assessment_of_Classifier_Effectiveness$Score,
                Score_validation = validation_verification$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  knitr::kable(.)

datetime <- stringr::str_replace_all(Sys.time(), ":", "-")
cv_verification$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(Metric, Score) %>%
  dplyr::rename(Score_cv = Score) %>%
  dplyr::mutate(Score_train = train_verification$Assessment_of_Classifier_Effectiveness$Score,
                Score_validation = validation_verification$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification$Assessment_of_Classifier_Effectiveness$Score,
                Cutoff = default_cutoff) %>%
  readr::write_csv2(path = paste(models_store_dir, paste(datetime, model_name, "binary_classification_summary_default_cutoff.csv", sep = "_"), sep = "/"))

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki