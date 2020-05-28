# ------------------------------------------------------------------------------
# K-FOLD CROSS VALIDATION CATEGORICAL MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/cifar-10/overview
utils::browseURL(url = "https://www.kaggle.com/c/cifar-10/overview")

# ------------------------------------------------------------------------------
# Model:
model_name <- "Cross_Validation"
model_type <- "Categorical"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'K-FOLD_CROSS' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))
# 3. Create 'Categorical' subfolder in 'K-Fold_Cross_Validation' main folder
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
train_dir <- "D:/GitHub/Datasets/Cifar10/train"
validation_dir <- "D:/GitHub/Datasets/Cifar10/validation"
test_dir <- "D:/GitHub/Datasets/Cifar10/test"
cv_data_dir <- "D:/GitHub/Datasets/Cifar10_Small/data"
models_store_dir <-  base::paste(base::getwd(), model_name, model_type, sep = "/")
repo_models_store_dir <- base::paste("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store", model_name, model_type, sep = "/")
base::dir.create(path = base::paste("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store", model_name, sep = "/"))
base::dir.create(path = base::paste("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store", model_name, model_type, sep = "/"))
folds_directory <- "D:/GitHub/Datasets/Cifar10_Small/folds"

# ------------------------------------------------------------------------------
# Create folds:
folds <- 5
Create_KFolds_Directories(data_dir = cv_data_dir,
                          target_dir = folds_directory,
                          folds = folds,
                          seed = 42)

# Display number of observations per class in each fold:
folds_dirs <- base::paste(folds_directory, base::list.files(path = folds_directory, pattern = "fold"), sep = "/")
for (i in base::seq_along(folds_dirs)){
  base::cat(folds_dirs[i]); base::cat("\n")
  base::print(Count_Files(path = base::paste(folds_dirs[i], sep = "/")))
  base::cat("\n")}

# Display number of observations per dataset type (train, validation) and class in each step of cross validation:
steps_dirs <- base::expand.grid(base::paste(folds_directory, base::list.files(folds_directory)[base::grepl("step", base::list.files(folds_directory))], sep = "/"),
                                base::c("train", "validation")) %>%
  dplyr::mutate(steps_dirs = base::paste(Var1, Var2, sep = "/")) %>%
  dplyr::select(steps_dirs) %>%
  dplyr::pull() %>%
  base::sort()
for (i in base::seq_along(steps_dirs)){
  base::cat(steps_dirs[i]); base::cat("\n")
  base::print(Count_Files(path = base::paste(steps_dirs[i], sep = "/")))
  base::cat("\n")}

# Count train files:
train_steps_dirs <- steps_dirs[base::grepl("train", steps_dirs)]
train_steps_files <- base::list()
for (i in 1:folds){train_steps_files[[i]] <- Count_Files(path = train_steps_dirs[i])}
train_steps_files

# Count validation files:
validation_steps_dirs <- steps_dirs[base::grepl("validation", steps_dirs)]
validation_steps_files <- base::list()
for (i in 1:folds){validation_steps_files[[i]] <- Count_Files(path = validation_steps_dirs[i])}
validation_steps_files

# Main datasets:
train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# Set current working directory:
base::setwd(models_store_dir)

# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Model architecture:
build_model <- function(image_size = 150, channels = 3,
                        activation_1 = "linear", activation_2 = "relu", activation_3 = "softmax",
                        loss = "categorical_crossentropy", optimizer = keras::optimizer_adam(), metrics = base::c("acc")){
  
  input_tensor <- keras::layer_input(shape = base::c(image_size, image_size, channels))
  output_tensor <- input_tensor %>%
    keras::layer_conv_2d(filters = 32, kernel_size = base::c(3, 3), strides = base::c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 32, kernel_size = base::c(3, 3), strides = base::c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = 2) %>%
    
    keras::layer_conv_2d(filters = 64, kernel_size = base::c(3, 3), strides = base::c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 64, kernel_size = base::c(3, 3), strides = base::c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_max_pooling_2d(pool_size = 2) %>%
    
    keras::layer_conv_2d(filters = 128, kernel_size = base::c(3, 3), strides = base::c(1, 1), activation = activation_1, padding = "same") %>%
    keras::layer_activation(activation = activation_2) %>%
    keras::layer_batch_normalization() %>%
    keras::layer_conv_2d(filters = 128, kernel_size = base::c(3, 3), strides = base::c(1, 1), activation = activation_1, padding = "same") %>%
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
    
    keras::layer_dense(units = base::length(base::list.files(path = cv_data_dir)), activation = activation_1) %>%
    keras::layer_activation(activation = activation_3)
  
  model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)
  
  model %>% keras::compile(loss = loss,
                           optimizer = optimizer, 
                           metrics = metrics)
  base::return(model)}

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

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
  
  datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  if (monitor == "val_loss"){mode <- "min"} else {mode <- "max"}
  history_list <- base::list()
  
  for (i in 1:folds){
    
    keras::k_clear_session()
    
    train_dir <- base::paste0(folds_directory, "/step_", i, "/train")
    validation_dir <- base::paste0(folds_directory, "/step_", i, "/validation")
    callback_model_checkpoint_path <- base::paste(models_store_dir, base::paste("fold", i, "keras_model.weights.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5", sep = "_"), sep = "/")
    callback_tensorboard_path <- base::paste(models_store_dir,  base::paste("fold", i, "logs", sep = "_"), sep = "/")
    callback_csv_logger_path <- base::paste(models_store_dir, base::paste("fold", i, "optimization_logger.csv", sep = "_"), sep = "/")
    
    base::cat(base::paste("Fold:", i, "\n"))
    base::cat("Train directory:", train_dir, "\n")
    base::cat("Validation directory:", validation_dir, "\n")
    
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
                                                         classes = base::list.files(path = cv_data_dir),
                                                         shuffle = shuffle)
    
    validation_datagen <- keras::image_data_generator(rescale = 1/255) 
    validation_generator <- keras::flow_images_from_directory(directory = validation_dir,
                                                              generator = validation_datagen,
                                                              target_size = base::c(image_size, image_size),
                                                              batch_size = batch_size,
                                                              class_mode = class_mode,
                                                              classes = base::list.files(path = cv_data_dir),
                                                              shuffle = shuffle)
    
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
    history_list[[i]] <- history$metrics %>%
      tibble::as_tibble() %>%
      dplyr::mutate(epoch = dplyr::row_number(),
                    fold = i) %>%
      tibble::as_tibble()
    
    
    files <- base::paste(base::getwd(), base::list.files(path = base::getwd(), pattern = base::paste("fold", i, sep = "_")), sep = "/")
    files <- base::sort(files[base::grepl(".hdf5", files)])
    for (j in 1:(base::length(files) - 1)){
      base::cat("Remove .hdf5 file:", files[j], "\n")
      base::unlink(files[j], recursive = TRUE, force = TRUE)}
  }
  
  history_list %>%
    base::do.call(dplyr::bind_rows, .) %>%
    readr::write_csv(base::paste(datetime, "CV_results.csv"))
}

Cross_Validation_Pipe_Runner(epochs = 5,
                             folds_directory = folds_directory,
                             models_store_dir = models_store_dir)

# ------------------------------------------------------------------------------
# Remove logs folders if are not important:
logs_folders <- base::paste(base::getwd(), base::list.files(path = base::getwd(), pattern = "logs"), sep = "/")
for (i in base::seq_along(logs_folders)){
  base::cat("Remove logs folder:", logs_folders[i], "\n")
  base::unlink(logs_folders[i], force = TRUE, recursive = TRUE)}

# ------------------------------------------------------------------------------
# Save optimal models in local models repository: 
optimal_models <- base::paste(base::getwd(), base::list.files(path = base::getwd(), pattern = ".hdf5"), sep = "/")
base::file.copy(from = optimal_models,
                to = base::paste(repo_models_store_dir, base::basename(optimal_models), sep = "/"))
for (i in base::seq_along(optimal_models)){
  base::cat("Remove model:", optimal_models[i], "\n")
  base::unlink(optimal_models[i])}
optimal_models_repo_store <- base::paste(repo_models_store_dir, base::basename(optimal_models), sep = "/")
for (i in base::seq_along(optimal_models_repo_store)){
  base::cat("Fold", i, "optimal model directory:", optimal_models_repo_store[i], "\n")}

# ------------------------------------------------------------------------------
# Cross validation results:
results <- readr::read_csv(base::list.files(base::getwd(), pattern = "CV_results.csv")); results

# ------------------------------------------------------------------------------
# Display cross validation results:
title_size <- 9
text_size <- 7

# Accuracy:
results %>%
  dplyr::select(dplyr::contains("acc"), epoch, fold) %>%
  tidyr::pivot_longer(cols = base::c("acc", "val_acc"),
                      names_to = "type",
                      values_to = "value") %>%
  dplyr::mutate(type = dplyr::case_when(type == "acc" ~ "Train accuracy",
                                        type == "val_acc" ~ "Validation accuracy")) %>%
  dplyr::group_by(epoch, type) %>%
  dplyr::mutate(median = stats::median(value),
                mean = base::mean(value),
                fold = base::factor(fold)) -> results_accuracy; results_accuracy

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
  tidyr::pivot_longer(cols = base::c("loss", "val_loss"),
                      names_to = "type",
                      values_to = "value") %>%
  dplyr::mutate(type = dplyr::case_when(type == "loss" ~ "Train loss",
                                        type == "val_loss" ~ "Validation loss")) %>%
  dplyr::group_by(epoch, type) %>%
  dplyr::mutate(median = stats::median(value),
                mean = base::mean(value),
                fold = base::factor(fold)) -> results_loss; results_loss

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
  
  for (i in base::seq_along(optimal_models_repo_store)){
    print(optimal_models_repo_store[i])
    model <- build_model() %>%
      keras::load_model_weights_hdf5(optimal_models_repo_store[i]) %>%
      keras::compile(loss = "categorical_crossentropy",
                     optimizer = keras::optimizer_adam(), 
                     metrics = base::c("acc"))
    
    datagen <- keras::image_data_generator(rescale = 1/255)
    generator <- keras::flow_images_from_directory(directory = data_dir,
                                                   generator = datagen,
                                                   target_size = base::c(model$input_shape[[2]], model$input_shape[[2]]),
                                                   batch_size = batch_size,
                                                   class_mode = "categorical",
                                                   shuffle = FALSE)
    prediction <- keras::predict_generator(model, generator, steps = base::ceiling(generator$n/generator$batch_size), verbose = 1)
    prediction <- prediction %>%
      tibble::as_tibble()
    
    filename <- base::paste(base::getwd(), base::paste(type, "prediction", i, "model.csv", sep = "_"), sep = "/")
    prediction %>% 
      dplyr::mutate(filepath = generator$filepaths,
                    actual_class = generator$classes) %>%
      readr::write_csv(filename)
    
    base::cat("Save file:", filename, "\n")}}

Predict_Folds_Models(data_dir = cv_data_dir, type = "cv_data")
Predict_Folds_Models(data_dir = train_dir, type = "train_data")
Predict_Folds_Models(data_dir = validation_dir, type = "validation_data")
Predict_Folds_Models(data_dir = test_dir, type = "test_data")

# ------------------------------------------------------------------------------
# Model verification:
labels <- base::list.files(cv_data_dir); labels
save_option <- FALSE

# Cross validation data:
cv_results <- base::list()
for (i in 1:folds){
  cv_results[[i]] <- readr::read_csv(base::list.files(pattern = "cv_data")[i])
  cv_results[[i]]$model <- base::paste("model", i, sep = "_")}
cv_results %>%
  base::do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                      names_to = "predicted_class",
                      values_to = "prediction") %>%
  dplyr::group_by(filepath, predicted_class) %>%
  dplyr::summarise(actual_class = base::mean(actual_class),
                   prediction = base::mean(prediction)) %>%
  dplyr::mutate(predicted_class = base::as.integer(stringr::str_sub(predicted_class, 2, -1))) %>%
  dplyr::arrange(filepath, predicted_class) %>%
  tidyr::pivot_wider(id_cols = base::c("filepath", "actual_class"),
                     names_from = "predicted_class",
                     values_from = "prediction",
                     names_prefix = "V") %>%
  dplyr::ungroup() -> cv_results; cv_results
  
cv_verification <- Categorical_Classifier_Verification(actual = cv_results$actual_class,
                                                       probabilities = cv_results %>%
                                                         dplyr::select(dplyr::starts_with("V")),
                                                       labels = labels,
                                                       type_info = "CV_data_default_cutoff",
                                                       save = save_option,
                                                       open = FALSE)

# Train data:
train_results <- base::list()
for (i in 1:folds){
  train_results[[i]] <- readr::read_csv(base::list.files(pattern = "train_data")[i])
  train_results[[i]]$model <- base::paste("model", i, sep = "_")}
train_results %>%
  base::do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                      names_to = "predicted_class",
                      values_to = "prediction") %>%
  dplyr::group_by(filepath, predicted_class) %>%
  dplyr::summarise(actual_class = base::mean(actual_class),
                   prediction = base::mean(prediction)) %>%
  dplyr::mutate(predicted_class = base::as.integer(stringr::str_sub(predicted_class, 2, -1))) %>%
  dplyr::arrange(filepath, predicted_class) %>%
  tidyr::pivot_wider(id_cols = base::c("filepath", "actual_class"),
                     names_from = "predicted_class",
                     values_from = "prediction",
                     names_prefix = "V") %>%
  dplyr::ungroup() -> train_results; train_results

train_verification <- Categorical_Classifier_Verification(actual = train_results$actual_class,
                                                          probabilities = train_results %>%
                                                            dplyr::select(dplyr::starts_with("V")),
                                                          labels = labels,
                                                          type_info = "Train_data_default_cutoff",
                                                          save = save_option,
                                                          open = FALSE)

# Validation data:
validation_results <- base::list()
for (i in 1:folds){
  validation_results[[i]] <- readr::read_csv(base::list.files(pattern = "validation_data")[i])
  validation_results[[i]]$model <- base::paste("model", i, sep = "_")}
validation_results %>%
  base::do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                      names_to = "predicted_class",
                      values_to = "prediction") %>%
  dplyr::group_by(filepath, predicted_class) %>%
  dplyr::summarise(actual_class = base::mean(actual_class),
                   prediction = base::mean(prediction)) %>%
  dplyr::mutate(predicted_class = base::as.integer(stringr::str_sub(predicted_class, 2, -1))) %>%
  dplyr::arrange(filepath, predicted_class) %>%
  tidyr::pivot_wider(id_cols = base::c("filepath", "actual_class"),
                     names_from = "predicted_class",
                     values_from = "prediction",
                     names_prefix = "V") %>%
  dplyr::ungroup() -> validation_results; validation_results

validation_verification <- Categorical_Classifier_Verification(actual = validation_results$actual_class,
                                                          probabilities = validation_results %>%
                                                            dplyr::select(dplyr::starts_with("V")),
                                                          labels = labels,
                                                          type_info = "Validation_data_default_cutoff",
                                                          save = save_option,
                                                          open = FALSE)

# Test data:
test_results <- base::list()
for (i in 1:folds){
  test_results[[i]] <- readr::read_csv(base::list.files(pattern = "test_data")[i])
  test_results[[i]]$model <- base::paste("model", i, sep = "_")}
test_results %>%
  base::do.call(dplyr::bind_rows, .) %>%
  dplyr::group_by(filepath) %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                      names_to = "predicted_class",
                      values_to = "prediction") %>%
  dplyr::group_by(filepath, predicted_class) %>%
  dplyr::summarise(actual_class = base::mean(actual_class),
                   prediction = base::mean(prediction)) %>%
  dplyr::mutate(predicted_class = base::as.integer(stringr::str_sub(predicted_class, 2, -1))) %>%
  dplyr::arrange(filepath, predicted_class) %>%
  tidyr::pivot_wider(id_cols = base::c("filepath", "actual_class"),
                     names_from = "predicted_class",
                     values_from = "prediction",
                     names_prefix = "V") %>%
  dplyr::ungroup() -> test_results; test_results

test_verification <- Categorical_Classifier_Verification(actual = test_results$actual_class,
                                                               probabilities = test_results %>%
                                                                 dplyr::select(dplyr::starts_with("V")),
                                                               labels = labels,
                                                               type_info = "Test_data_default_cutoff",
                                                               save = save_option,
                                                               open = FALSE)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki