# ------------------------------------------------------------------------------
# K-FOLD CROSS VALIDATION BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'K-FOLD_CROSS' folder in cwd
base::dir.create(path = base::paste(base::getwd(), "K-Fold_Cross_Validation", sep = "/"))
# 3. Create 'Binary' subfolder in 'K-Fold_Cross_Validation' main folder
base::dir.create(path = base::paste(base::getwd(), "K-Fold_Cross_Validation", "Binary", sep = "/"))

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
models_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR/K-Fold_Cross_Validation/Binary"
data_directory <- "D:/GitHub/Datasets/Cats_And_Dogs_Small/data"
folds_directory <- "D:/GitHub/Datasets/Cats_And_Dogs_Small/K-folds"
base::unlink(folds_directory, recursive = TRUE, force = TRUE)
folds <- 5
Create_KFolds_Directories(data_dir = data_directory,
                          target_dir = folds_directory,
                          folds = folds,
                          seed = 42)
base::setwd(models_store_dir)

folds_dirs <- base::paste(folds_directory, base::list.files(folds_directory)[base::grepl("fold", base::list.files(folds_directory))], sep = "/")
for (i in base::seq_along(folds_dirs)){
  base::cat(folds_dirs[i]); base::cat("\n")
  base::print(Count_Files(path = base::paste(folds_dirs[i], sep = "/")))
  base::cat("\n")}

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

train_files <- Count_Files(base::paste(folds_directory, base::list.files(folds_directory)[base::grepl("step_1", base::list.files(folds_directory))], "train", sep = "/")); train_files
validation_files <- Count_Files(base::paste(folds_directory, base::list.files(folds_directory)[base::grepl("step_1", base::list.files(folds_directory))], "validation", sep = "/")); validation_files

# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Setting pipeline parameters values: 
# # Training:
# batch_size <- 16
# class_mode <- "categorical"
# shuffle <- TRUE
# epochs <- 5
# early_stopping_patience <- 10
# reduce_lr_on_plateu_patience <- 5
# monitor <- "val_loss"
# save_best_only <- TRUE
# if (monitor == "val_loss"){mode <- "min"} else {mode <- "max"}
# verbose <- 1
# write_graph <- TRUE
# write_grads <- TRUE
# write_images <- TRUE
# restore_best_weights <- FALSE
# histogram_freq <- 1
# min_delta <- 0

# ------------------------------------------------------------------------------
# Model architecture:
build_model <- function(image_size = 150, channels = 3, classes = 2,
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
    
    keras::layer_dense(units = classes, activation = activation_1) %>%
    keras::layer_activation(activation = activation_3)
  
  model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)
  
  model %>% keras::compile(loss = loss,
                           optimizer = optimizer, 
                           metrics = metrics)
  base::return(model)}

model <- build_model()

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Cross validation algorithm:

history_list <- base::list()
for (i in 1:folds){
  
  keras::k_clear_session()
  
  image_size <- 150
  classes <- 2
  batch_size <- 32
  class_mode <- "categorical"
  shuffle <- TRUE
  epochs <- 30
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
  
  train_dir <- base::paste0(folds_directory, "/step_", i, "/train")
  validation_dir <- base::paste0(folds_directory, "/step_", i, "/validation")
  callback_model_checkpoint_path <- base::paste(models_store_dir, base::paste("fold", i, "keras_model.weights.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5", sep = "_"), sep = "/")
  callback_tensorboard_path <- base::paste(models_store_dir,  base::paste("fold", i, "logs", sep = "_"), sep = "/")
  callback_csv_logger_path <- base::paste(models_store_dir, base::paste("fold", i, "Optimization_logger.csv", sep = "_"), sep = "/")
  
  base::cat(base::paste("Fold:", i, "\n"))
  base::cat(train_dir, "\n")
  base::cat(validation_dir, "\n")
  
  model <- build_model()
  
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
                                                       classes = base::levels(train_files$category),
                                                       shuffle = shuffle)
  
  validation_datagen <- keras::image_data_generator(rescale = 1/255) 
  validation_generator <- keras::flow_images_from_directory(directory = validation_dir,
                                                            generator = validation_datagen,
                                                            target_size = base::c(image_size, image_size),
                                                            batch_size = batch_size,
                                                            class_mode = class_mode,
                                                            classes = base::levels(train_files$category),
                                                            shuffle = shuffle)
  
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

  
  history_list[[i]] <- history$metrics %>%
    tibble::as_tibble() %>%
    dplyr::mutate(epoch = dplyr::row_number(),
                  fold = i) %>%
    base::as.data.frame() 
  

}


history_list %>%
  base::do.call(dplyr::bind_rows, .) %>%
  readr::write_csv("results.csv")


title_size <- 9
text_size <- 7

history_list %>%
  base::do.call(dplyr::bind_rows, .) %>%
  tibble::as_tibble(.) %>%
  tidyr::pivot_longer(cols = dplyr::contains(base::c("loss", "acc")),
                      names_to = "type",
                      values_to = "value") %>%
  dplyr::mutate(type_2 = base::ifelse(base::grepl("acc", type), "Accuracy", "Loss"),
                set = base::ifelse(base::grepl("val", type), "Validation", "Train"),
                fold = base::factor(fold)) %>%
  dplyr::filter(type_2 == "Accuracy") %>%
  ggplot2::ggplot(data = .) +
  ggplot2::geom_point(mapping = ggplot2::aes(x = epoch, y = value, color = fold), alpha = 0.5) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = value, color = fold), alpha = 0.5, stat = "smooth") +
  ggplot2::geom_smooth(mapping = ggplot2::aes(x = epoch, y = value), se = FALSE, col = "black", lwd = 1) +
  ggplot2::facet_wrap(.~set) +
  ggplot2::labs(x = "Epoch",
                y = "Accuracy",
                title = "Accuracy distribution for cross validation") +
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
                 legend.position = "none",
                 strip.background = element_rect(color = "black", fill = "gray80", size = 0.5, linetype = "solid"),
                 strip.text = element_text(size = text_size, face = "bold"))
  
history_list %>%
  base::do.call(dplyr::bind_rows, .) %>%
  tibble::as_tibble(.) %>%
  tidyr::pivot_longer(cols = dplyr::contains(base::c("loss", "acc")),
                      names_to = "type",
                      values_to = "value") %>%
  dplyr::mutate(type_2 = base::ifelse(base::grepl("acc", type), "Accuracy", "Loss"),
                set = base::ifelse(base::grepl("val", type), "Validation", "Train"),
                fold = base::factor(fold)) %>%
  dplyr::filter(type_2 == "Loss") %>%
  ggplot2::ggplot(data = .) +
  ggplot2::geom_point(mapping = ggplot2::aes(x = epoch, y = value, color = fold), alpha = 0.5) +
  ggplot2::geom_line(mapping = ggplot2::aes(x = epoch, y = value, color = fold), alpha = 0.5, stat = "smooth") +
  ggplot2::geom_smooth(mapping = ggplot2::aes(x = epoch, y = value), se = FALSE, col = "black", lwd = 1) +
  ggplot2::facet_wrap(.~set) +
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
                 legend.position = "none",
                 strip.background = element_rect(color = "black", fill = "gray80", size = 0.5, linetype = "solid"),
                 strip.text = element_text(size = text_size, face = "bold"))
  
























# ------------------------------------------------------------------------------
# Tensorboard:
base::dir.create(path = callback_tensorboard_path)
keras::tensorboard(log_dir = callback_tensorboard_path, host = "127.0.0.1")
# If 'ERROR: invalid version specification':
# 1. Anaconda Prompt
# 2. conda activate GPU_ML_2
# 3. cd D:/GitHub/DeepNeuralNetworksRepoR/Xception/Binary
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
last_model <- base::list.files(path = models_store_dir, pattern = ".hdf5")[base::length(base::list.files(path = models_store_dir, pattern = ".hdf5"))]; last_model
model <- keras::load_model_hdf5(filepath = paste(models_store_dir, last_model, sep = "/"), compile = FALSE)
# model <- keras::load_model_hdf5(filepath = "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Binary_Xception_Model.hdf5", compile = FALSE)
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
readr::write_csv2(tibble::as_tibble(train_probabilities) %>%
                    dplyr::mutate(filepath = train_generator$filepaths), base::paste(datetime, "Xception_train_binary_probabilities.csv"))
readr::write_csv2(tibble::as_tibble(validation_probabilities) %>%
                    dplyr::mutate(filepath = validation_generator$filepaths), base::paste(datetime, "Xception_validation_binary_probabilities.csv"))
readr::write_csv2(tibble::as_tibble(test_probabilities) %>%
                    dplyr::mutate(filepath = test_generator$filepaths), base::paste(datetime, "Xception_test_binary_probabilities.csv"))

# ------------------------------------------------------------------------------
# Model verification - default cutoff:
default_cutoff <- 0.5

train_actual <- base::rep(base::c(0, 1), times = train_files$category_obs)
train_predicted <- train_probabilities[,2]
train_verification_1 <- Binary_Classifier_Verification(actual = train_actual,
                                                       predicted = train_predicted,
                                                       cutoff = default_cutoff,
                                                       type_info = "Train Xception default cutoff",
                                                       save = TRUE,
                                                       open = FALSE)

validation_actual <- base::rep(base::c(0, 1), times = validation_files$category_obs)
validation_predicted <- validation_probabilities[,2]
validation_verification_1 <- Binary_Classifier_Verification(actual = validation_actual,
                                                            predicted = validation_predicted,
                                                            cutoff = default_cutoff,
                                                            type_info = "Validation Xception default cutoff",
                                                            save = TRUE,
                                                            open = FALSE)

test_actual <- base::c(base::rep(0, test_files$category_obs[1]/2), base::rep(1, test_files$category_obs[1]/2))
test_predicted <- test_probabilities[,2]
test_verification_1 <- Binary_Classifier_Verification(actual = test_actual,
                                                      predicted = test_predicted,
                                                      cutoff = default_cutoff,
                                                      type_info = "Test Xception default cutoff",
                                                      save = TRUE,
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
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Summary_Default_Cutoff_Xception.csv"), sep = "/"))

# ------------------------------------------------------------------------------
# Model verification - cutoff optimization on validation set:
train_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = train_actual,
                                                                   predicted = train_predicted,
                                                                   type_info = "Train Xception",
                                                                   seed_value = 42,
                                                                   top = 10,
                                                                   cuts = 100,
                                                                   key_metric = ACC,
                                                                   ascending = FALSE,
                                                                   save = TRUE,
                                                                   open = FALSE)
train_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> train_optimal_cutoff; train_optimal_cutoff

validation_cutoff_optimization <- Binary_Classifier_Cutoff_Optimization(actual = validation_actual,
                                                                        predicted = validation_predicted,
                                                                        type_info = "Validation Xception",
                                                                        seed_value = 42,
                                                                        top = 10,
                                                                        cuts = 100,
                                                                        key_metric = ACC,
                                                                        ascending = FALSE,
                                                                        save = TRUE,
                                                                        open = FALSE)
validation_cutoff_optimization %>%
  dplyr::select(CUTOFF) %>%
  dplyr::pull() %>%
  base::mean() -> validation_optimal_cutoff; validation_optimal_cutoff

selected_cutoff <- validation_optimal_cutoff

train_verification_2 <- Binary_Classifier_Verification(actual = train_actual,
                                                       predicted = train_predicted,
                                                       cutoff = selected_cutoff,
                                                       type_info = "Train Xception optimized cutoff",
                                                       save = TRUE,
                                                       open = FALSE)

validation_verification_2 <- Binary_Classifier_Verification(actual = validation_actual,
                                                            predicted = validation_predicted,
                                                            cutoff = selected_cutoff,
                                                            type_info = "Validation Xception optimized cutoff",
                                                            save = TRUE,
                                                            open = FALSE)

test_verification_2 <- Binary_Classifier_Verification(actual = test_actual,
                                                      predicted = test_predicted,
                                                      cutoff = selected_cutoff,
                                                      type_info = "Test Xception optimized cutoff",
                                                      save = TRUE,
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
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Summary_Optimized_Cutoff_Xception.csv"), sep = "/"))

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
  readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Summary_Comparison_Default_Optimized_Cutoff_Xception.csv"), sep = "/"))

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
# Save true and false predictions:
# Train:
Train_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/train",
                                                                                                    actual_classes = train_actual,
                                                                                                    prediction = train_predicted,
                                                                                                    cwd = models_store_dir,
                                                                                                    cutoff = 0.5,
                                                                                                    save_summary_files = TRUE,
                                                                                                    save_correct_images = FALSE,
                                                                                                    save_incorrect_images = FALSE)

# Validation:
Validation_Correct_Incorrect_Binary_Classifications <- Organize_Correct_Incorrect_Binary_Classifications(dataset_dir = "D:/GitHub/Datasets/Cats_And_Dogs/validation",
                                                                                                         actual_classes = validation_actual,
                                                                                                         prediction = validation_predicted,
                                                                                                         cwd = models_store_dir,
                                                                                                         cutoff = 0.5,
                                                                                                         save_summary_files = TRUE,
                                                                                                         save_correct_images = FALSE,
                                                                                                         save_incorrect_images = FALSE)

# ------------------------------------------------------------------------------
# Visualize predictions distribution:
labels <- base::sort(base::as.character(train_files$category)); labels
train_predicted_2 <- train_probabilities[base::matrix(data = base::c(1:base::nrow(train_probabilities), train_actual + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = train_actual,
                                              predicted = train_predicted_2,
                                              labels = labels,
                                              bins = 10)

validation_predicted_2 <- validation_probabilities[base::matrix(data = base::c(1:base::nrow(validation_probabilities), validation_actual + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = validation_actual,
                                              predicted = validation_predicted_2,
                                              labels = labels,
                                              bins = 10)

test_predicted_2 <- test_probabilities[base::matrix(data = base::c(1:base::nrow(test_probabilities), test_actual + 1), byrow = FALSE, ncol = 2)]
Display_Target_Class_Predictions_Distribution(actual = test_actual,
                                              predicted = test_predicted_2,
                                              labels = labels,
                                              bins = 10)

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to all classes:
train_actual_2 <- train_actual + 1
Display_All_Classes_Predictions_Distribution(actual = train_actual_2,
                                             predicted = train_probabilities,
                                             labels = labels,
                                             bins = 10)

validation_actual_2 <- validation_actual + 1
Display_All_Classes_Predictions_Distribution(actual = validation_actual_2,
                                             predicted = validation_probabilities,
                                             labels = labels,
                                             bins = 10)

test_actual_2 <- test_actual + 1
Display_All_Classes_Predictions_Distribution(actual = test_actual_2,
                                             predicted = test_probabilities,
                                             labels = labels,
                                             bins = 10)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki