# ------------------------------------------------------------------------------
# CNN BINARY MODEL IMPLEMENTATION - HYPERPARAMETERS OPTIMIZATION (2)
# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Directories:
# callback_model_checkpoint_path <- base::paste(models_store_dir, base::paste(i, "keras_model.weights.{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5", sep = "_"), sep = "/")
callback_model_checkpoint_path <- base::paste(models_store_dir, base::paste("{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}_keras_model_weights", i, "logs.hdf5", sep = "_"), sep = "/")
callback_tensorboard_path <- base::paste(models_store_dir, base::paste(i, "logs", sep = "_"), sep = "/")
callback_csv_logger_path <- base::paste(models_store_dir, base::paste0(base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), i, model_name, hyperparameter_vector, sep = "_"), ".csv"), sep = "/")

# ------------------------------------------------------------------------------
# Setting pipeline parameters values: 
# Image:
image_size <- 64
channels <- 3

# Model structure:
activation_1 <- "relu"
activation_2 <- "linear"
activation_3 <- "softmax"
dropout_rate <- 0.5

# Model compilation:
loss <- "categorical_crossentropy"
optimizer <- keras::optimizer_adam()
metrics <- base::c("acc")

# Training:
batch_size <- 128
class_mode <- "categorical"
shuffle <- TRUE
epochs <- 10
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
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# CNN model architecture:
input_tensor <- keras::layer_input(shape = base::c(image_size, image_size, channels))
output_tensor <- input_tensor %>%
  
  keras::layer_conv_2d(filters = filters_1, kernel_size = base::c(3, 3), strides = base::c(1, 1), padding = "same", activation = activation_2) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_conv_2d(filters = filters_2, kernel_size = base::c(3, 3), strides = base::c(1, 1), padding = "same", activation = activation_2) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_max_pooling_2d(pool_size = base::c(2, 2), strides = base::c(2, 2), padding = "valid") %>%
  
  keras::layer_conv_2d(filters = filters_3, kernel_size = base::c(3, 3), strides = base::c(1, 1), padding = "same", activation = activation_2) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_conv_2d(filters = filters_4, kernel_size = base::c(3, 3), strides = base::c(1, 1), padding = "same", activation = activation_2) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_max_pooling_2d(pool_size = base::c(2, 2), strides = base::c(2, 2), padding = "valid") %>%
  
  keras::layer_flatten() %>%
  
  keras::layer_dense(units = dense_units_1, activation = activation_2) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_dropout(rate = dropout_rate) %>%
  
  keras::layer_dense(units = dense_units_2, activation = activation_2) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation = activation_1) %>%
  keras::layer_dropout(rate = dropout_rate) %>%
  
  keras::layer_dense(units = base::length(base::levels(train_files$category)), activation = activation_3)

model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)

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
# Model optimization:
history_[[i]] <- model %>% keras::fit_generator(generator = train_generator,
                                          steps_per_epoch = base::ceiling(train_generator$n/batch_size), 
                                          epochs = epochs,
                                          validation_data = validation_generator,
                                          validation_steps = base::ceiling(validation_generator$n/batch_size), 
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

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki