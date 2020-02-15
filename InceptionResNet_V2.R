# ------------------------------------------------------------------------------
# INCEPTIONRESNET_V2 MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")
# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)

train_dir <- "C:\\Users\\admin\\Desktop\\GitHub\\Datasets\\Cats_And_Dogs\\train"
validation_dir <- "C:\\Users\\admin\\Desktop\\GitHub\\Datasets\\Cats_And_Dogs\\validation"
test_dir <- "C:\\Users\\admin\\Desktop\\GitHub\\Datasets\\Cats_And_Dogs\\test"
callback_model_checkpoint_path <- "C:\\Users\\admin\\Desktop\\GitHub\\Models_Store\\InceptionResNet_V2-Models\\keras_model.weights.{epoch:02d}-{val_acc:.2f}.hdf5"
callback_tensorboard_path <- "C:\\Users\\admin\\Desktop\\GitHub\\Models_Store\\InceptionResNet_V2-Models"
callback_csv_logger_path <- "C:\\Users\\admin\\Desktop\\GitHub\\Models_Store\\InceptionResNet_V2-Models\\Optimization_logger.csv"

count_files = function(path){
  dirs <- base::list.dirs(path = path)
  dirs <- dirs[2:base::length(dirs)]
  files <- base::integer(base::length(dirs))
  folder <- base::character(base::length(dirs))
  for (i in base::seq_along(dirs)){
    files[i] <- base::length(base::list.files(path = dirs[i]))
    folder[i] <- base::basename(path = dirs[i])}
  result = base::data.frame(category = folder, category_obs = files)
  return(result)
}
train_files <- count_files(path = train_dir); train_files
validation_files <- count_files(path = validation_dir); validation_files
test_files <- count_files(path = test_dir); test_files

# ------------------------------------------------------------------------------
# InceptionResNet_V2 model architecture:
inception_resnet_v2_model <- keras::application_inception_resnet_v2(include_top = FALSE, weights = NULL, input_shape = c(299, 299, 3))

input_tensor <- keras::layer_input(shape = c(299, 299, 3))
output_tensor <- input_tensor %>%
  inception_resnet_v2_model %>%
  keras::layer_global_average_pooling_2d() %>%
  keras::layer_dense(units = base::length(base::levels(validation_files$category)), activation = "softmax") 

inception_resnet_v2_model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)
inception_resnet_v2_model

# ------------------------------------------------------------------------------
# Upload pre-trained model for training:
models_store <- "C:\\Users\\admin\\Desktop\\GitHub\\Models_Store\\InceptionResNet_V2-Models"
last_model <- base::list.files(path = models_store, pattern = ".hdf5")[base::length(base::list.files(path = models_store, pattern = ".hdf5"))]
inception_resnet_v2_model <- keras::load_model_hdf5(filepath = paste(models_store, last_model, sep = "\\"), compile = TRUE)

# ------------------------------------------------------------------------------
# Model compilation:
inception_resnet_v2_model %>% keras::compile(
  loss = "categorical_crossentropy",
  optimizer = keras::optimizer_adam(), 
  metrics = c("acc"))

# ------------------------------------------------------------------------------
# Generators:
# train:
train_datagen <- keras::image_data_generator(
  rescale = 1/255,
  rotation_range = 25, 
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  shear_range = 0.1,
  zoom_range = 0.1,
  brightness_range = base::c(0.5, 1.5),
  horizontal_flip = TRUE,
  vertical_flip = FALSE,
  fill_mode = "nearest")
train_generator <- keras::flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen, 
  target_size = base::c(299, 299),
  batch_size = 8,
  class_mode = "categorical",
  classes = base::levels(validation_files$category),
  shuffle = TRUE)

# validation:
validation_datagen <- keras::image_data_generator(rescale = 1/255) 
validation_generator <- keras::flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = base::c(299, 299),
  batch_size = 8,
  class_mode = "categorical",
  classes = base::levels(validation_files$category),
  shuffle = TRUE)

# ------------------------------------------------------------------------------
# Model optimization:
history <- inception_resnet_v2_model %>% keras::fit_generator(
  generator = train_generator,
  steps_per_epoch = base::ceiling(base::sum(train_files$category_obs)/train_generator$batch_size), 
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = base::ceiling(base::sum(validation_files$category_obs)/train_generator$batch_size), 
  callbacks = c(keras::callback_model_checkpoint(filepath = callback_model_checkpoint_path,
                                                 monitor = "val_acc",
                                                 verbose = 1,
                                                 save_best_only = TRUE,
                                                 mode = "max"),
                keras::callback_early_stopping(monitor = "val_acc",
                                               min_delta = 0,
                                               verbose = 1,
                                               patience = 15,
                                               restore_best_weights = FALSE),
                keras::callback_tensorboard(log_dir = callback_tensorboard_path,
                                            histogram_freq = 1,
                                            write_graph = TRUE,
                                            write_grads = TRUE,
                                            write_images = TRUE),
                keras::callback_csv_logger(filename = callback_csv_logger_path,
                                           separator = ";",
                                           append = TRUE)))

# ------------------------------------------------------------------------------
# Model verification:
models_store <- "C:\\Users\\admin\\Desktop\\GitHub\\Models_Store\\InceptionResNet_V2-Models"
last_model <- base::list.files(path = models_store, pattern = ".hdf5")[base::length(base::list.files(path = models_store, pattern = ".hdf5"))]
best_model <- keras::load_model_hdf5(filepath = paste(models_store, last_model, sep = "\\"), compile = TRUE)
source("C:\\Users\\admin\\Desktop\\GitHub\\DeepNeuralNetworks\\Binary_Model_Evaluation.R")

train_datagen <- keras::image_data_generator(rescale = 1/255)
train_generator <- keras::flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen, 
  target_size = base::c(299, 299),
  batch_size = 8,
  class_mode = "categorical",
  classes = base::levels(validation_files$category),
  shuffle = FALSE)

validation_datagen <- keras::image_data_generator(rescale = 1/255)
validation_generator <- keras::flow_images_from_directory(
  directory = validation_dir,
  generator = validation_datagen,
  target_size = base::c(299, 299),
  batch_size = 8,
  class_mode = "categorical",
  classes = base::levels(validation_files$category),
  shuffle = FALSE)

test_datagen <- keras::image_data_generator(rescale = 1/255)
test_generator <- keras::flow_images_from_directory(
  directory = test_dir,
  generator = test_datagen,
  target_size = base::c(299, 299),
  batch_size = 8,
  class_mode = "categorical",
  shuffle = FALSE)

train_probabilities <- keras::predict_generator(best_model, train_generator, steps = base::ceiling(base::sum(train_files$category_obs)/train_generator$batch_size), verbose = 1)
validation_probabilities <- keras::predict_generator(best_model, validation_generator, steps = base::ceiling(base::sum(validation_files$category_obs)/validation_generator$batch_size), verbose = 1)
test_probabilities <- keras::predict_generator(best_model, test_generator, steps = base::ceiling(base::sum(test_files$category_obs)/test_generator$batch_size), verbose = 1)

# Train:
train_files <- count_files(path = train_dir)
train_actual <- base::c(base::rep(0, train_files$category_obs[1]), base::rep(1, train_files$category_obs[2]))
train_predicted <- train_probabilities[,2]
train_verification <- Binary_Classifier_Verification(actual = train_actual, predicted = train_predicted)
Binary_Classifier_Cutoff_Optimization(actual = train_actual, predicted = train_predicted, cuts = 50)

# Validation:
validation_files <- count_files(path = validation_dir)
validation_actual <- base::c(base::rep(0, validation_files$category_obs[1]), base::rep(1, validation_files$category_obs[2]))
validation_predicted <- validation_probabilities[,2]
validation_verification <- Binary_Classifier_Verification(actual = validation_actual, predicted = validation_predicted)
Binary_Classifier_Cutoff_Optimization(actual = validation_actual, predicted = validation_predicted, cuts = 50)

# Test:
test_files <- count_files(path = test_dir)
test_actual <- base::c(base::rep(0, test_files$category_obs[1]/2), base::rep(1, test_files$category_obs[1]/2))
test_predicted <- test_probabilities[,2]
test_verification <- Binary_Classifier_Verification(actual = test_actual, predicted = test_predicted)
Binary_Classifier_Cutoff_Optimization(actual = test_actual, predicted = test_predicted, cuts = 50)

# Summary:
train_verification$Assessment_of_Classifier_Effectiveness %>%
  dplyr::select(ID, Metric, Score) %>%
  dplyr::rename(Score_train = Score) %>%
  dplyr::mutate(Score_validation = validation_verification$Assessment_of_Classifier_Effectiveness$Score,
                Score_test = test_verification$Assessment_of_Classifier_Effectiveness$Score) %>%
  knitr::kable(.)
# ------------------------------------------------------------------------------