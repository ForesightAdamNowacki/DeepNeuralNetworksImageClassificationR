# ------------------------------------------------------------------------------
# EFFICIENTNET B0 BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model:
model_name <- "EfficientNet_B0"
model_type <- "Binary"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'model_name' folder in cwd:
if (base::dir.exists(base::paste(base::getwd(), model_name, sep = "/")) == FALSE){base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))}
# 3. Create 'model_type' subfolder in 'model_name' main folder:
if (base::dir.exists(base::paste(base::getwd(), model_name, model_type, sep = "/")) == FALSE){base::dir.create(path = base::paste(base::getwd(), model_name, model_type, sep = "/"))}

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_1", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Useful_Functions.R")

# Directories:
train_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/train"
validation_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/validation"
test_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/test"
models_store_dir <- base::paste(base::getwd(), model_name, model_type, sep = "/")
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
batch_size <- 16
class_mode <- "categorical"
shuffle <- TRUE
epochs <- 3
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


act
swish_activation <- function(x){
  base::return(keras$backend$sigmoid(x) * x)}
tf$keras$utils$get_custom_objects(update('swish_activation' = keras::layer_activation(swish_activation)))
keras::()

tf$
model <- keras::load_model_hdf5("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/EfficientNet_B0.hdf5")
# ------------------------------------------------------------------------------
# EfficientNet B0 model architecture:
tf <- reticulate::import("tensorflow")
tf$keras$applications()


efficientnet$model$EfficientNetB5
  

keras <- reticulate::import("keras")
keras$activations$
efficientnet <- reticulate::import("efficientnet")
efficientnet$init_keras_custom_objects()
model <- efficientnet$model$EfficientNetB0(include_top = include_top,
                                           weights = weights,
                                           input_shape = base::c(image_size, image_size, channels))

efficientnet$model$EfficientNetB1(include_top = include_top,
                                  weights = weights,
                                  input_shape = base::c(image_size, image_size, channels))


efficientnet$model$EfficientNet





model <- keras::application_xception(include_top = include_top,
                                     weights = weights,
                                     input_shape = base::c(image_size, image_size, channels))

input_tensor <- keras::layer_input(shape = base::c(image_size, image_size, channels))
output_tensor <- input_tensor %>%
  model %>%
  keras::layer_global_average_pooling_2d() %>%
  keras::layer_dense(units = base::length(base::levels(validation_files$category)), activation = activation) 

model <- keras::keras_model(inputs = input_tensor, outputs = output_tensor)






#
efficient <- reticulate::import("efficientnet")
model <- efficient$EfficientNetB5(weights = "imagenet",
                                  include_top = FALSE)
