# ------------------------------------------------------------------------------
# CNN BINARY MODEL IMPLEMENTATION - HYPERPARAMETERS OPTIMIZATION (1)
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
# utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "CNN_Hyperparameters_Optimization"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'CNN' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))
# 3. Create 'Binary' subfolder in 'CNN' main folder
base::dir.create(path = base::paste(base::getwd(), model_name, "Binary", sep = "/"))

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::library(tfruns)
base::library(tfestimators)
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Useful_Functions.R")

# Directories:
train_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/train"
validation_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/validation"
test_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/test"
models_store_dir <- base::paste(base::getwd(), model_name, "Binary", sep = "/")
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"

train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# ------------------------------------------------------------------------------
# Clear session:
keras::k_clear_session()

# ------------------------------------------------------------------------------
# Optimize model's hyperparameters:
Hyperparametrization_Optimization_Results <- Hyperparametrization_Optimization(hyperparameters_list = base::list(filters_1 = base::c(64, 128),
                                                                                                                 filters_2 = base::c(64, 128),
                                                                                                                 filters_3 = base::c(128, 256),
                                                                                                                 filters_4 = base::c(128, 256),
                                                                                                                 dense_units_1 = base::c(256, 512),
                                                                                                                 dense_units_2 = base::c(256, 512)),
                                  script_directory = "Binary_Classification_CNN_Hyperparameters_Optimization_2.R")

Hyperparametrization_Optimization_Results %>%
  dplyr::arrange(dplyr::desc(val_acc))
