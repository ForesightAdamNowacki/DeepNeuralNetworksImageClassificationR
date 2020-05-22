# ------------------------------------------------------------------------------
# ENSEMBLE CATEGORICAL MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Script variables:
model_name <- "Ensemble_Model"
model_type <- "Categorical"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'CNN' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))
# 3. Create 'Binary' subfolder in 'CNN' main folder
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
models_store_dir <- base::paste(base::getwd(), model_name, model_type, sep = "/")
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"

train_files <- Count_Files(path = train_dir); train_files
validation_files <- Count_Files(path = validation_dir); validation_files
test_files <- Count_Files(path = test_dir); test_files

# ------------------------------------------------------------------------------

abc <- Categorical_Ensemble_Model(models_vector = base::c("Xception", "Inception_V3", "Inception_ResNet_V2"),
                                  labels = base::as.character(train_files$category),
                                  optimization_dataset = "train",
                                  save_option = FALSE,
                                  weights = 50,
                                  key_metric = "Accuracy", 
                                  key_metric_as_string = TRUE,
                                  ascending = FALSE,
                                  top = 10,
                                  seed = 42,
                                  summary_type = "mean",
                                  cwd = models_store_dir,
                                  n = 3)

abc











