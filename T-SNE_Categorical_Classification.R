# ------------------------------------------------------------------------------
# T-SNE EMBEDDINGS VISUALISATION FOR CATEGORICAL IMAGE CLASSIFICATION:
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/cifar-10/overview
# browseURL(url = "https://www.kaggle.com/c/cifar-10/overview")

# ------------------------------------------------------------------------------
# Model:
model_name <- "Xception"
model_type <- "Categorical"
folder_name <- "T-SNE_Dimensionality_Reduction"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
# 2. Create 'VGG16' folder in cwd
if (dir.exists(paste(getwd(), folder_name, sep = "/")) == FALSE){dir.create(path = paste(getwd(), folder_name, sep = "/"))}

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
library(tensorflow)
library(keras)
library(tidyverse)
library(deepviz)
library(Rtsne)
source("D:/GitHub/DeepNeuralNetworksImageClassificationR/Useful_Functions.R")

# Directories:
train_dir <- "D:/GitHub/Datasets/Cifar10/train"
validation_dir <- "D:/GitHub/Datasets/Cifar10/validation"
test_dir <- "D:/GitHub/Datasets/Cifar10/test"
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"

# ------------------------------------------------------------------------------
# Clear session and import the best trained model:
keras::k_clear_session()
optimal_model_repo_dir <- paste(models_repo_store_dir, paste(model_type, model_name, "Model.hdf5", sep = "_"), sep = "/")
model <- keras::load_model_hdf5(filepath = optimal_model_repo_dir, compile = FALSE)

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% summary()

# ------------------------------------------------------------------------------
# Model compilation:
loss <- "categorical_crossentropy"
optimizer <- keras::optimizer_adam()
metrics <- c("acc")
model %>% keras::compile(loss = loss,
                         optimizer = optimizer, 
                         metrics = metrics)

# ------------------------------------------------------------------------------
# T-SNE dimensionality reduction:
setwd(paste(getwd(), folder_name, sep = "/"))
save_plot <- TRUE

T_SNE_Dimensionality_Reduction_Visualisation(data_dir = train_dir,
                                             model = model, 
                                             type_info = paste(model_name, model_type, "train", sep = "_"),
                                             save_plot = save_plot)
T_SNE_Dimensionality_Reduction_Visualisation(data_dir = validation_dir,
                                             model = model, 
                                             type_info = paste(model_name, model_type, "validation", sep = "_"),
                                             save_plot = save_plot)
T_SNE_Dimensionality_Reduction_Visualisation(data_dir = test_dir,
                                             model = model, 
                                             type_info = paste(model_name, model_type, "test", sep = "_"),
                                             save_plot = save_plot)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki