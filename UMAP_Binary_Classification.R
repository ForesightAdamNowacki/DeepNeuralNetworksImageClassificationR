# ------------------------------------------------------------------------------
# UMAP EMBEDDINGS VISUALISATION FOR BINARY IMAGE CLASSIFICATION:
# ------------------------------------------------------------------------------
# Data
# https://www.kaggle.com/c/dogs-vs-cats
# utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model:
model_name <- "Xception"
model_type <- "Binary"
folder_name <- "UMAP_Dimensionality_Reduction"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
# 2. Create 'VGG16' folder in cwd
if (base::dir.exists(base::paste(base::getwd(), folder_name, sep = "/")) == FALSE){base::dir.create(path = base::paste(base::getwd(), folder_name, sep = "/"))}

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::library(uwot)
base::source("D:/GitHub/DeepNeuralNetworksImageClassificationR/Useful_Functions.R")

# Directories:
train_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/train"
validation_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/validation"
test_dir <- "D:/GitHub/Datasets/Cats_And_Dogs/test"
models_repo_store_dir <- "D:/GitHub/DeepNeuralNetworksRepoR_Models_Store"

# ------------------------------------------------------------------------------
# Clear session and import the best trained model:
keras::k_clear_session()
optimal_model_repo_dir <- base::paste(models_repo_store_dir, base::paste(model_type, model_name, "Model.hdf5", sep = "_"), sep = "/")
model <- keras::load_model_hdf5(filepath = optimal_model_repo_dir, compile = FALSE)

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Model compilation:
loss <- "categorical_crossentropy"
optimizer <- keras::optimizer_adam()
metrics <- base::c("acc")
model %>% keras::compile(loss = loss,
                         optimizer = optimizer, 
                         metrics = metrics)

# ------------------------------------------------------------------------------
# UMAP dimensionality reduction:
base::setwd(base::paste(base::getwd(), folder_name, sep = "/"))
save_plot <- TRUE

UMAP_Dimensionality_Reduction_Visualisation(data_dir = train_dir,
                                            model = model, 
                                            type_info = base::paste(model_name, model_type, "train", sep = "_"),
                                            save_plot = save_plot)
UMAP_Dimensionality_Reduction_Visualisation(data_dir = validation_dir,
                                            model = model, 
                                            type_info = base::paste(model_name, model_type, "validation", sep = "_"),
                                            save_plot = save_plot)
UMAP_Dimensionality_Reduction_Visualisation(data_dir = test_dir,
                                            model = model, 
                                            type_info = base::paste(model_name, model_type, "test", sep = "_"),
                                            save_plot = save_plot)

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki