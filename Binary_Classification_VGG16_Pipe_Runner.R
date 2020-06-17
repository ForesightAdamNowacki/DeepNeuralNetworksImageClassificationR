# ------------------------------------------------------------------------------
# VGG16 BINARY MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
# utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Train binary VGG16 model:
# 1. Set current working directory with apropriate scripts for VGG16 binary model training:
base::setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")

# 2. Train VGG model with frozen weights:
base::source("Binary_Classification_VGG16_1st_Stage.R")
# 3. Remove all variables from Global Environment and clear session:
base::setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
base::rm(list = base::ls())
keras::k_clear_session()
# 4. Train VGG model with unfrozen weights and conduct model validation and testing:
base::source("Binary_Classification_VGG16_2nd_Stage.R")

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki


