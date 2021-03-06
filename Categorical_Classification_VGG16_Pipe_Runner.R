# ------------------------------------------------------------------------------
# VGG16 CATEGORICAL MODEL IMPLEMENTATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/cifar-10/overview
# browseURL(url = "https://www.kaggle.com/c/cifar-10/overview")

# ------------------------------------------------------------------------------
# Train categorical VGG16 model:
# 1. Set current working directory with apropriate scripts for VGG16 categorical model training:
setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")

# 2. Train VGG16 model with frozen weights:
source("Categorical_Classification_VGG16_1st_Stage.R")
# 3. Remove all variables from Global Environment and clear session:
setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
rm(list = ls())
keras::k_clear_session()
# 4. Train VGG16 model with unfrozen weights and conduct model validation and testing:
source("Categorical_Classification_VGG16_2nd_Stage.R")

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki


