# ------------------------------------------------------------------------------
# BINARY MODEL EVALUATION EXAMPLES
# ------------------------------------------------------------------------------
# Environment:
base::source("C:\\Users\\admin\\Desktop\\GitHub\\DeepNeuralNetworks\\Binary_Model_Evaluation.R")
# ------------------------------------------------------------------------------
# Data: Johns Hopkins University Ionosphere database
# Description: Predict high-energy structures in the atmosphere from antenna data.
# Type: Classification
# Dimensions: 351 instances, 35 attributes
# Inputs: Numeric
# Output: Categorical, 2 class labels
# UCI Machine Learning Repository:
utils::browseURL(url = "https://archive.ics.uci.edu/ml/datasets/Ionosphere")
base::library(mlbench)
base::library(randomForest)
utils::data(Ionosphere)
data <- Ionosphere
data %>%
  tibble::as_tibble(.) %>%
  dplyr::mutate(V1 = NULL,
                V2 = NULL,
                Class = base::factor(base::ifelse(Class == "good", 1, 0))) -> data; data

model <- randomForest::randomForest(Class ~ ., data = data, ntree = 100)
predicted <- stats::predict(model, data, "prob")[,2]
actual <- base::as.numeric(data$Class) - 1

Binary_Classifier_Verification(actual = actual, predicted = predicted, cutoff = 0.25)
Binary_Classifier_Cutoff_Optimization(actual = actual, predicted = predicted, cuts = 50)
# ------------------------------------------------------------------------------