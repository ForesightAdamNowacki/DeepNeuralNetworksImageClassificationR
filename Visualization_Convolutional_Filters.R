# ------------------------------------------------------------------------------
# CONVOLUTIONAL FILTERS VISUALISATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "Visualisation_Convolutional_Filters"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'VGG16' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_1", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
base::library(deepviz)
base::source("D:/GitHub/DeepNeuralNetworksRepoR/Useful_Functions.R")

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
include_top <- TRUE

# ------------------------------------------------------------------------------
# VGG16 model architecture:
model <- keras::application_vgg16(include_top = include_top,
                                  weights = weights,
                                  input_shape = base::c(image_size, image_size, channels))

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% base::summary()

# ------------------------------------------------------------------------------
# Display selected filter in indicated convolutional layer:
Convolutional_Layers <- base::c()
for (i in 1:base::length(model$layers)){
  Layer <- model$layers[[i]]$name[base::grepl("conv", model$layers[[i]]$name)]
  Convolutional_Layers <- base::c(Convolutional_Layers, Layer)}
for (i in 1:base::length(Convolutional_Layers)){base::cat(i, "Convolutional 2D Layer:", Convolutional_Layers[i], "\n")}

Convolutional_Filter_Visualisation <- function(model, convolutional_layer_name, filter_index, step = 1, iterations = 50, constant = 0.000001){
  
  output_layer <- model$get_layer(convolutional_layer_name)$output
  loss <- keras::k_mean(output_layer[,,,filter_index])
  
  gradients <- keras::k_gradients(loss, model$input)[[1]]
  gradients <- gradients/(keras::k_sqrt(keras::k_mean(keras::k_square(gradients))) + constant)
  
  iterate <- keras::k_function(inputs = base::list(model$input),
                               outputs = base::list(loss, gradients))
  
  input_image_data <- base::array(stats::runif(model$input_shape[[2]] * model$input_shape[[3]] * model$input_shape[[4]]),
                                  dim = base::c(1, model$input_shape[[2]], model$input_shape[[3]], model$input_shape[[4]])) # * 20 + 128

  for (i in 1:iterations){
    c(loss_value, gradients_value) %<-% iterate(base::list(input_image_data))
    input_image_data <- input_image_data + (gradients_value * step)}
  
  image <- input_image_data[1,,,]
  Normalize_Values <- function(x){(x-min(x))/(max(x)-min(x))}
  for (j in 1:base::dim(image)[3]){image[,,j] <- Normalize_Values(image[,,j])}
  base::return(image)}

graphics::par(mfrow = base::c(1, 1))
grid::grid.raster(Convolutional_Filter_Visualisation(model = model,
                                                     convolutional_layer_name = "block1_conv1",
                                                     filter_index = 15))

# ------------------------------------------------------------------------------
# Display all filters in indicated convolutional layer:
Display_All_Convolutional_Filters <- function(model, convolutional_layer_name, step = 1, iterations = 50, constant = 0.000001, save_plot = FALSE){
  
  filters <- base::dim(model$get_layer(convolutional_layer_name)$output)[[4]]

  if(save_plot == TRUE){
    file_name <- base::paste(base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), model_name, convolutional_layer_name, sep = "_"), "png", sep = ".")
    file_path <- base::paste(base::getwd(), model_name, file_name, sep = "/")
    
    grDevices::png(file_path, width = 1000, height = 1000)
    grobs <- base::list()
    pb = utils::txtProgressBar(min = 0, max = filters, initial = 0, style = 3)
    for (k in 1:filters){
      Filter_Matrix <- Convolutional_Filter_Visualisation(model = model, 
                                                          convolutional_layer_name = convolutional_layer_name,
                                                          filter_index = k, 
                                                          step = step,
                                                          iterations = iterations,
                                                          constant = constant)
      grobs[[k]] <- grid::rasterGrob(image = Filter_Matrix)
      utils::setTxtProgressBar(pb, k)}
  gridExtra::grid.arrange(grobs = grobs, ncol = base::ceiling(base::sqrt(filters)), nrow = base::ceiling(base::sqrt(filters)))
  grDevices::dev.off()
  base::cat("\n", "Plot saved:", file_path, "\n")
  } else {
    grobs <- base::list()
    pb = utils::txtProgressBar(min = 0, max = filters, initial = 0, style = 3)
    for (k in 1:filters){
      Filter_Matrix <- Convolutional_Filter_Visualisation(model = model, 
                                                          convolutional_layer_name = convolutional_layer_name,
                                                          filter_index = k, 
                                                          step = step,
                                                          iterations = iterations,
                                                          constant = constant)
      grobs[[k]] <- grid::rasterGrob(image = Filter_Matrix)
      utils::setTxtProgressBar(pb, k)}
    gridExtra::grid.arrange(grobs = grobs, ncol = base::ceiling(base::sqrt(filters)), nrow = base::ceiling(base::sqrt(filters)))
    }
}

Display_All_Convolutional_Filters(model = model, 
                                  convolutional_layer_name = "block1_conv1",
                                  iterations = 50, 
                                  save_plot = FALSE)

for (i in 1:base::length(Convolutional_Layers)){
  Display_All_Convolutional_Filters(model = model, 
                                    convolutional_layer_name = Convolutional_Layers[i],
                                    iterations = 40, 
                                    save_plot = TRUE)}

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki