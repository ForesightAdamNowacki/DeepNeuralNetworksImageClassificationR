# ------------------------------------------------------------------------------
# ACTIVATION POOLING LAYERS VISUALISATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
# browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "Visualisation_Pooling_Activations"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
# 2. Create 'VGG16' folder in cwd
dir.create(path = paste(getwd(), model_name, sep = "/"))

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
library(tensorflow)
library(keras)
library(tidyverse)
library(deepviz)
source("D:/GitHub/DeepNeuralNetworksImageClassificationR/Useful_Functions.R")

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
                                  input_shape = c(image_size, image_size, channels))

# ------------------------------------------------------------------------------
# Visualize model:
model %>% deepviz::plot_model()
model %>% summary()

# ------------------------------------------------------------------------------
# Function Implementation for Visualisation Pooling 2D Layers Activation:
Pooling_2D_Layers_Activations <- function(model,
                                          image_path,
                                          layer_id,
                                          save_plot = FALSE){
  
  img <- keras::image_load(image_path, target_size = c(model$input_shape[[2]], model$input_shape[[2]]))
  img_tensor <- keras::image_to_array(img)
  img_tensor <- keras::array_reshape(img_tensor, c(1, model$input_shape[[2]], model$input_shape[[2]], 3))
  img_tensor <- img_tensor/255
  
  layer_outputs <- lapply(model$layers[grepl('pooling', sapply(model$layers, paste))], function(layer) layer$output)
  activation_model <- keras::keras_model(inputs = model$input, outputs = layer_outputs)
  
  activations <- stats::predict(activation_model, img_tensor)
  layer_activation <- activations[[layer_id]]
  
  plot_channel <- function(channel){
    rotate <- function(x) t(apply(x, 2, rev))
    graphics::image(rotate(channel), axes = FALSE, asp = 1, col = grDevices::gray.colors(n = 12))}
  
  if(save_plot == TRUE){
    file_name <- paste(paste(stringr::str_replace_all(Sys.time(), ":", "-"), "Pooling_2D_Layer_Visualisation", layer_id, sep = "_"), "png", sep = ".")
    file_path <- paste(getwd(), model_name, file_name, sep = "/")
    grDevices::png(file_path, width = 1000, height = 1000)
    graphics::par(mfrow = c(ceiling(sqrt(dim(layer_activation)[4])),
                                  ceiling(sqrt(dim(layer_activation)[4]))))
    graphics::par(mar = rep(0.1, 4))
    pb = utils::txtProgressBar(min = 0, max = dim(layer_activation)[4], initial = 0, style = 3)
    for (i in 1:dim(layer_activation)[4]){
      plot_channel(layer_activation[1,,,i])
      utils::setTxtProgressBar(pb,i)}
    grDevices::dev.off()
    cat("\n", "Plot saved:", file_path, "\n")
  } else {
    graphics::par(mfrow = c(ceiling(sqrt(dim(layer_activation)[4])), 
                                  ceiling(sqrt(dim(layer_activation)[4]))))
    graphics::par(mar = rep(0.1, 4))
    pb = utils::txtProgressBar(min = 0, max = dim(layer_activation)[4], initial = 0, style = 3)
    for (i in 1:dim(layer_activation)[4]){
      plot_channel(layer_activation[1,,,i])
      utils::setTxtProgressBar(pb,i)}
  }
}

# ------------------------------------------------------------------------------
# Plot Convolutional Layer Activation for choosen layer:
Pooling_Layers <- c()
for (i in 1:length(model$layers)){
  Layer <- model$layers[[i]]$name[grepl("pool", model$layers[[i]]$name)]
  Pooling_Layers <- c(Pooling_Layers, Layer)}
for (i in 1:length(Pooling_Layers)){cat(i, "Pooling 2D Layer:", Pooling_Layers[i], "\n")}

Pooling_2D_Layers_Activations(model = model,
                              image_path = paste(getwd(), "Data", "Tank_Rosomak.png", sep = "/"),
                              layer_id = 1,
                              save_plot = FALSE)

# ------------------------------------------------------------------------------
# Plot Convolutional Layer Activations for all layers:
for (i in 1:length(Pooling_Layers)){
  Pooling_2D_Layers_Activations(model = model,
                                image_path = paste(getwd(), "Data", "Tank_Rosomak.png", sep = "/"),
                                layer_id = i,
                                save_plot = TRUE)}

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki
