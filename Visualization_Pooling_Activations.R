# ------------------------------------------------------------------------------
# ACTIVATION POOLING LAYERS VISUALISATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "Visualisation_Pooling_Activations"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
base::setwd("D:/GitHub/DeepNeuralNetworksRepoR")
# 2. Create 'VGG16' folder in cwd
base::dir.create(path = base::paste(base::getwd(), model_name, sep = "/"))

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
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
# Function Implementation for Visualisation Pooling 2D Layers Activation:
Pooling_2D_Layers_Activations <- function(model,
                                          image_path,
                                          layer_id,
                                          save_plot = FALSE){
  
  img <- keras::image_load(image_path, target_size = base::c(model$input_shape[[2]], model$input_shape[[2]]))
  img_tensor <- keras::image_to_array(img)
  img_tensor <- keras::array_reshape(img_tensor, c(1, model$input_shape[[2]], model$input_shape[[2]], 3))
  img_tensor <- img_tensor/255
  
  layer_outputs <- base::lapply(model$layers[base::grepl('pooling', base::sapply(model$layers, base::paste))], function(layer) layer$output)
  activation_model <- keras::keras_model(inputs = model$input, outputs = layer_outputs)
  
  activations <- stats::predict(activation_model, img_tensor)
  layer_activation <- activations[[layer_id]]
  
  plot_channel <- function(channel){
    rotate <- function(x) base::t(base::apply(x, 2, rev))
    graphics::image(rotate(channel), axes = FALSE, asp = 1, col = grDevices::gray.colors(n = 12))}
  
  if(save_plot == TRUE){
    file_name <- base::paste(base::paste(stringr::str_replace_all(base::Sys.time(), ":", "-"), "Pooling_2D_Layer_Visualisation", layer_id, sep = "_"), "png", sep = ".")
    file_path <- base::paste(base::getwd(), model_name, file_name, sep = "/")
    grDevices::png(file_path, width = 1000, height = 1000)
    graphics::par(mfrow = base::c(base::ceiling(base::sqrt(base::dim(layer_activation)[4])),
                                  base::ceiling(base::sqrt(base::dim(layer_activation)[4]))))
    graphics::par(mar = base::rep(0.1, 4))
    pb = utils::txtProgressBar(min = 0, max = base::dim(layer_activation)[4], initial = 0, style = 3)
    for (i in 1:base::dim(layer_activation)[4]){
      plot_channel(layer_activation[1,,,i])
      utils::setTxtProgressBar(pb,i)}
    grDevices::dev.off()
    base::cat("\n", "Plot saved:", file_path, "\n")
  } else {
    graphics::par(mfrow = base::c(base::ceiling(base::sqrt(base::dim(layer_activation)[4])), 
                                  base::ceiling(base::sqrt(base::dim(layer_activation)[4]))))
    graphics::par(mar = base::rep(0.1, 4))
    pb = utils::txtProgressBar(min = 0, max = base::dim(layer_activation)[4], initial = 0, style = 3)
    for (i in 1:base::dim(layer_activation)[4]){
      plot_channel(layer_activation[1,,,i])
      utils::setTxtProgressBar(pb,i)}
  }
}

# ------------------------------------------------------------------------------
# Plot Convolutional Layer Activation for choosen layer:
Pooling_Layers <- model$layers[base::grepl('pooling', base::sapply(model$layers, base::paste))] %>%
  base::sapply(., base::paste)
for (i in 1:base::length(Pooling_Layers)){base::cat(i, "Pooling 2D Layer:", Pooling_Layers[i], "\n")}

Pooling_2D_Layers_Activations(model = model,
                              image_path = base::paste(base::getwd(), "Data", "Tank_Rosomak.png", sep = "/"),
                              layer_id = 1,
                              save_plot = FALSE)

# ------------------------------------------------------------------------------
# Plot Convolutional Layer Activations for all layers:
for (i in 1:base::length(Pooling_Layers)){
  Pooling_2D_Layers_Activations(model = model,
                                image_path = base::paste(base::getwd(), "Data", "Tank_Rosomak.png", sep = "/"),
                                layer_id = i,
                                save_plot = TRUE)}

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki
