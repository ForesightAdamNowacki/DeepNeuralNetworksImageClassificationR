# ------------------------------------------------------------------------------
# GRAD-CAM HEATMAPS VISUALISATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
# browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "Visualisation_Heatmaps"

# ------------------------------------------------------------------------------
# Intro:
# 1. Set currect working directory:
setwd("D:/GitHub/DeepNeuralNetworksImageClassificationR")
# 2. Create 'VGG16' folder in cwd
if (dir.exists(paste(getwd(), model_name, sep = "/")) == FALSE){dir.create(path = paste(getwd(), model_name, sep = "/"))}

# ------------------------------------------------------------------------------
# Environment:
reticulate::use_condaenv("GPU_ML_1", required = TRUE)
library(tensorflow)
library(keras)
library(tidyverse)
library(deepviz)
library(magick) 
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
# Heatmap visualization:
Display_Heatmap <- function(model, image_path){
  
  Image_Preprocessing <- function(model, image_path){
    image <- keras::image_load(path = image_path, target_size = c(model$input_shape[[2]], model$input_shape[[3]])) %>%
      keras::image_to_array() %>%
      keras::array_reshape(dim = c(1, model$input_shape[[2]], model$input_shape[[2]], 3)) %>%
      keras::imagenet_preprocess_input()
    return(image)}
  
  image <- Image_Preprocessing(model = model, image_path = image_path)
  
  prediction <- model %>% stats::predict(image)
  model_output <- model$output[,which.max(prediction)]
  
  Convolutional_Layers <- c()
  for (i in 1:length(model$layers)){
    Layer <- model$layers[[i]]$name[grepl("conv", model$layers[[i]]$name)]
    Convolutional_Layers <- c(Convolutional_Layers, Layer)}
  
  last_convolutional_layer <- model %>% keras::get_layer(name = Convolutional_Layers[length(Convolutional_Layers)])
  
  gradients <- keras::k_gradients(model_output, last_convolutional_layer$output)[[1]]
  pooled_gradients <- keras::k_mean(gradients, axis = c(1, 2, 3))
  
  iterate <- keras::k_function(inputs = list(model$input),
                               outputs = list(pooled_gradients, last_convolutional_layer$output[1,,,]))
  
  c(pooled_gradients_value, convolutional_layer_output_value) %<-% iterate(list(image))
  
  for (i in 1:pooled_gradients$shape[[1]]){convolutional_layer_output_value[,,i] <- convolutional_layer_output_value[,,i] * pooled_gradients_value[[i]]}
  
  heatmap <- apply(convolutional_layer_output_value, c(1, 2), mean)
  heatmap <- pmax(heatmap, 0)
  heatmap <- heatmap/max(heatmap)
  
  base_name_1 <- stringr::str_sub(basename(image_path), start = 1, end = stringr::str_locate_all(basename(image_path), "\\.")[[1]][1] - 1)
  file_name_1 <- paste(paste("Heatmap_Visualisation", base_name_1, "1", sep = "_"), "png", sep = ".")
  file_path_1 <- paste(getwd(), model_name, file_name_1, sep = "/")
  
  save_heatmap <- function(heatmap, file_path, width, height, bg = "white", col = rev(grDevices::heat.colors(12))){
    
    grDevices::png(file_path, width = width, height = height, bg = bg)
    graphics::par(mar = c(0, 0, 0, 0))
    
    plot_channel <- function(heatmap){
      rotate <- function(x) t(apply(x, 2, rev))
      graphics::image(rotate(heatmap), axes = FALSE, asp = 1, col = col)}
    
    plot_channel(heatmap)
    grDevices::dev.off()
    cat("\n", "Plot saved:", file_path_1, "\n")}

  save_heatmap(heatmap = heatmap, file_path = file_path_1, width = model$input_shape[[2]], height = model$input_shape[[3]])
  
  original_image <- magick::image_read(image_path)
  info <- image_info(original_image)
  geometry <- paste0(info$width, "x", info$height, "!")
  
  magick::image_read(file_path_1) %>%
    magick::image_resize(geometry = geometry, filter = "quadratic") %>%
    magick::image_composite(image = original_image, operator = "blend", compose_args = "50") %>%
    graphics::plot()
  
  base_name_2 <- stringr::str_sub(basename(image_path), start = 1, end = stringr::str_locate_all(basename(image_path), "\\.")[[1]][1] - 1)
  file_name_2 <- paste(paste("Heatmap_Visualisation", base_name_2, "2", sep = "_"), "png", sep = ".")
  file_path_2 <- paste(getwd(), model_name, file_name_2, sep = "/")

  magick::image_read(file_path_1) %>%
    magick::image_resize(geometry = geometry, filter = "quadratic") %>%
    magick::image_composite(image = original_image, operator = "blend", compose_args = "50") %>%
    magick::image_write(file_path_2)
  cat("\n", "Plot saved:", file_path_2, "\n")
}

# ------------------------------------------------------------------------------
# Display and save heatmaps for uploaded images:
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/Elephant.png")
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/Tank_Rosomak.png")
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/German_Shepherd.png")
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/Taxi.png")

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki