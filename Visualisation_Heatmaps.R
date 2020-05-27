# ------------------------------------------------------------------------------
# GRAD-CAM HEATMAPS VISUALISATION
# ------------------------------------------------------------------------------
# Data:
# https://www.kaggle.com/c/dogs-vs-cats
utils::browseURL(url = "https://www.kaggle.com/c/dogs-vs-cats")

# ------------------------------------------------------------------------------
# Model name:
model_name <- "Visualisation_Heatmaps"

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
base::library(magick) 
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
# Heatmap visualization:
Display_Heatmap <- function(model, image_path){
  
  Image_Preprocessing <- function(model, image_path){
    image <- keras::image_load(path = image_path, target_size = base::c(model$input_shape[[2]], model$input_shape[[3]])) %>%
      keras::image_to_array() %>%
      keras::array_reshape(dim = base::c(1, model$input_shape[[2]], model$input_shape[[2]], 3)) %>%
      keras::imagenet_preprocess_input()
    base::return(image)}
  
  image <- Image_Preprocessing(model = model, image_path = image_path)
  
  prediction <- model %>% stats::predict(image)
  model_output <- model$output[,base::which.max(prediction)]
  
  Convolutional_Layers <- base::c()
  for (i in 1:base::length(model$layers)){
    Layer <- model$layers[[i]]$name[base::grepl("conv", model$layers[[i]]$name)]
    Convolutional_Layers <- base::c(Convolutional_Layers, Layer)}
  
  last_convolutional_layer <- model %>% keras::get_layer(name = Convolutional_Layers[base::length(Convolutional_Layers)])
  
  gradients <- keras::k_gradients(model_output, last_convolutional_layer$output)[[1]]
  pooled_gradients <- keras::k_mean(gradients, axis = base::c(1, 2, 3))
  
  iterate <- keras::k_function(inputs = base::list(model$input),
                               outputs = base::list(pooled_gradients, last_convolutional_layer$output[1,,,]))
  
  c(pooled_gradients_value, convolutional_layer_output_value) %<-% iterate(base::list(image))
  
  for (i in 1:pooled_gradients$shape[[1]]){convolutional_layer_output_value[,,i] <- convolutional_layer_output_value[,,i] * pooled_gradients_value[[i]]}
  
  heatmap <- base::apply(convolutional_layer_output_value, base::c(1, 2), base::mean)
  heatmap <- base::pmax(heatmap, 0)
  heatmap <- heatmap/base::max(heatmap)
  
  base_name_1 <- stringr::str_sub(base::basename(image_path), start = 1, end = stringr::str_locate_all(base::basename(image_path), "\\.")[[1]][1] - 1)
  file_name_1 <- base::paste(base::paste("Heatmap_Visualisation", base_name_1, "1", sep = "_"), "png", sep = ".")
  file_path_1 <- base::paste(base::getwd(), model_name, file_name_1, sep = "/")
  
  save_heatmap <- function(heatmap, file_path, width, height, bg = "white", col = base::rev(grDevices::heat.colors(12))){
    
    grDevices::png(file_path, width = width, height = height, bg = bg)
    graphics::par(mar = base::c(0, 0, 0, 0))
    
    plot_channel <- function(heatmap){
      rotate <- function(x) base::t(base::apply(x, 2, rev))
      graphics::image(rotate(heatmap), axes = FALSE, asp = 1, col = col)}
    
    plot_channel(heatmap)
    grDevices::dev.off()
    base::cat("\n", "Plot saved:", file_path_1, "\n")}

  save_heatmap(heatmap = heatmap, file_path = file_path_1, width = model$input_shape[[2]], height = model$input_shape[[3]])
  
  original_image <- magick::image_read(image_path)
  info <- image_info(original_image)
  geometry <- base::paste0(info$width, "x", info$height, "!")
  
  magick::image_read(file_path_1) %>%
    magick::image_resize(geometry = geometry, filter = "quadratic") %>%
    magick::image_composite(image = original_image, operator = "blend", compose_args = "50") %>%
    graphics::plot()
  
  base_name_2 <- stringr::str_sub(base::basename(image_path), start = 1, end = stringr::str_locate_all(base::basename(image_path), "\\.")[[1]][1] - 1)
  file_name_2 <- base::paste(base::paste("Heatmap_Visualisation", base_name_2, "2", sep = "_"), "png", sep = ".")
  file_path_2 <- base::paste(base::getwd(), model_name, file_name_2, sep = "/")

  magick::image_read(file_path_1) %>%
    magick::image_resize(geometry = geometry, filter = "quadratic") %>%
    magick::image_composite(image = original_image, operator = "blend", compose_args = "50") %>%
    magick::image_write(file_path_2)
  base::cat("\n", "Plot saved:", file_path_2, "\n")}

# ------------------------------------------------------------------------------
# Display and save heatmaps for uploaded images:
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/Elephant.png")
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/Tank_Rosomak.png")
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/German_Shepherd.png")
Display_Heatmap(model = model, image_path = "D:/GitHub/DeepNeuralNetworksRepoR/Data/Taxi.png")

# ------------------------------------------------------------------------------
# https://github.com/ForesightAdamNowacki