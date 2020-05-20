# ------------------------------------------------------------------------------
# IMAGE DATA AUGMENTATION
# The script used to test data augmentation settings.

# ------------------------------------------------------------------------------
# Environment:
base::library(reticulate)
reticulate::use_condaenv(condaenv = "GPU_ML_2", required = TRUE)
base::library(tensorflow)
base::library(keras)
base::library(tidyverse)
# ------------------------------------------------------------------------------
# Augmentation function:
Image_Augmentation <- function(image_path,
                               image_size = c(224, 224),
                               featurewise_center_parameter = FALSE,
                               samplewise_center_parameter = FALSE,
                               featurewise_std_normalization_parameter = FALSE,
                               samplewise_std_normalization_parameter = FALSE,
                               zca_epsilon_parameter = 1e-06,
                               zca_whitening_parameter = FALSE,
                               rotation_range_parameter = 25,
                               width_shift_range_parameter = 0.1,
                               height_shift_range_parameter = 0.1,
                               brightness_range_parameter = c(0.5, 1.5),
                               shear_range_parameter = 0.1,
                               zoom_range_parameter = 0.1,
                               channel_shift_range_parameter = 0.0,
                               fill_mode_parameter = "nearest",
                               cval_parameter = 0.0,
                               horizontal_flip_parameter = TRUE,
                               vertical_flip_parameter = FALSE,
                               rescale_parameter = 1/255, 
                               n = 16,
                               margin = 0.1,
                               plot_save = FALSE,
                               quality_parameter = 100,
                               width_parameter = 1000,
                               height_parameter = 1000){
  
  # Packages:
  if (!base::require(tensorflow)){install.packages('tensorflow'); base::require('tensorflow')}
  if (!base::require(keras)){install.packages('keras'); base::require('keras')}
  if (!base::require(tidyverse)){install.packages('tidyverse'); base::require('tidyverse')}
  
  # Load image:
  image <- keras::image_load(path = image_path, target_size = base::c(image_size[1], image_size[2]))
  image_array <- keras::image_to_array(img = image)
  image_array <- keras::array_reshape(x = image_array, dim = base::c(1, image_size[1], image_size[2], 3))
  
  # Create datagenerator:
  datagen <- keras::image_data_generator(
    featurewise_center = featurewise_center_parameter,
    samplewise_center = samplewise_center_parameter,
    featurewise_std_normalization = featurewise_std_normalization_parameter,
    samplewise_std_normalization = samplewise_std_normalization_parameter,
    zca_epsilon = zca_epsilon_parameter,
    zca_whitening = zca_whitening_parameter,
    rotation_range = rotation_range_parameter,
    width_shift_range = width_shift_range_parameter,
    height_shift_range = height_shift_range_parameter,
    brightness_range = brightness_range_parameter,
    shear_range = shear_range_parameter,
    zoom_range = zoom_range_parameter,
    channel_shift_range = channel_shift_range_parameter,
    fill_mode = fill_mode_parameter,
    cval = cval_parameter,
    horizontal_flip = horizontal_flip_parameter,
    vertical_flip = vertical_flip_parameter,
    rescale = rescale_parameter)
  
  augmentation_generator <- keras::flow_images_from_data(x = image_array, generator = datagen, batch_size = 1)
  
  # Plot parameters:
  if(plot_save == TRUE){
    file_name = stringr::str_sub(string = base::basename(path = image_path),
                                 start = 1,
                                 end = stringr::str_locate(string = base::basename(path = image_path), pattern = "\\.")[1, 1] - 1)
    plot_name = base::paste0(stringr::str_replace_all(string = base::Sys.time(), pattern = ":", replacement = "-"),
                             "_image_augmentation_", file_name, ".jpg")
    grDevices::jpeg(filename = plot_name,
                    quality = quality_parameter,
                    width = width_parameter,
                    height = height_parameter)
    graphics::par(mfrow = base::c(base::ceiling(base::sqrt(n)), base::ceiling(base::sqrt(n))),
                  mar = base::rep(margin, 4))
    for (i in 1:n){
      batch <- keras::generator_next(generator = augmentation_generator)
      graphics::plot(grDevices::as.raster(x = batch[1,,,]))}
    grDevices::dev.off()
    base::print(paste0("Plot saved: ", plot_name))
    } else {
    graphics::par(mfrow = base::c(base::ceiling(base::sqrt(n)), base::ceiling(base::sqrt(n))),
                  mar = base::rep(margin, 4))
    for (i in 1:n){
      batch <- keras::generator_next(generator = augmentation_generator)
      graphics::plot(grDevices::as.raster(x = batch[1,,,]))
    }
  }
}    
# ------------------------------------------------------------------------------
# Test image augmentation function:
base::setwd(dir = "D:\\GitHub\\DeepNeuralNetworksRepoR\\Images")
Image_Augmentation(image_path = "Dog_1.png", n = 9, plot_save = FALSE)
# ------------------------------------------------------------------------------