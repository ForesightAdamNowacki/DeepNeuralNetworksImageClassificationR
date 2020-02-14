# Image augmentation:
# The script used to test data augmentation settings.

base::library(tensorflow)
base::library(keras)
base::library(tidyverse)

# Augmentation function:
image_augmentation <- function(image_path,
                               image_size = c(224, 224),
                               featurewise_center_parameter = FALSE,
                               samplewise_center_parameter = FALSE,
                               featurewise_std_normalization_parameter = FALSE,
                               samplewise_std_normalization_parameter = FALSE,
                               zca_epsilon_parameter = 1e-06,
                               zca_whitening_parameter = FALSE,
                               rotation_range_parameter = 15,
                               width_shift_range_parameter = 0.2,
                               height_shift_range_parameter = 0.2,
                               brightness_range_parameter = c(0.5, 1.5),
                               shear_range_parameter = 0.2,
                               zoom_range_parameter = 0.2,
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
  if (!require(tensorflow)){install.packages('tensorflow'); require('tensorflow')}
  if (!require(keras)){install.packages('keras'); require('keras')}
  if (!require(tidyverse)){install.packages('tidyverse'); require('tidyverse')}
  
  # Load image:
  image <- keras::image_load(image_path, target_size = c(image_size[1], image_size[2]))
  image_array <- keras::image_to_array(image)
  image_array <- keras::array_reshape(image_array, c(1, image_size[1], image_size[2], 3))
  
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
    file_name = stringr::str_sub(basename(image_path), 1, stringr::str_locate(basename(image_path), "\\.")[1, 1] - 1)
    plot_name = base::paste0(stringr::str_replace_all(base::Sys.time(), ":", "-"), "_image_augmentation_", file_name, ".jpg")
    grDevices::jpeg(plot_name, quality = quality_parameter, width = width_parameter, height = height_parameter)
    graphics::par(mfrow = base::c(base::ceiling(base::sqrt(n)), base::ceiling(base::sqrt(n))), mar = base::rep(margin, 4))
    for (i in 1:n){
      batch <- keras::generator_next(generator = augmentation_generator)
      graphics::plot(grDevices::as.raster(batch[1,,,]))}
    grDevices::dev.off()
    base::print(paste("Plot saved:", plot_name))
    } else {
    graphics::par(mfrow = base::c(base::ceiling(base::sqrt(n)), base::ceiling(base::sqrt(n))), mar = base::rep(margin, 4))
    for (i in 1:n){
      batch <- keras::generator_next(generator = augmentation_generator)
      graphics::plot(grDevices::as.raster(batch[1,,,]))
    }
  }
}    
  
setwd("C:\\Users\\adam.nowacki\\Desktop\\My_Files")
image_augmentation(image_path = "C:\\Users\\adam.nowacki\\Desktop\\ITMAGINATIONS\\Irena_Eris\\Data\\Adam.jpg", n = 30, plot_save = FALSE)
image_augmentation(image_path = "C:\\Users\\adam.nowacki\\Desktop\\ITMAGINATIONS\\Irena_Eris\\Data\\68590.jpg", n = 64, plot_save = TRUE)


