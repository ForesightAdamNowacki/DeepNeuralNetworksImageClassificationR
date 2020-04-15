# ------------------------------------------------------------------------------
# USEFUL FUNCTIONS
# ------------------------------------------------------------------------------
# Show and predict image class:
Predict_Image <- function(image_path, model, plot_image = TRUE){
  
  image_size <- base::dim(model$input)[[2]]
  
  image <- keras::image_load(path = image_path, target_size = base::c(image_size, image_size))
  image_array <- keras::image_to_array(img = image)
  image_array <- keras::array_reshape(x = image_array, dim = base::c(1, image_size, image_size, 3))
  
  datagen <- keras::image_data_generator(rescale = 1/255)
  generator <- keras::flow_images_from_data(x = image_array, generator = datagen, batch_size = 1)
  prediction <- model %>% keras::predict_generator(generator = generator, steps = 1) -> prediction
  
  if (plot_image == TRUE){
    batch <- keras::generator_next(generator = generator)
    graphics::plot(grDevices::as.raster(x = batch[1,,,]))
  }
  
  base::return(base::list(image_path = base::normalizePath(image_path),
                          prediction = prediction))
}
# ------------------------------------------------------------------------------
# Automaticaly organize correct and incorrect classification:
Organize_Correct_Incorrect_Classifications <- function(dataset_dir,
                                                       actual_classes,
                                                       prediction,
                                                       cwd = models_store,
                                                       cutoff = 0.5){
  
  base::print(base::paste("Current working directory:", cwd))
  base::setwd(cwd)
  
  categories <- base::list.files(path = dataset_dir)
  dataset_label <- base::basename(dataset_dir)
  
  category_1 <- base::paste(base::paste(dataset_dir, categories[1], sep = "/"), base::list.files(base::paste(dataset_dir, categories[1], sep = "/")), sep = "/")
  category_2 <- base::paste(base::paste(dataset_dir, categories[2], sep = "/"), base::list.files(base::paste(dataset_dir, categories[2], sep = "/")), sep = "/")
  
  summary_data <- tibble::tibble(files = base::c(category_1, category_2),
                                 actual_class = actual_classes,
                                 prediction = prediction,
                                 cutoff = cutoff) %>%
    dplyr::mutate(predicted_class = base::ifelse(prediction < cutoff, 0, 1))
  
  summary_data_correct <- summary_data %>%
    dplyr::filter(actual_class == predicted_class) 
  
  summary_data_incorrect <- summary_data %>%
    dplyr::filter(actual_class != predicted_class) 
  
  datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  readr::write_csv(summary_data, base::paste(datetime, dataset_label, "all_classification.csv", sep = "_"))
  readr::write_csv(summary_data_correct, base::paste(datetime, dataset_label, "correct_classification.csv", sep = "_"))
  readr::write_csv(summary_data_incorrect, base::paste(datetime, dataset_label, "incorrect_classification.csv", sep = "_"))
  base::print(base::paste("File created:", base::paste(datetime, dataset_label, "all_classification.csv", sep = "_")))
  base::print(base::paste("File created:", base::paste(datetime, dataset_label, "correct_classification.csv", sep = "_")))
  base::print(base::paste("File created:", base::paste(datetime, dataset_label, "incorrect_classification.csv", sep = "_")))  
  
  # correct:
  correct_classification_folder <- base::paste(dataset_label, "correct_classification", sep = "_")
  base::unlink(correct_classification_folder, recursive = TRUE)
  base::dir.create(correct_classification_folder, recursive  = TRUE, showWarnings = FALSE)
  base::print(base::paste("Folder created:", correct_classification_folder))
  
  correct_classification_dir <- base::paste(base::getwd(), correct_classification_folder, sep = "/")
  base::file.copy(from = summary_data_correct$files,
                  to = base::paste(correct_classification_dir, base::basename(summary_data_correct$files), sep = "/"))
  # incorrect:
  incorrect_classification_folder <- base::paste(dataset_label, "incorrect_classification", sep = "_")
  base::unlink(incorrect_classification_folder, recursive = TRUE)
  base::dir.create(incorrect_classification_folder, recursive  = TRUE, showWarnings = FALSE)
  base::print(base::paste("Folder created:", incorrect_classification_folder))
  
  incorrect_classification_dir <- base::paste(base::getwd(), incorrect_classification_folder, sep = "/")
  base::file.copy(from = summary_data_incorrect$files,
                  to = base::paste(incorrect_classification_dir, base::basename(summary_data_incorrect$files), sep = "/"))
  
  base::invisible(base::list(all_files = summary_data,
                             correct_classification = summary_data_correct,
                             incorrect_classification = summary_data_incorrect))
}
