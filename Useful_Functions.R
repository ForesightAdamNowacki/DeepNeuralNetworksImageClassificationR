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

# ------------------------------------------------------------------------------
# Count files in train/validation/test directory per class:
count_files = function(path){
  dirs <- base::list.dirs(path = path)
  dirs <- dirs[2:base::length(dirs)]
  files <- base::integer(base::length(dirs))
  folder <- base::character(base::length(dirs))
  for (i in base::seq_along(dirs)){
    files[i] <- base::length(base::list.files(path = dirs[i]))
    folder[i] <- base::basename(path = dirs[i])}
  result <- base::data.frame(category = folder, category_obs = files) %>%
    dplyr::mutate(category_freq = category_obs/base::sum(category_obs))
  base::return(result)}

# ------------------------------------------------------------------------------
# Automaticaly split data folder with classes into train, validation and test 
# datasets with similar target variable distribution
Split_Data_Train_Validation_Test <- function(data_dir,
                                             target_dir = base::getwd(),
                                             proportions = base::c(3, 1, 1),
                                             train_folder_label = "data_train",
                                             validation_folder_label = "data_validation",
                                             test_folder_label = "data_label",
                                             info_folder_label = "data_info",
                                             seed = 42){
  
  base::setwd(target_dir)
  sys_time <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  class_dirs <- base::list.dirs(data_dir)[2:base::length(base::list.dirs(data_dir))]
  
  files <- base::list()
  for (i in base::seq_along(class_dirs)){
    files[[i]] <- base::list.files(class_dirs[i])}
  names(files) <- base::basename(class_dirs)
  
  base::set.seed(seed = seed)
  files <- base::do.call(cbind, files) %>%
    tibble::as_tibble() %>%
    tidyr::pivot_longer(cols = base::basename(class_dirs), names_to = "class", values_to = "file") %>%
    dplyr::mutate(original_file_path = base::paste(data_dir, class, file, sep = "/"),
                  fold = caret::createFolds(class, k = base::sum(proportions), list = FALSE))
  
  # Split data:
  train_files <- files %>% dplyr::filter(fold %in% 1:proportions[1])
  validation_files <- files %>% dplyr::filter(fold %in% (proportions[1] + 1):(proportions[1] + proportions[2]))
  test_files <- files %>% dplyr::filter(fold %in% (proportions[1] + proportions[2] + 1):base::sum(proportions))
  
  # Train:
  base::unlink(base::paste(base::getwd(), train_folder_label, sep = "/"), recursive = TRUE)
  base::dir.create(base::paste(base::getwd(), train_folder_label, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  base::setwd(base::paste(base::getwd(), train_folder_label, sep = "/"))
  train_files %>%
    dplyr::mutate(final_file_path = base::paste(base::getwd(), class, file, sep = "/"),
                  fold = NULL) %>%
    dplyr::select(file, class, original_file_path, final_file_path) %>%
    dplyr::arrange(class, file) -> train_files
  
  for (i in base::seq_along(base::basename(class_dirs))){
    base::dir.create(base::paste(base::getwd(), base::basename(class_dirs)[i], sep = "/"), showWarnings = FALSE, recursive = TRUE)
    files_from <- train_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(original_file_path) %>%
      dplyr::pull()
    files_to <- train_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(file) %>%
      dplyr::pull()
    files_to <- base::paste(base::getwd(), base::basename(class_dirs)[i], files_to, sep = "/")
    base::file.copy(from = files_from, to = files_to)}
  base::print(base::paste("Train data generated successfully -", base::getwd()))
  base::setwd("..")
  
  # Validation:
  base::unlink(base::paste(base::getwd(), validation_folder_label, sep = "/"), recursive = TRUE)
  base::dir.create(base::paste(base::getwd(), validation_folder_label, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  base::setwd(base::paste(base::getwd(), validation_folder_label, sep = "/"))
  validation_files %>%
    dplyr::mutate(final_file_path = base::paste(base::getwd(), class, file, sep = "/"),
                  fold = NULL) %>%
    dplyr::select(file, class, original_file_path, final_file_path) %>%
    dplyr::arrange(class, file) -> validation_files
  
  for (i in base::seq_along(base::basename(class_dirs))){
    base::dir.create(base::paste(base::getwd(), base::basename(class_dirs)[i], sep = "/"), showWarnings = FALSE, recursive = TRUE)
    files_from <- validation_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(original_file_path) %>%
      dplyr::pull()
    files_to <- validation_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(file) %>%
      dplyr::pull()
    files_to <- base::paste(base::getwd(), base::basename(class_dirs)[i], files_to, sep = "/")
    base::file.copy(from = files_from, to = files_to)}
  base::print(base::paste("Validation data generated successfully -", base::getwd()))
  base::setwd("..")
  
  # Test:
  base::unlink(base::paste(base::getwd(), test_folder_label, sep = "/"), recursive = TRUE)
  base::dir.create(base::paste(base::getwd(), test_folder_label, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  base::setwd(base::paste(base::getwd(), test_folder_label, sep = "/"))
  test_files %>%
    dplyr::mutate(final_file_path = base::paste(base::getwd(), class, file, sep = "/"),
                  fold = NULL) %>%
    dplyr::select(file, class, original_file_path, final_file_path) %>%
    dplyr::arrange(class, file) -> test_files
  
  for (i in base::seq_along(base::basename(class_dirs))){
    base::dir.create(base::paste(base::getwd(), base::basename(class_dirs)[i], sep = "/"), showWarnings = FALSE, recursive = TRUE)
    files_from <- test_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(original_file_path) %>%
      dplyr::pull()
    files_to <- test_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(file) %>%
      dplyr::pull()
    files_to <- base::paste(base::getwd(), base::basename(class_dirs)[i], files_to, sep = "/")
    base::file.copy(from = files_from, to = files_to)}
  base::print(base::paste("Test data generated successfully -", base::getwd()))
  base::setwd("..")
  
  # Info:
  base::unlink(base::paste(base::getwd(), info_folder_label, sep = "/"), recursive = TRUE)
  base::dir.create(base::paste(base::getwd(), info_folder_label, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  base::setwd(base::paste(base::getwd(), info_folder_label, sep = "/"))
  readr::write_csv(train_files, base::paste(sys_time, "train_split_info.csv"))
  readr::write_csv(validation_files, base::paste(sys_time, "validation_split_info.csv"))
  readr::write_csv(test_files, base::paste(sys_time, "test_split_info.csv"))
  base::print(base::paste("Info data generated successfully -", base::getwd()))
  base::setwd("..")
  
  base::list(Train = count_files(path = base::paste(base::getwd(), train_folder_label, sep = "/")),
             Validation = count_files(path = base::paste(base::getwd(), validation_folder_label, sep = "/")),
             Test = count_files(path = base::paste(base::getwd(), test_folder_label, sep = "/"))) %>%
    base::return(.)}

# ------------------------------------------------------------------------------
# Automaticaly split data folder with classes into train and validation 
# datasets with similar target variable distribution
Split_Data_Train_Validation <- function(data_dir,
                                        target_dir = base::getwd(),
                                        proportions = base::c(3, 1),
                                        train_folder_label = "data_train",
                                        validation_folder_label = "data_validation",
                                        info_folder_label = "data_info",
                                        seed = 42){
  
  base::setwd(target_dir)
  sys_time <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  class_dirs <- base::list.dirs(data_dir)[2:base::length(base::list.dirs(data_dir))]
  
  files <- base::list()
  for (i in base::seq_along(class_dirs)){
    files[[i]] <- base::list.files(class_dirs[i])}
  names(files) <- base::basename(class_dirs)
  
  base::set.seed(seed = seed)
  files <- base::do.call(cbind, files) %>%
    tibble::as_tibble() %>%
    tidyr::pivot_longer(cols = base::basename(class_dirs), names_to = "class", values_to = "file") %>%
    dplyr::mutate(original_file_path = base::paste(data_dir, class, file, sep = "/"),
                  fold = caret::createFolds(class, k = base::sum(proportions), list = FALSE))
  
  # Split data:
  train_files <- files %>% dplyr::filter(fold %in% 1:proportions[1])
  validation_files <- files %>% dplyr::filter(fold %in% (proportions[1] + 1):(base::sum(proportions)))
  
  # Train:
  base::unlink(base::paste(base::getwd(), train_folder_label, sep = "/"), recursive = TRUE)
  base::dir.create(base::paste(base::getwd(), train_folder_label, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  base::setwd(base::paste(base::getwd(), train_folder_label, sep = "/"))
  train_files %>%
    dplyr::mutate(final_file_path = base::paste(base::getwd(), class, file, sep = "/"),
                  fold = NULL) %>%
    dplyr::select(file, class, original_file_path, final_file_path) %>%
    dplyr::arrange(class, file) -> train_files
  
  for (i in base::seq_along(base::basename(class_dirs))){
    base::dir.create(base::paste(base::getwd(), base::basename(class_dirs)[i], sep = "/"), showWarnings = FALSE, recursive = TRUE)
    files_from <- train_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(original_file_path) %>%
      dplyr::pull()
    files_to <- train_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(file) %>%
      dplyr::pull()
    files_to <- base::paste(base::getwd(), base::basename(class_dirs)[i], files_to, sep = "/")
    base::file.copy(from = files_from, to = files_to)}
  base::print(base::paste("Train data generated successfully -", base::getwd()))
  base::setwd("..")
  
  # Validation:
  base::unlink(base::paste(base::getwd(), validation_folder_label, sep = "/"), recursive = TRUE)
  base::dir.create(base::paste(base::getwd(), validation_folder_label, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  base::setwd(base::paste(base::getwd(), validation_folder_label, sep = "/"))
  validation_files %>%
    dplyr::mutate(final_file_path = base::paste(base::getwd(), class, file, sep = "/"),
                  fold = NULL) %>%
    dplyr::select(file, class, original_file_path, final_file_path) %>%
    dplyr::arrange(class, file) -> validation_files
  
  for (i in base::seq_along(base::basename(class_dirs))){
    base::dir.create(base::paste(base::getwd(), base::basename(class_dirs)[i], sep = "/"), showWarnings = FALSE, recursive = TRUE)
    files_from <- validation_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(original_file_path) %>%
      dplyr::pull()
    files_to <- validation_files %>%
      dplyr::filter(class == base::basename(class_dirs)[i]) %>%
      dplyr::select(file) %>%
      dplyr::pull()
    files_to <- base::paste(base::getwd(), base::basename(class_dirs)[i], files_to, sep = "/")
    base::file.copy(from = files_from, to = files_to)}
  base::print(base::paste("Validation data generated successfully -", base::getwd()))
  base::setwd("..")
  
  # Info:
  base::unlink(base::paste(base::getwd(), info_folder_label, sep = "/"), recursive = TRUE)
  base::dir.create(base::paste(base::getwd(), info_folder_label, sep = "/"), showWarnings = FALSE, recursive = TRUE)
  base::setwd(base::paste(base::getwd(), info_folder_label, sep = "/"))
  readr::write_csv(train_files, base::paste(sys_time, "train_split_info.csv"))
  readr::write_csv(validation_files, base::paste(sys_time, "validation_split_info.csv"))
  base::print(base::paste("Info data generated successfully -", base::getwd()))
  base::setwd("..")
  
  base::list(Train = count_files(path = base::paste(base::getwd(), train_folder_label, sep = "/")),
             Validation = count_files(path = base::paste(base::getwd(), validation_folder_label, sep = "/"))) %>%
    base::return(.)
}
