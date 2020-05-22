# ------------------------------------------------------------------------------
# USEFUL FUNCTIONS
# ------------------------------------------------------------------------------
# Show and predict image class:
Predict_Image <- function(image_path, model, classes, plot_image = TRUE){
  
  image_size <- base::dim(model$input)[[2]]
  
  image <- keras::image_load(path = image_path, target_size = base::c(image_size, image_size))
  image_array <- keras::image_to_array(img = image)
  image_array <- keras::array_reshape(x = image_array, dim = base::c(1, image_size, image_size, 3))
  
  datagen <- keras::image_data_generator(rescale = 1/255)
  generator <- keras::flow_images_from_data(x = image_array, generator = datagen, batch_size = 1)
  prediction <- model %>% keras::predict_generator(generator = generator, steps = 1)
  colnames(prediction) <- labels
  
  if (plot_image == TRUE){
    batch <- keras::generator_next(generator = generator)
    graphics::plot(grDevices::as.raster(x = batch[1,,,]))
  }
  
  base::return(base::list(image_path = base::normalizePath(image_path),
                          predictions = prediction,
                          predicted_class = labels[base::which.max(prediction)]))}

# ------------------------------------------------------------------------------
# Automaticaly organize correct and incorrect binary classifications:
Organize_Correct_Incorrect_Binary_Classifications <- function(dataset_dir,
                                                              actual_classes,
                                                              predicted,
                                                              type_info,
                                                              cwd = models_store_dir,
                                                              cutoff = 0.5,
                                                              save_summary_files = TRUE,
                                                              save_correct_images = TRUE,
                                                              save_incorrect_images = TRUE){
  
  base::cat("Current working directory:", cwd, "\n")
  base::setwd(cwd)
  
  datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  categories <- base::list.files(path = dataset_dir)
  dataset_label <- base::basename(dataset_dir)
  
  category_1 <- base::paste(base::paste(dataset_dir, categories[1], sep = "/"), base::list.files(base::paste(dataset_dir, categories[1], sep = "/")), sep = "/")
  category_2 <- base::paste(base::paste(dataset_dir, categories[2], sep = "/"), base::list.files(base::paste(dataset_dir, categories[2], sep = "/")), sep = "/")
  
  summary_data <- tibble::tibble(files = base::c(category_1, category_2),
                                 actual_class = actual_classes,
                                 predicted = predicted,
                                 cutoff = cutoff) %>%
    dplyr::mutate(predicted_class = base::ifelse(predicted < cutoff, 0, 1))
  
  summary_data_correct <- summary_data %>%
    dplyr::filter(actual_class == predicted_class) 
  
  summary_data_incorrect <- summary_data %>%
    dplyr::filter(actual_class != predicted_class) 
  
  if (base::isTRUE(save_summary_files)){
    readr::write_csv2(summary_data, base::paste(datetime, type_info, dataset_label, "all_classifications.csv", sep = "_"))
    readr::write_csv2(summary_data_correct, base::paste(datetime, type_info, dataset_label, "correct_classifications.csv", sep = "_"))
    readr::write_csv2(summary_data_incorrect, base::paste(datetime, type_info, dataset_label, "incorrect_classifications.csv", sep = "_"))
    base::cat("File created:", base::paste(datetime, type_info, dataset_label, "all_classifications.csv", sep = "_"), "\n")
    base::cat("File created:", base::paste(datetime, type_info, dataset_label, "correct_classifications.csv", sep = "_"), "\n")
    base::cat("File created:", base::paste(datetime, type_info, dataset_label, "incorrect_classifications.csv", sep = "_"), "\n")}
  
  # correct:
  if (base::isTRUE(save_correct_images)){
    correct_classification_folder <- base::paste(datetime, type_info, dataset_label, "correct_classifications", sep = "_")
    base::unlink(correct_classification_folder, recursive = TRUE)
    base::dir.create(correct_classification_folder, recursive  = TRUE, showWarnings = FALSE)
    base::cat("Folder created:", correct_classification_folder, "\n")
    
    correct_classification_dir <- base::paste(base::getwd(), correct_classification_folder, sep = "/")
    base::file.copy(from = summary_data_correct$files,
                    to = base::paste(correct_classification_dir, base::basename(summary_data_correct$files), sep = "/"))}
  
  # incorrect:
  if (base::isTRUE(save_incorrect_images)){
    incorrect_classification_folder <- base::paste(datetime, type_info, dataset_label, "incorrect_classifications", sep = "_")
    base::unlink(incorrect_classification_folder, recursive = TRUE)
    base::dir.create(incorrect_classification_folder, recursive  = TRUE, showWarnings = FALSE)
    base::cat("Folder created:", incorrect_classification_folder, "\n")
    
    incorrect_classification_dir <- base::paste(base::getwd(), incorrect_classification_folder, sep = "/")
    base::file.copy(from = summary_data_incorrect$files,
                    to = base::paste(incorrect_classification_dir, base::basename(summary_data_incorrect$files), sep = "/"))}
  
  if (base::isTRUE(save_summary_files)){
    base::invisible(base::list(all_files = summary_data,
                               correct_classification = summary_data_correct,
                               incorrect_classification = summary_data_incorrect))}}

# ------------------------------------------------------------------------------
# Automaticaly organize correct and incorrect catogorical classifications:
Organize_Correct_Incorrect_Categorical_Classifications <- function(dataset_dir,
                                                                   actual_classes,
                                                                   predicted,
                                                                   type_info,
                                                                   cwd = models_store,
                                                                   save_summary_files = TRUE,
                                                                   save_correct_images = TRUE,
                                                                   save_incorrect_images = TRUE){
  
  base::cat("Current working directory:", cwd, "\n")
  base::setwd(cwd)
  
  datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  categories <- base::list.files(path = dataset_dir)
  dataset_label <- base::basename(dataset_dir)
  
  lista <- base::list()
  for (i in 1:base::length(categories)){
    lista[[i]] <- tibble::tibble(files = base::paste(base::paste(dataset_dir, categories[i], sep = "/"), base::list.files(base::paste(dataset_dir, categories[i], sep = "/")), sep = "/"))
  }
  summary_data <- base::do.call("bind_rows", lista) %>%
    dplyr::mutate(actual_class = actual_classes,
                  predicted_class = base::max.col(predicted))
  
  summary_data_correct <- summary_data %>%
    dplyr::filter(actual_class == predicted_class) 
  
  summary_data_incorrect <- summary_data %>%
    dplyr::filter(actual_class != predicted_class) 
  
  if (base::isTRUE(save_summary_files)){
    readr::write_csv2(summary_data, base::paste(datetime, type_info, dataset_label, "all_classifications.csv", sep = "_"))
    readr::write_csv2(summary_data_correct, base::paste(datetime, type_info, dataset_label, "correct_classifications.csv", sep = "_"))
    readr::write_csv2(summary_data_incorrect, base::paste(datetime, type_info, dataset_label, "incorrect_classifications.csv", sep = "_"))
    base::cat(base::paste("File created:", base::paste(datetime, type_info, dataset_label, "all_classifications.csv", sep = "_")), "\n")
    base::cat(base::paste("File created:", base::paste(datetime, type_info, dataset_label, "correct_classifications.csv", sep = "_")), "\n")
    base::cat(base::paste("File created:", base::paste(datetime, type_info, dataset_label, "incorrect_classifications.csv", sep = "_")), "\n")}
  
  # correct:
  if (base::isTRUE(save_correct_images)){
    correct_classification_folder <- base::paste(datetime, type_info, dataset_label, "correct_classification", sep = "_")
    base::unlink(correct_classification_folder, recursive = TRUE)
    base::dir.create(correct_classification_folder, recursive  = TRUE, showWarnings = FALSE)
    base::cat("Folder created:", correct_classification_folder, "\n")
    
    correct_classification_dir <- base::paste(base::getwd(), correct_classification_folder, sep = "/")
    base::file.copy(from = summary_data_correct$files,
                    to = base::paste(correct_classification_dir, base::basename(summary_data_correct$files), sep = "/"))}
  
  # incorrect:
  if (base::isTRUE(save_incorrect_images)){
    incorrect_classification_folder <- base::paste(datetime, type_info, dataset_label, "incorrect_classification", sep = "_")
    base::unlink(incorrect_classification_folder, recursive = TRUE)
    base::dir.create(incorrect_classification_folder, recursive  = TRUE, showWarnings = FALSE)
    base::cat("Folder created:", incorrect_classification_folder, "\n")
    
    incorrect_classification_dir <- base::paste(base::getwd(), incorrect_classification_folder, sep = "/")
    base::file.copy(from = summary_data_incorrect$files,
                    to = base::paste(incorrect_classification_dir, base::basename(summary_data_incorrect$files), sep = "/"))}
  
  if (base::isTRUE(save_summary_files)){
    base::invisible(base::list(all_files = summary_data,
                               correct_classification = summary_data_correct,
                               incorrect_classification = summary_data_incorrect))}}

# ------------------------------------------------------------------------------
# Count files in train/validation/test directory per class:
Count_Files = function(path){
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
                                             train_folder_label = "train_dataset",
                                             validation_folder_label = "validation_dataset",
                                             test_folder_label = "test_dataset",
                                             info_folder_label = "info_dataset",
                                             seed = 42){
  base::setwd(target_dir)
  base::set.seed(seed = seed)
  sys_time <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  class_dirs <- base::list.dirs(data_dir)[2:base::length(base::list.dirs(data_dir))]
  
  files <- base::list()
  for (i in base::seq_along(class_dirs)){
    files[[i]] <- base::list.files(class_dirs[i])}
  names(files) <- base::basename(class_dirs)
  
  files <- tibble::tibble(file = base::do.call(c, files),
                          class = base::rep(names(base::sapply(files, length)), times = base::sapply(files, length))) %>%
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
  
  base::list(Counts = base::list(train = train_files %>%
                                   dplyr::mutate(dataset = train_folder_label),
                                 validation = validation_files %>%
                                   dplyr::mutate(dataset = validation_folder_label),
                                 test = test_files %>%
                                   dplyr::mutate(dataset = test_folder_label)) %>%
               dplyr::bind_rows(.) %>%
               dplyr::group_by(dataset, class) %>%
               dplyr::summarise(count = dplyr::n()) %>%
               tidyr::pivot_wider(id_col = "class",
                                  names_from = "dataset",
                                  values_from = "count") %>%
               dplyr::select(class, train_folder_label, validation_folder_label, test_folder_label),
             Proportions = base::list(train = train_files %>%
                                        dplyr::mutate(dataset = train_folder_label),
                                      validation = validation_files %>%
                                        dplyr::mutate(dataset = validation_folder_label),
                                      test = test_files %>%
                                        dplyr::mutate(dataset = test_folder_label)) %>%
               dplyr::bind_rows(.) %>%
               dplyr::group_by(dataset, class) %>%
               dplyr::summarise(count = dplyr::n()) %>%
               dplyr::group_by(dataset) %>%
               dplyr::mutate(proportion = count/base::sum(count)) %>%
               tidyr::pivot_wider(id_cols = "class",
                                  names_from = "dataset",
                                  values_from = "proportion") %>%
               dplyr::select(class, train_folder_label, validation_folder_label, test_folder_label)) %>%
    base::return(.)}

# ------------------------------------------------------------------------------
# Automaticaly split data folder with classes into train and validation
# datasets with similar target variable distribution
Split_Data_Train_Validation <- function(data_dir,
                                        target_dir = base::getwd(),
                                        proportions = base::c(3, 1),
                                        train_folder_label = "train_dataset",
                                        validation_folder_label = "validation_dataset",
                                        info_folder_label = "info_dataset",
                                        seed = 42){
  base::setwd(target_dir)
  base::set.seed(seed = seed)
  sys_time <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  class_dirs <- base::list.dirs(data_dir)[2:base::length(base::list.dirs(data_dir))]
  
  files <- base::list()
  for (i in base::seq_along(class_dirs)){
    files[[i]] <- base::list.files(class_dirs[i])}
  names(files) <- base::basename(class_dirs)
  
  files <- tibble::tibble(file = base::do.call(c, files),
                          class = base::rep(names(base::sapply(files, length)), times = base::sapply(files, length))) %>%
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
  
  base::list(Counts = base::list(train = train_files %>%
                                   dplyr::mutate(dataset = train_folder_label),
                                 validation = validation_files %>%
                                   dplyr::mutate(dataset = validation_folder_label)) %>%
               dplyr::bind_rows(.) %>%
               dplyr::group_by(dataset, class) %>%
               dplyr::summarise(count = dplyr::n()) %>%
               tidyr::pivot_wider(id_col = "class",
                                  names_from = "dataset",
                                  values_from = "count") %>%
               dplyr::select(class, train_folder_label, validation_folder_label),
             Proportions = base::list(train = train_files %>%
                                        dplyr::mutate(dataset = train_folder_label),
                                      validation = validation_files %>%
                                        dplyr::mutate(dataset = validation_folder_label)) %>%
               dplyr::bind_rows(.) %>%
               dplyr::group_by(dataset, class) %>%
               dplyr::summarise(count = dplyr::n()) %>%
               dplyr::group_by(dataset) %>%
               dplyr::mutate(proportion = count/base::sum(count)) %>%
               tidyr::pivot_wider(id_cols = "class",
                                  names_from = "dataset",
                                  values_from = "proportion") %>%
               dplyr::select(class, train_folder_label, validation_folder_label)) %>%
    base::return(.)}

# ------------------------------------------------------------------------------
# Optimize ensemble binary model:
Optimize_Binary_Ensemble_Cutoff_Model <- function(actual_class,
                                                  predictions,
                                                  cuts,
                                                  weights,
                                                  key_metric = ACC,
                                                  key_metric_as_string = FALSE,
                                                  ascending = FALSE,
                                                  summary_type = "mean",
                                                  seed = 42,
                                                  top = 10,
                                                  TN_cost = 0,
                                                  FP_cost = 1,
                                                  FN_cost = 1,
                                                  TP_cost = 0){
  
  # Libraries:
  if (!require(tidyverse)){utils::install.packages('tidyverse'); require('tidyverse')}  
  
  
  if (key_metric_as_string == FALSE){
    key_metric <- dplyr::enquo(key_metric) 
    key_metric_name <- dplyr::quo_name(key_metric)}
  
  if (key_metric_as_string == TRUE){
    key_metric <- rlang::sym(key_metric)
    key_metric <- dplyr::enquo(key_metric) 
    key_metric_name <- dplyr::quo_name(key_metric)}
  
  # Generate cuts:
  base::set.seed(seed = seed)
  cuts_ <- stats::runif(n = cuts, min = 0, max = 1)
  cuts_ <- base::sort(x = cuts_, decreasing = FALSE)
  
  # Generate waights:
  base::set.seed(seed = seed)
  weights_ <- base::matrix(data = stats::runif(base::ncol(predictions) * weights, min = 0, max = 1),
                           nrow = weights,
                           ncol = base::ncol(predictions))
  
  # Cutoff and weights optimization:
  results <- base::list()
  base::cat("\n", "Ensemble model optimization:", "\n")
  pb = txtProgressBar(min = 0, max = cuts, initial = 0, style = 3) 
  
  for (i in 1:cuts){
    cut_value <- cuts_[i]
    df <- tibble::tibble(CUTOFF = base::numeric(weights),
                         WEIGHTS = base::character(weights),
                         TN = base::numeric(weights),
                         FP = base::numeric(weights),
                         FN = base::numeric(weights),
                         TP = base::numeric(weights),
                         P = base::numeric(weights),
                         N = base::numeric(weights),
                         ACC = base::numeric(weights),
                         BACC = base::numeric(weights),
                         BIAS = base::numeric(weights),
                         CE = base::numeric(weights),
                         TPR = base::numeric(weights),
                         TNR = base::numeric(weights),
                         PPV = base::numeric(weights),
                         NPV = base::numeric(weights),
                         FNR = base::numeric(weights),
                         FPR = base::numeric(weights),
                         FDR = base::numeric(weights),
                         FOR = base::numeric(weights),
                         TS = base::numeric(weights),
                         F1 = base::numeric(weights),
                         BM = base::numeric(weights),
                         MK = base::numeric(weights),
                         COST = base::numeric(weights))
    
    for (j in 1:base::nrow(weights_)){
      weight_value <- weights_[j,]/base::sum(weights_[j,])
      predictions_table <- mapply("*", base::as.data.frame(predictions), weight_value) %>%
        tibble::as_tibble() %>%
        dplyr::mutate(prediction = base::rowSums(.),
                      cutoff = cut_value,
                      predicted_class = base::ifelse(prediction < cutoff, 0, 1))
      predicted_class <- base::factor(predictions_table$predicted_class, levels = base::c(0, 1), labels = base::c(0, 1))
      confusion_matrix <- base::table(actual_class, predicted_class)
      
      df$CUTOFF[j] <- cut_value
      df$WEIGHTS[j] <- base::paste(weight_value, collapse = ", ")
      df$TN[j] <- confusion_matrix[1, 1]
      df$FP[j] <- confusion_matrix[1, 2]
      df$FN[j] <- confusion_matrix[2, 1]
      df$TP[j] <- confusion_matrix[2, 2]
      df$N <- df$TN[j] + df$FP[j]
      df$P <- df$FN[j] + df$TP[j]
      df$ACC[j] <- (df$TN[j] + df$TP[j])/(df$TN[j] + df$FN[j] + df$FP[j] + df$TP[j])
      df$BACC[j] <- (df$TN[j]/(df$TN[j] + df$FP[j]) + df$TP[j]/(df$FN[j] + df$TP[j]))/2
      df$BIAS[j] <- base::mean(base::as.numeric(actual_class)) - base::mean(base::as.numeric(predicted_class))
      df$CE[j] <- (df$FN[j] + df$FP[j])/(df$TN[j] + df$FN[j] + df$FP[j] + df$TP[j])
      df$TPR[j] <- df$TP[j]/(df$TP[j] + df$FN[j])
      df$TNR[j] <- df$TN[j]/(df$TN[j] + df$FP[j])
      df$PPV[j] <- df$TP[j]/(df$TP[j] + df$FP[j])
      df$NPV[j] <- df$TN[j]/(df$TN[j] + df$FN[j])
      df$FNR[j] <- df$FN[j]/(df$FN[j] + df$TP[j])
      df$FPR[j] <- df$FP[j]/(df$FP[j] + df$TN[j])
      df$FDR[j] <- df$FP[j]/(df$FP[j] + df$TP[j])
      df$FOR[j] <- df$FN[j]/(df$FN[j] + df$TN[j])
      df$TS[j] <- df$TP[j]/(df$TP[j] + df$FN[j] + df$FP[j])
      df$F1[j] <- (2 * df$PPV[j] * df$TPR[j])/(df$PPV[j] + df$TPR[j])
      df$BM[j] <- df$TPR[j] + df$TNR[j] - 1
      df$MK[j] <- df$PPV[j] + df$NPV[j] - 1
      df$COST[j] <- TN_cost * df$TN[j] + FP_cost * df$FP[j] + FN_cost * df$FN[j] + TP_cost * df$TP[j]}
    results[[i]] <- df
    utils::setTxtProgressBar(pb,i)}
  
  base::cat("\n")
  
  # Convert list results do tibble data frame:
  final_results <- base::do.call(bind_rows, results) %>%
    tidyr::separate(col = WEIGHTS, sep = ", ", into = base::paste("model", 1:base::ncol(predictions), sep = "_"), convert = TRUE)
  
  # Arrange according to selected metric:
  if(ascending == TRUE){
    final_results %>%
      dplyr::arrange(!!key_metric) -> final_results
  } else {
    final_results %>%
      dplyr::arrange(dplyr::desc(!!key_metric)) -> final_results
  }
  
  
  # Return results:
  if (summary_type == "mean"){
    base::return(base::list(all_results = final_results,
                            top_results = final_results %>% utils::head(top),
                            optimized_cutoff = final_results %>%
                              utils::head(top) %>%
                              dplyr::select(CUTOFF) %>%
                              dplyr::summarise_all(base::mean),
                            optimized_weights = final_results %>%
                              utils::head(top) %>%
                              dplyr::select(dplyr::starts_with("model_")) %>%
                              dplyr::summarise_all(base::mean)))}
  
  if (summary_type == "median"){
    base::return(base::list(all_results = final_results,
                            top_results = final_results %>% utils::head(top),
                            optimized_cutoff = final_results %>%
                              utils::head(top) %>%
                              dplyr::select(CUTOFF) %>%
                              dplyr::summarise_all(stats::median),
                            optimized_weights = final_results %>%
                              utils::head(top) %>%
                              dplyr::select(dplyr::starts_with("model_")) %>%
                              dplyr::summarise_all(stats::median)))}
}

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to target classes:
Display_Target_Class_Predictions_Distribution <- function(actual,
                                                          predicted,
                                                          labels,
                                                          type_info,
                                                          bins = 10,
                                                          text_size = 7,
                                                          title_size = 9,
                                                          save_plot = FALSE,
                                                          plot_size = 20){
  
  datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  
  tibble::tibble(actual = base::factor(actual),
                 predicted = predicted) %>%
    dplyr::mutate(cut = ggplot2::cut_interval(predicted, length = 1/bins)) %>%
    dplyr::group_by(cut, actual) %>%
    dplyr::summarise(n = dplyr::n()) %>%
    dplyr::ungroup() %>%
    tidyr::complete(cut, actual, fill = base::list(n = 0)) %>%
    dplyr::mutate(actual = base::factor(actual, labels = labels, ordered = TRUE)) %>%
    ggplot2::ggplot(data = ., mapping = ggplot2::aes(x = cut, y = n, label = n)) +
    ggplot2::geom_bar(stat = "identity", col = "black") +
    ggplot2::geom_label() +
    ggplot2::facet_grid(actual~.) +
    ggplot2::labs(x = "Prediction value",
                  y = "Count",
                  title = "Probability distribution per target class") +
    ggplot2::theme(plot.title = element_text(size = title_size, color = "black", face = "bold", hjust = 0.5, vjust = 0.5),
                   axis.text.y = element_text(size = text_size, color = "black", face = "plain"),
                   axis.text.x = element_text(size = text_size, color = "black", face = "plain"),
                   axis.title.y = element_text(size = text_size, color = "black", face = "bold"),
                   axis.title.x = element_text(size = text_size, color = "black", face = "bold"),
                   axis.ticks = element_line(size = 1, color = "black", linetype = "solid"),
                   axis.ticks.length = unit(0.1, "cm"),
                   plot.background = element_rect(fill = "gray80", color = "black", size = 1, linetype = "solid"),
                   panel.background = element_rect(fill = "gray90", color = "black", size = 0.5, linetype = "solid"),
                   panel.border = element_rect(fill = NA, color = "black", size = 0.5, linetype = "solid"),
                   panel.grid.major.x = element_line(color = "black", linetype = "dotted"),
                   panel.grid.major.y = element_line(color = "black", linetype = "dotted"),
                   panel.grid.minor.x = element_line(linetype = "blank"),
                   panel.grid.minor.y = element_line(linetype = "blank"),
                   plot.caption = element_text(size = text_size, color = "black", face = "bold", hjust = 1),
                   legend.position = "none",
                   strip.background = element_rect(color = "black", fill = "gray80", size = 0.5, linetype = "solid"),
                   strip.text = element_text(size = text_size, face = "bold")) -> plot
  
  bars <- plot$data %>%
    dplyr::select(cut) %>%
    dplyr::distinct() %>%
    base::nrow()
  
  tibble::tibble(actual = base::factor(actual),
                 predicted = predicted) %>%
    dplyr::mutate(cut = ggplot2::cut_interval(predicted, length = 1/bins)) %>%
    dplyr::group_by(cut, actual) %>%
    dplyr::summarise(n = dplyr::n()) %>%
    dplyr::ungroup() %>%
    tidyr::complete(cut, actual, fill = base::list(n = 0)) %>%
    dplyr::mutate(actual = base::factor(actual, labels = labels, ordered = TRUE)) %>%
    tidyr::pivot_wider(id_cols = "actual",
                       names_from = "cut",
                       values_from = "n") %>%
    mutate(Observations = rowSums(.[2:(bars + 1)])) -> results
  
  if (save_plot == TRUE){
    filename <- base::paste(datetime, type_info, "probability_distribution_per_target_class.png", sep = "_")
    ggplot2::ggsave(filename = filename, plot = plot, units = "cm", width = plot_size, height = plot_size)
    base::cat("Plot saved:", base::paste(base::getwd(), filename, sep = "/"), "\n")}
  
  base::invisible(results)
  plot %>%
    base::print(.)
  results %>%
    knitr::kable(.)}

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to all classes:
Display_All_Classes_Predictions_Distribution <- function(actual,
                                                         predicted,
                                                         labels,
                                                         type_info,
                                                         bins = 10,
                                                         text_size = 7,
                                                         title_size = 9,
                                                         save_plot = FALSE,
                                                         plot_size = 20){
  
  datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  
  predicted %>%
    tibble::as_tibble() %>%
    dplyr::mutate(actual = actual,
                  predicted = base::max.col(predicted)) %>%
    tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                        names_to = "class",
                        values_to = "probability") %>%
    dplyr::mutate(cut = ggplot2::cut_interval(probability, length = 1/bins),
                  cut = forcats::fct_rev(base::factor(cut, ordered = TRUE)),
                  class = base::as.numeric(stringr::str_sub(class, 2, -1)),
                  class = base::factor(class, labels = base::paste0("predicted_", labels), levels = 1:base::length(labels), ordered = TRUE),
                  actual = base::factor(actual, labels = base::paste0("actual_", labels), levels = 1:base::length(labels), ordered = TRUE)) %>%
    dplyr::group_by(actual, class, cut) %>%
    dplyr::summarise(n = dplyr::n()) %>%
    dplyr::ungroup() %>%
    tidyr::complete(actual, class, cut, fill = base::list(n = 0)) %>%
    ggplot2::ggplot(data = ., mapping = ggplot2::aes(y = cut, x = n, label = n)) +
    ggplot2::geom_bar(stat = "identity", col = "black") +
    ggplot2::geom_label(color = "black", size = 3, label.size = 0.5, fontface = 1, fill = "white",label.padding = unit(0.15, "lines"), label.r = unit(0, "lines")) +
    ggplot2::facet_grid(actual ~ class) +
    ggplot2::labs(x = "Predicted class",
                  y = "Actual class",
                  title = "Probability distribution per classes") +
    ggplot2::theme(plot.title = element_text(size = title_size, color = "black", face = "bold", hjust = 0.5, vjust = 0.5),
                   axis.text.y = element_text(size = text_size, color = "black", face = "plain"),
                   axis.text.x = element_text(size = text_size, color = "black", face = "plain"),
                   axis.title.y = element_text(size = text_size, color = "black", face = "bold"),
                   axis.title.x = element_text(size = text_size, color = "black", face = "bold"),
                   axis.ticks = element_line(size = 1, color = "black", linetype = "solid"),
                   axis.ticks.length = unit(0.1, "cm"),
                   plot.background = element_rect(fill = "gray80", color = "black", size = 1, linetype = "solid"),
                   panel.background = element_rect(fill = "gray90", color = "black", size = 0.5, linetype = "solid"),
                   panel.border = element_rect(fill = NA, color = "black", size = 0.5, linetype = "solid"),
                   panel.grid.major.x = element_line(color = "black", linetype = "dotted"),
                   panel.grid.major.y = element_line(color = "black", linetype = "dotted"),
                   panel.grid.minor.x = element_line(linetype = "blank"),
                   panel.grid.minor.y = element_line(linetype = "blank"),
                   plot.caption = element_text(size = text_size, color = "black", face = "bold", hjust = 1),
                   legend.position = "none",
                   strip.background = element_rect(color = "black", fill = "gray80", size = 0.5, linetype = "solid"),
                   strip.text = element_text(size = text_size, face = "bold")) +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0.1, 0.1))) -> plot
  
  predicted %>%
    tibble::as_tibble() %>%
    dplyr::mutate(actual = actual,
                  predicted = base::max.col(predicted)) %>%
    tidyr::pivot_longer(cols = dplyr::starts_with("V"),
                        names_to = "class",
                        values_to = "probability") %>%
    dplyr::mutate(cut = ggplot2::cut_interval(probability, length = 1/bins),
                  class = base::as.numeric(stringr::str_sub(class, 2, -1)),
                  class = base::factor(class, labels = labels, levels = 1:base::length(labels), ordered = TRUE),
                  actual = base::factor(actual, labels = labels, levels = 1:base::length(labels), ordered = TRUE)) %>%
    dplyr::group_by(actual, class, cut) %>%
    dplyr::summarise(n = dplyr::n()) %>%
    dplyr::ungroup() %>%
    tidyr::complete(actual, class, cut, fill = base::list(n = 0)) %>%
    dplyr::mutate(actual = base::as.character(actual),
                  class = base::as.character(class),
                  cut = base::as.character(cut)) %>%
    tidyr::pivot_wider(id_cols = base::c("actual", "cut"),
                       names_from = "class",
                       values_from = "n",
                       names_prefix = "predicted_") %>%
    dplyr::mutate(actual = base::paste0("actual_", actual)) -> results
  
  if (save_plot == TRUE){
    filename <- base::paste(datetime, type_info, "probability_distribution_all_classes.png", sep = "_")
    ggplot2::ggsave(filename = filename, plot = plot, units = "cm", width = plot_size, height = plot_size)
    base::cat("Plot saved:", base::paste(base::getwd(), filename, sep = "/"), "\n")}
  
  base::invisible(results)
  plot %>%
    base::print(.)
  results %>%
    knitr::kable(.)}

# ------------------------------------------------------------------------------
# BINARY MODEL EVALUATION
# Function to verify the predictive and classification capabilities of the binary model:
Binary_Classifier_Verification <- function(actual,
                                           predicted,
                                           type_info = "",
                                           cutoff = 0.5,
                                           FN_cost = 1,
                                           FP_cost = 1,
                                           TN_cost = 0,
                                           TP_cost = 0,
                                           save = FALSE,
                                           open = TRUE){
  
  sys_time <- base::Sys.time()
  
  # Libraries:
  if (!base::require(Metrics)){utils::install.packages('Metrics'); base::require('Metrics')}  
  if (!base::require(tidyverse)){utils::install.packages('tidyverse'); base::require('tidyverse')}  
  if (!base::require(tibble)){utils::install.packages('tibble'); base::require('tibble')}  
  if (!base::require(knitr)){utils::install.packages('knitr'); base::require('knitr')}  
  if (!base::require(gt)){utils::install.packages('gt'); base::require('gt')}  
  if (!require(webshot)){utils::install.packages('webshot'); require('webshot')} 
  if (!require(stringr)){utils::install.packages('stringr'); require('stringr')} 
  
  # Confusion matrix explanation:
  result_1 <- tibble::tibble("Confusion Matrix" = base::c("Actual Negative (0)", "Actual Positive (1)"),
                             "Predicted Negative (0)" = base::c("True Negative (TN)", "False Negative (FN)"),
                             "Predicted Positive (1)" = base::c("False Positive (FP)", "True Positive (TP)"))
  
  probability <- predicted
  if(base::length(base::unique(predicted)) > 2){predicted <- base::ifelse(predicted < cutoff, 0, 1)}
  predicted <- base::factor(predicted, levels = base::c(0, 1), labels = base::c(0, 1))
  
  # Confusion matrix result:
  confusion_matrix <- base::table(actual, predicted)
  result_2 <- tibble::tibble("Confusion Matrix" = base::c("Actual Negative (0)", "Actual Positive (1)"),
                             "Predicted Negative (0)" = base::c(confusion_matrix[1, 1], confusion_matrix[2, 1]),
                             "Predicted Positive (1)" = base::c(confusion_matrix[1, 2], confusion_matrix[2, 2])) 
  
  # Assessment of classifier effectiveness:
  OBS <- base::sum(confusion_matrix); OBS_label <- "= TN + FP + FN + TP"
  TN <- confusion_matrix[1, 1]; TN_label <- "= TN"
  FP <- confusion_matrix[1, 2]; FP_label <- "= FP"
  FN <- confusion_matrix[2, 1]; FN_label <- "= FN"
  TP <- confusion_matrix[2, 2]; TP_label <- "= TP"
  P <- FN + TP; P_label <- "= FN + TP"
  N <- TN + FP; N_label <- "= TN + FP"
  
  # Accuracy (ACC):
  ACC <- (TN + TP)/(TN + FN + FP + TP)
  ACC_label <- "= (TN + TP)/(TN + FN + FP + TP) = (TN + TP)/(P + N)"
  
  # Balanced Accuracy (BACC):
  BACC <- (TN/(TN + FP) + TP/(FN + TP))/2
  BACC_label <- "= (TN/(TN + FP) + TP/(FN + TP))/2"
  
  # Area Under Curve (AUC):
  AUC <- Metrics::auc(actual = actual, predicted = probability)
  AUC_label <- "= Area Under ROC Curve"
  # Bias:
  BIAS <- base::mean(base::as.numeric(actual)) - base::mean(base::as.numeric(predicted))
  BIAS_label <- "= mean(actual) - mean(predicted)"
  # Classification Error (CE):
  CE <- (FN + FP)/(TN + FN + FP + TP)
  CE_label <- "= (FN + FP)/(TN + FN + FP + TP) = 1 - (TN + TP)/(TN + FN + FP + TP)"
  # Recall, Sensitivity, hit rate, True Positive Rate (TPR):
  TPR <- TP/(TP + FN)
  TPR_label <- "= TP/(TP + FN) = TP/P = 1 - FNR"
  # Specifity, selectivity, True Negative Rate (TNR):
  TNR <- TN/(TN + FP)
  TNR_label <- "= TN/(TN + FP) = TN/N = 1 - FPR"
  # Precision, Positive Prediction Value (PPV):
  PPV <- TP/(TP + FP)
  PPV_label <- "= TP/(TP + FP) = 1 - FDR"
  # Negative Predictive Value (NPV):
  NPV <- TN/(TN + FN)
  NPV_label <- "= TN/(TN + FN) = 1 - FOR"
  # False Negative Rate (FNR), miss rate:
  FNR <- FN/(FN + TP)
  FNR_label <- "= FN/(FN + TP) = FN/P = 1 - TPR"
  # False Positive Rate (FPR), fall-out:
  FPR <- FP/(FP + TN)
  FPR_label <- "= FP/(FP + TN) = FP/N = 1 - TNR"
  # False Discovery Rate (FDR):
  FDR <- FP/(FP + TP)
  FDR_label <- "= FP/(FP + TP) = 1 - PPV"
  # False Omission Rate (FOR):
  FOR <- FN/(FN + TN)
  FOR_label <- "= FN/(FN + TN) = 1 - NPV"
  # Threat Score (TS), Critical Success Index (CSI):
  TS <- TP/(TP + FN + FP)
  TS_label <- "= TP/(TP + FN + FP)"
  # F1:
  F1 <- (2 * PPV * TPR)/(PPV + TPR)
  F1_label <- "= (2 * PPV * TPR)/(PPV + TPR) = 2 * TP/(2 * TP + FP + FN)"
  # Informedness, Bookmaker Informedness (BM):
  BM <- TPR + TNR - 1
  BM_label <- "= TPR + TNR - 1"
  # Markedness (MK):
  MK <- PPV + NPV - 1
  MK_label <- "= PPV + NPV - 1"
  # Gini Index:
  GINI <- 2 * AUC - 1
  GINI_label <- "= 2 * AUC - 1"
  # Cost:
  COST <- FN * FN_cost + FP * FP_cost + TN * TN_cost + TP * TP_cost
  COST_label <- "= FN * FN_cost + FP * FP_cost + TN * TN_cost + TP * TP_cost"
  
  result_3 <- tibble::tibble(Metric = base::c("Number of Observations", "True Negative", "False Positive", "False Negative", "True Positive",
                                              "Condition Negative", "Condition Positive", "Accuracy", "Balanced Accuracy", "Area Under ROC Curve",
                                              "Bias", "Classification Error", "True Positive Rate", "True Negative Rate",
                                              "Positive Prediction Value", "Negative Predictive Value", "False Negative Rate", "False Positive Rate",
                                              "False Discovery Rate", "False Omission Rate", "Threat Score", "F1 Score",
                                              "Bookmaker Informedness", "Markedness", "Gini Index", "Cost"),
                             `Metric Abb` = base::c("RECORDS", "TN", "FP", "FN", "TP",
                                                    "N", "P", "ACC", "BACC", "AUC",
                                                    "BIAS", "CE", "TPR", "TNR", 
                                                    "PPV", "NPV", "FNR", "FPR",
                                                    "FDR", "FOR", "TS", "F1",
                                                    "BM", "MK", "GINI", "COST"),
                             `Metric Name` = base::c("-", "-", "Type I Error", "Type II Error", "-",
                                                     "-", "-", "-", "-", "-",
                                                     "-", "-", "Sensitivity, Recall, Hit Rate", "Specifity, Selectivity",
                                                     "Precision", "-", "Miss Rate", "Fall-Out",
                                                     "-", "-", "Critical Success Index", "-",
                                                     "-", "-", "-", "-"),
                             Score = base::round(base::c(OBS, TN, FP, FN, TP,
                                                         N, P, ACC, BACC, AUC,
                                                         BIAS, CE, TPR, TNR,
                                                         PPV, NPV, FNR, FPR,
                                                         FDR, FOR, TS, F1,
                                                         BM, MK, GINI, COST), digits = 6),
                             Calculation = base::c(OBS_label, TN_label, FP_label, FN_label, TP_label,
                                                   N_label, P_label, ACC_label, BACC_label, AUC_label,
                                                   BIAS_label, CE_label, TPR_label, TNR_label,
                                                   PPV_label, NPV_label, FNR_label, FPR_label,
                                                   FDR_label, FOR_label, TS_label, F1_label,
                                                   BM_label, MK_label, GINI_label, COST_label),
                             ID = base::c(1:7, 1:19)) %>%
    dplyr::select(ID, Metric, `Metric Abb`, `Metric Name`, Score, Calculation)
  
  result_3_label <- result_3 %>% knitr::kable(.)
  
  result_3 %>%
    dplyr::mutate(Group = base::ifelse(Metric %in% base::c("Number of Observations", "True Negative",
                                                           "False Positive", "False Negative",
                                                           "True Positive", "Condition Positive",
                                                           "Condition Negative"), 
                                       "Confusion Matrix Result", "Assessment of Classifier Effectiveness")) %>%
    gt::gt(rowname_col = "ID", groupname_col = "Group") %>%
    gt::tab_header(title = gt::md(base::paste("Model's evaluation metrics", sys_time)),
                   subtitle = gt::md("Binary classification model")) %>%
    gt::tab_source_note(gt::md(base::paste0("**Options**: ",
                                            "**cutoff** = ", cutoff,
                                            ", **TN_cost** = ", TN_cost,
                                            ", **FP_cost** = ", FP_cost,
                                            ", **FN_cost** = ", FN_cost,
                                            ", **TP_cost** = ", TP_cost))) %>%
    gt::tab_source_note(gt::md("More information available at: **https://github.com/ForesightAdamNowacki/DeepNeuralNetworksRepoR**.")) %>%
    gt::tab_spanner(label = "Metrics section",
                    columns = dplyr::vars(Metric, `Metric Abb`, `Metric Name`)) %>%
    gt::tab_spanner(label = "Performance section",
                    columns = dplyr::vars(Score, Calculation)) %>%
    gt::fmt_number(columns = dplyr::vars(Score),
                   decimals = 4,
                   use_seps = FALSE) %>%
    gt::cols_align(align = "left", columns = dplyr::vars(Metric, `Metric Abb`, `Metric Name`, Calculation)) %>%
    gt::tab_options(heading.background.color = "black",
                    table.background.color = "grey",
                    column_labels.background.color = "black",
                    row_group.background.color = "black",
                    source_notes.background.color = "black",
                    table.border.top.color = "black",
                    table.border.top.width = gt::px(3),
                    table.border.bottom.color = "black",
                    table.border.bottom.width = gt::px(3),
                    heading.title.font.size = 16,
                    table.font.size = 12,
                    source_notes.font.size = 10,
                    table.width = gt::pct(100),
                    data_row.padding = gt::px(5),
                    row_group.padding = gt::px(10),
                    source_notes.padding = gt::px(5)) %>% 
    gt::opt_table_outline(width = gt::px(3), color = "black") %>%
    gt::opt_table_lines() -> gt_table
  
  if (save == TRUE){
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste(sys_time, type_info, "binary_model_evaluation_metrics.png", sep = "_"), ":", "-"),
               vwidth = 900,
               vheight = 1600,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, type_info, "binary_model_evaluation_metrics.png", sep = "_"), ":", "-"))
    }
  }
  
  gt_table %>% base::print(.)
  
  base::invisible(base::list("Confusion_Matrix_Explanation" = result_1,
                             "Confusion_Matrix_Result" = result_2,
                             "Assessment_of_Classifier_Effectiveness" = result_3))}

# ------------------------------------------------------------------------------
# CUT-OFF OPTIMIZATION
# Function to optimize the cut-off level in relation to many evaluation metrics.
Binary_Classifier_Cutoff_Optimization <- function(actual,
                                                  predicted, 
                                                  type_info = "",
                                                  cuts = 25,
                                                  TN_cost = 0, 
                                                  FP_cost = 1, 
                                                  FN_cost = 1, 
                                                  TP_cost = 0,
                                                  text_size = 8,
                                                  title_size = 10,
                                                  key_metric = CUTOFF,
                                                  key_metric_as_string = FALSE,
                                                  ascending = TRUE,
                                                  seed_value = 42,
                                                  top = 25,
                                                  save = FALSE,
                                                  open = TRUE){
  
  sys_time <- base::Sys.time()
  
  # Libraries:
  if (!require(Metrics)){utils::install.packages('Metrics'); require('Metrics')}  
  if (!require(tidyverse)){utils::install.packages('tidyverse'); require('tidyverse')}  
  if (!require(tibble)){utils::install.packages('tibble'); require('tibble')}  
  if (!require(gridExtra)){utils::install.packages('gridExtra'); require('gridExtra')}  
  if (!require(gt)){utils::install.packages('gt'); require('gt')}  
  if (!require(webshot)){utils::install.packages('webshot'); require('webshot')} 
  if (!require(stringr)){utils::install.packages('stringr'); require('stringr')} 
  
  
  if (key_metric_as_string == FALSE){
    key_metric <- dplyr::enquo(key_metric) 
    key_metric_name <- dplyr::quo_name(key_metric)}
  
  if (key_metric_as_string == TRUE){
    key_metric <- rlang::sym(key_metric)
    key_metric <- dplyr::enquo(key_metric) 
    key_metric_name <- dplyr::quo_name(key_metric)}
  
  
  
  
  # key_metric <- dplyr::enquo(key_metric) 
  # key_metric_name <- dplyr::quo_name(key_metric)
  
  base::set.seed(seed = seed_value)
  cuts_values <- stats::runif(n = cuts, min = 0, max = 1)
  cuts_values <- base::sort(x = cuts_values, decreasing = FALSE)
  
  df <- tibble::tibble(ID = 1:cuts,
                       CUTOFF = cuts_values,
                       RECORDS = base::numeric(cuts),
                       TN = base::numeric(cuts),
                       FP = base::numeric(cuts),
                       FN = base::numeric(cuts),
                       TP = base::numeric(cuts),
                       P = base::numeric(cuts),
                       N = base::numeric(cuts),
                       ACC = base::numeric(cuts),
                       BACC = base::numeric(cuts),
                       AUC = base::numeric(cuts),
                       BIAS = base::numeric(cuts),
                       CE = base::numeric(cuts),
                       TPR = base::numeric(cuts),
                       TNR = base::numeric(cuts),
                       PPV = base::numeric(cuts),
                       NPV = base::numeric(cuts),
                       FNR = base::numeric(cuts),
                       FPR = base::numeric(cuts),
                       FDR = base::numeric(cuts),
                       FOR = base::numeric(cuts),
                       TS = base::numeric(cuts),
                       F1 = base::numeric(cuts),
                       BM = base::numeric(cuts),
                       MK = base::numeric(cuts),
                       GINI = base::numeric(cuts),
                       COST = base::numeric(cuts))
  
  actual = base::factor(actual, levels = base::c(0, 1))
  
  for (i in 1:cuts){
    
    predicted_class <- base::factor(base::ifelse(predicted < df$CUTOFF[i], 0, 1), levels = c(0, 1))
    confusion_matrix <- base::table(actual, predicted_class)
    
    df$RECORDS <- base::sum(confusion_matrix)
    df$TN[i] <- confusion_matrix[1, 1]
    df$FP[i] <- confusion_matrix[1, 2]
    df$FN[i] <- confusion_matrix[2, 1]
    df$TP[i] <- confusion_matrix[2, 2]
    df$N <- df$TN[i] + df$FP[i]
    df$P <- df$FN[i] + df$TP[i]
    df$ACC[i] <- (df$TN[i] + df$TP[i])/(df$TN[i] + df$FN[i] + df$FP[i] + df$TP[i])
    df$BACC[i] <- (df$TN[i]/(df$TN[i] + df$FP[i]) + df$TP[i]/(df$FN[i] + df$TP[i]))/2
    df$AUC[i] <- Metrics::auc(actual = actual, predicted = predicted)
    df$BIAS[i] <- base::mean(base::as.numeric(actual)) - base::mean(base::as.numeric(predicted_class))
    df$CE[i] <- (df$FN[i] + df$FP[i])/(df$TN[i] + df$FN[i] + df$FP[i] + df$TP[i])
    df$TPR[i] <- df$TP[i]/(df$TP[i] + df$FN[i])
    df$TNR[i] <- df$TN[i]/(df$TN[i] + df$FP[i])
    df$PPV[i] <- df$TP[i]/(df$TP[i] + df$FP[i])
    df$NPV[i] <- df$TN[i]/(df$TN[i] + df$FN[i])
    df$FNR[i] <- df$FN[i]/(df$FN[i] + df$TP[i])
    df$FPR[i] <- df$FP[i]/(df$FP[i] + df$TN[i])
    df$FDR[i] <- df$FP[i]/(df$FP[i] + df$TP[i])
    df$FOR[i] <- df$FN[i]/(df$FN[i] + df$TN[i])
    df$TS[i] <- df$TP[i]/(df$TP[i] + df$FN[i] + df$FP[i])
    df$F1[i] <- (2 * df$PPV[i] * df$TPR[i])/(df$PPV[i] + df$TPR[i])
    df$BM[i] <- df$TPR[i] + df$TNR[i] - 1
    df$MK[i] <- df$PPV[i] + df$NPV[i] - 1
    df$GINI[i] <- 2 * df$AUC[i] - 1
    df$COST[i] <- TN_cost * df$TN[i] + FP_cost * df$FP[i] + FN_cost * df$FN[i] + TP_cost * df$TP[i]
  }
  
  if(ascending == TRUE){
    df %>%
      dplyr::arrange(!!key_metric) -> df
  } else {
    df %>%
      dplyr::arrange(dplyr::desc(!!key_metric)) -> df
  }
  
  df %>%
    utils::head(., top) %>%
    dplyr::mutate(ID = dplyr::row_number()) -> df
  
  df %>%
    gt::gt(rowname_col = "ID") %>%
    gt::tab_header(title = gt::md(base::paste("Cut-off value optimization", sys_time)),
                   subtitle = gt::md("Binary classification model")) %>%
    gt::fmt_number(columns = dplyr::vars(CUTOFF, ACC, BACC, AUC, BIAS, CE, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, F1, BM, MK, GINI),
                   decimals = 4,
                   use_seps = FALSE) %>%
    gt::tab_spanner(label = "Confusion Matrix Result",
                    columns = dplyr::vars(RECORDS, TN, FP, FN, TP, N, P)) %>%
    gt::tab_spanner(label = "Assessment of Classifier Effectiveness",
                    columns = dplyr::vars(ACC, BACC, AUC, BIAS, CE, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, F1, BM, MK, GINI, COST)) %>%
    gt::tab_options(heading.background.color = "black",
                    table.background.color = "grey",
                    column_labels.background.color = "black",
                    row_group.background.color = "black",
                    source_notes.background.color = "black",
                    table.border.top.color = "black",
                    table.border.top.width = gt::px(3),
                    table.border.bottom.color = "black",
                    table.border.bottom.width = gt::px(3),
                    heading.title.font.size = 16,
                    table.font.size = 12,
                    source_notes.font.size = 10,
                    table.width = gt::pct(100),
                    data_row.padding = gt::px(5),
                    source_notes.padding = gt::px(5),
                    grand_summary_row.border.color = "black",
                    grand_summary_row.border.width = gt::px(3),
                    grand_summary_row.padding = gt::px(5)) %>% 
    gt::opt_table_outline(width = gt::px(3), color = "black") %>%
    gt::grand_summary_rows(fns = list("Mean value" = "mean",
                                      "Median value" = "median",
                                      "Minimum value" = "min",
                                      "Maximum value" = "max"),
                           formatter = fmt_number,
                           columns = dplyr::vars(CUTOFF, ACC, BACC, AUC, BIAS, CE, TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, TS, F1, BM, MK, GINI),
                           decimals = 4,
                           use_seps = FALSE) %>%
    gt::grand_summary_rows(fns = list("Mean value" = "mean",
                                      "Median value" = "median",
                                      "Minimum value" = "min",
                                      "Maximum value" = "max"),
                           formatter = fmt_number,
                           columns = dplyr::vars(RECORDS, TN, FP, FN, TP, N, P, COST),
                           decimals = 0,
                           use_seps = FALSE) %>%
    gt::tab_source_note(gt::md(base::paste0("**Options**: ",
                                            "**cuts** = ", cuts,
                                            ", **TN_cost** = ", TN_cost,
                                            ", **FP_cost** = ", FP_cost,
                                            ", **FN_cost** = ", FN_cost,
                                            ", **TP_cost** = ", TP_cost,
                                            ", **key_metric** = ", key_metric_name, 
                                            ", **ascending** = ", ascending,
                                            ", **seed_value** = ", seed_value,
                                            ", **top** = ", top))) %>%
    gt::tab_source_note(gt::md("More information available at: **https://github.com/ForesightAdamNowacki/DeepNeuralNetworksRepoR**.")) %>%
    gt::opt_table_lines() -> gt_table
  
  if (save == TRUE){
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste(sys_time, type_info, "binary_model_cutoff_value_optimization.png", sep = "_"), ":", "-"),
               vwidth = 1600,
               vheight = 900,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste(sys_time, type_info, "binary_model_cutoff_value_optimization.png", sep = "_"), ":", "-"))
    }
  }
  
  gt_table %>% base::print(.)
  base::invisible(df)}

# ------------------------------------------------------------------------------
# Categorical model evaluation:
Categorical_Classifier_Verification <- function(actual,
                                                probabilities,
                                                labels,
                                                type_info = "",
                                                save = TRUE,
                                                open = FALSE){
  
  sys_time = base::Sys.time()
  
  # Libraries:
  if (!require(Metrics)){utils::install.packages('Metrics'); require('Metrics')}  
  if (!require(tidyverse)){utils::install.packages('tidyverse'); require('tidyverse')}  
  if (!require(tibble)){utils::install.packages('tibble'); require('tibble')}  
  if (!require(gridExtra)){utils::install.packages('gridExtra'); require('gridExtra')}  
  if (!require(gt)){utils::install.packages('gt'); require('gt')}  
  if (!require(webshot)){utils::install.packages('webshot'); require('webshot')} 
  if (!require(stringr)){utils::install.packages('stringr'); require('stringr')} 
  
  predicted <- base::max.col(probabilities)
  
  base::table(base::factor(actual, levels = 1:base::length(labels), labels = labels), 
              base::factor(predicted, levels = 1:base::length(labels), labels = labels)) %>%
    base::as.data.frame() %>%
    tibble::as_tibble() %>%
    dplyr::rename(actual = Var1,
                  predicted = Var2,
                  count = Freq) %>%
    tidyr::complete(actual, predicted, fill = base::list(count = 0)) %>%
    dplyr::arrange(actual, predicted) %>%
    dplyr::group_by(actual) %>%
    dplyr::mutate(records = base::sum(count),
                  freq = count/records) %>%
    dplyr::filter(actual == predicted) %>%
    dplyr::mutate(predicted = NULL) %>%
    dplyr::rename(correct = count,
                  accuracy = freq) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(incorrect = records - correct,
                  inaccuracy = 1 - accuracy) %>%
    dplyr::select(actual, records, correct, incorrect, accuracy, inaccuracy) -> stats; stats
  
  base::table(base::factor(actual, levels = 1:base::length(labels), labels = labels), 
              base::factor(predicted, levels = 1:base::length(labels), labels = labels)) %>%
    base::as.data.frame() %>%
    tibble::as_tibble() %>%
    dplyr::rename(actual = Var1,
                  predicted = Var2,
                  count = Freq) %>%
    tidyr::complete(actual, predicted, fill = base::list(count = 0)) %>%
    dplyr::group_by(predicted) %>%
    dplyr::mutate(records = base::sum(count),
                  precision = count/records) %>%
    dplyr::filter(actual == predicted) %>%
    dplyr::ungroup() %>%
    dplyr::select(actual, precision) -> precision; precision
  
  base::table(base::factor(actual, levels = 1:base::length(labels), labels = labels), 
              base::factor(predicted, levels = 1:base::length(labels), labels = labels)) %>%
    base::as.data.frame() %>%
    tibble::as_tibble() %>%
    dplyr::rename(actual = Var1,
                  predicted = Var2,
                  count = Freq) %>%
    tidyr::complete(actual, predicted, fill = base::list(count = 0)) %>%
    dplyr::group_by(actual) %>%
    dplyr::mutate(records = base::sum(count),
                  recall = count/records) %>%
    dplyr::filter(actual == predicted) %>%
    dplyr::ungroup() %>%
    dplyr::select(actual, recall) -> recall; recall
  
  stats %>%
    dplyr::left_join(precision, by = "actual") %>%
    dplyr::left_join(recall, by = "actual") %>%
    dplyr::mutate(f1 = (2 * precision * recall)/(precision + recall)) %>%
    tidyr::replace_na(base::list(f1 = 0)) %>%
    dplyr::mutate(type = "Split into categories",
                  actual = base::as.character(actual)) %>%
    dplyr::rename(class = actual) -> stats_2; stats_2
  
  stats_2 %>%
    dplyr::mutate(class = "Overall") %>%
    dplyr::group_by(class) %>%
    dplyr::summarise(records = base::sum(records),
                     correct = base::sum(correct),
                     incorrect = records - correct,
                     accuracy = correct/records,
                     inaccuracy = incorrect/records,
                     precision = base::mean(precision),
                     recall = base::mean(recall),
                     f1 = base::mean(f1),
                     type = "All categories") -> stats_3; stats_3
  
  dplyr::bind_rows(stats_2, stats_3) %>%
    dplyr::rename(Class = class,
                  Records = records,
                  Correct = correct,
                  Incorrect = incorrect, 
                  Accuracy = accuracy,
                  Inaccuracy = inaccuracy,
                  Precision = precision,
                  Recall = recall,
                  F1 = f1,
                  Type = type) -> stats_4; stats_4
  
  stats_2 %>%
    dplyr::rename(Class = class,
                  Records = records,
                  Correct = correct,
                  Incorrect = incorrect, 
                  Accuracy = accuracy,
                  Inaccuracy = inaccuracy,
                  Precision = precision,
                  Recall = recall,
                  F1 = f1,
                  Type = type) -> stats_2
  
  stats_4 %>%
    dplyr::ungroup() %>%
    gt::gt(rowname_col = "Class", groupname_col = "Type") %>%
    gt::tab_header(title = gt::md(base::paste("Model's evaluation metrics", sys_time)),
                   subtitle = gt::md("Categorical classification model")) %>%
    gt::tab_source_note(gt::md("More information available at: **https://github.com/ForesightAdamNowacki/DeepNeuralNetworksRepoR**.")) %>%
    gt::fmt_number(columns = dplyr::vars(Accuracy, Inaccuracy, Precision, Recall, F1),
                   decimals = 4,
                   use_seps = FALSE) %>%
    gt::fmt_number(columns = dplyr::vars(Records, Correct, Incorrect),
                   decimals = 0,
                   use_seps = FALSE) %>%
    gt::tab_spanner(label = "Counts",
                    columns = dplyr::vars(Records, Correct, Incorrect)) %>% 
    gt::tab_spanner(label = "Metrics",
                    columns = dplyr::vars(Accuracy, Inaccuracy, Precision, Recall, F1)) %>%
    gt::tab_options(heading.background.color = "black",
                    table.background.color = "grey",
                    column_labels.background.color = "black",
                    row_group.background.color = "black",
                    source_notes.background.color = "black",
                    table.border.top.color = "black",
                    table.border.top.width = gt::px(3),
                    table.border.bottom.color = "black",
                    table.border.bottom.width = gt::px(3),
                    heading.title.font.size = 16,
                    table.font.size = 12,
                    source_notes.font.size = 10,
                    table.width = gt::pct(100),
                    data_row.padding = gt::px(5),
                    row_group.padding = gt::px(10),
                    source_notes.padding = gt::px(5)) %>% 
    gt::opt_table_outline(width = gt::px(3), color = "black") %>%
    gt::opt_table_lines() -> gt_table
  
  if (save == TRUE){
    
    base::print(gt_table)
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste(sys_time, type_info, "categorical_model_evaluation_metrics.png", sep = "_"), ":", "-"),
               vwidth = 900,
               vheight = 600,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste(sys_time, type_info, "categorical_model_evaluation_metrics.png", sep = "_"), ":", "-"))
    }
  }
  
  base::return(base::list(stats_2,
                          stats_3,
                          stats_4))
}

# ------------------------------------------------------------------------------
# Prepare data for K-Fold Cross Validation:
Create_KFolds_Directories <- function(data_dir,
                                      target_dir,
                                      folds = 5,
                                      seed = 42){
  
  base::set.seed(seed = seed)
  sys_time <- stringr::str_replace_all(base::Sys.time(), ":", "-")
  class_dirs <- base::list.dirs(data_dir)[2:base::length(base::list.dirs(data_dir))]
  classes <- base::basename(class_dirs)
  folders <- base::paste("Fold", 1:folds, sep = "_")
  
  files <- base::list()
  for (i in base::seq_along(class_dirs)){
    files[[i]] <- base::list.files(class_dirs[i])}
  names(files) <- base::basename(class_dirs)
  
  files <- tibble::tibble(file = base::do.call(c, files),
                          class = base::rep(names(base::sapply(files, length)), times = base::sapply(files, length))) %>%
    dplyr::mutate(original_file_path = base::paste(data_dir, class, file, sep = "/"),
                  fold = caret::createFolds(class, k = folds, list = FALSE))
  
  # base::unlink(target_dir, recursive = TRUE, force = TRUE)
  base::dir.create(path = target_dir, showWarnings = FALSE, recursive = TRUE)
  
  for (i in 1:folds){
    
    files_2 <- files %>%
      dplyr::filter(fold == i)
    fold <- base::paste("fold", i, sep = "_")
    fold_dir <- base::paste(target_dir, fold, sep = "/")
    base::dir.create(path = fold_dir, showWarnings = FALSE, recursive = TRUE)
    
    for (j in base::seq_along(classes)){
      
      class_dir <- base::paste(target_dir, fold, classes[j], sep = "/")
      base::dir.create(path = class_dir, showWarnings = FALSE, recursive = TRUE)
      files_3 <- files_2 %>%
        dplyr::filter(class == classes[j])
      base::file.copy(from = files_3$original_file_path,
                      to = base::paste(class_dir, files_3$file, sep = "/"))
    }
  }
  
  folds <- base::list.files(target_dir)[base::grepl("fold_", list.files(target_dir))]
  steps <- base::paste("step", 1:base::length(folds), sep = "_")
  
  '%!in%' <- Negate('%in%')
  
  for (i in base::seq_along(steps)){
    
    step_dir <- base::paste(target_dir, steps[i], sep = "/")
    train_dir <- base::paste(step_dir, "train", sep = "/")
    validation_dir <- base::paste(step_dir, "validation", sep = "/")
    set_dirs <- base::c(train_dir, validation_dir)
    
    base::dir.create(path = step_dir, showWarnings = FALSE, recursive = TRUE)
    base::dir.create(path = train_dir, showWarnings = FALSE, recursive = TRUE)
    base::dir.create(path = validation_dir, showWarnings = FALSE, recursive = TRUE)
    
    for (j in base::seq_along(set_dirs)){
      
      base::setwd(set_dirs[j])
      
      for (k in base::seq_along(classes)){
        
        if (j == 1){
          
          class_dir <- base::paste(base::getwd(), classes[k], sep = "/")
          base::dir.create(path = class_dir, showWarnings = FALSE, recursive = TRUE)
          train_files_from <- files %>%
            dplyr::filter(fold %!in% i) %>%
            dplyr::filter(class == classes[k]) 
          
          base::file.copy(from = train_files_from$original_file_path,
                          to = base::paste(class_dir, train_files_from$file, sep = "/"))}
        
        if (j == 2){
          
          class_dir <- base::paste(base::getwd(), classes[k], sep = "/")
          base::dir.create(path = class_dir, showWarnings = FALSE, recursive = TRUE)
          validation_files_from <- files %>%
            dplyr::filter(fold %in% i) %>%
            dplyr::filter(class == classes[k]) 
          base::file.copy(from = validation_files_from$original_file_path,
                          to = base::paste(class_dir, validation_files_from$file, sep = "/"))}
      }
    }
  }
  
  base::setwd(target_dir)
  
  for (i in base::seq_along(steps)){
    
    col_name <- steps[i]
    files <- files %>%
      dplyr::mutate(!!rlang::sym(col_name) := base::ifelse(fold == i, "Validation", "Train"))}
  
  readr::write_csv(files, base::paste(sys_time, "Cross_Validation_Splits.csv"))
}

# ------------------------------------------------------------------------------
# Display list structure:
Display_List_Structure <- function(list, n = 1){
  for(i in 1:base::length(list)){
    for(j in 1:base::length(list[[i]])){
      list[[i]][[j]] %>%
        utils::head(n = n) %>%
        knitr::kable() %>%
        base::print()}}}

# ------------------------------------------------------------------------------
# Build Ensemble model for binary classification problem:
Binary_Ensemble_Model <- function(models_vector,
                                  optimization_dataset = "train", # "train", "validation", "train+validation"
                                  save_option = FALSE,
                                  default_cutoff = 0.5,
                                  cuts = 10,
                                  weights = 10,
                                  key_metric = "ACC",
                                  key_metric_as_string = TRUE,
                                  ascending = FALSE,
                                  top = 10,
                                  seed = 42,
                                  summary_type = "mean",
                                  cwd = models_store_dir,
                                  n = 3){ 
  
  train_pattern <- "train_binary_probabilities"
  validation_pattern <- "validation_binary_probabilities"
  test_dir <- "test_binary_probabilities"
  
  dataset_types <- base::c("train_dataset", "validation_dataset", "test_dataset")
  
  all_predictions <- base::list()
  for (i in base::seq_along(models_vector)){
    model <- base::list()
    model[[1]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = train_pattern, full.names = TRUE))
    model[[2]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = validation_pattern, full.names = TRUE))
    model[[3]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = test_dir, full.names = TRUE))
    all_predictions[[i]] <- model}
  
  Display_List_Structure(all_predictions, n = n)
  
  # ------------------------------------------------------------------------------
  # Predictions:
  # Train:
  train_predictions <- base::list()
  for (i in 1:base::length(all_predictions)){train_predictions[[i]] <- all_predictions[[i]][[1]]$V2}
  train_predictions <- base::do.call(base::cbind, train_predictions)
  base::colnames(train_predictions) <- models_vector
  
  # Validation:
  validation_predictions <- base::list()
  for (i in 1:base::length(all_predictions)){validation_predictions[[i]] <- all_predictions[[i]][[2]]$V2}
  validation_predictions <- base::do.call(base::cbind, validation_predictions)
  base::colnames(validation_predictions) <- models_vector
  
  # Test:
  test_predictions <- base::list()
  for (i in 1:base::length(all_predictions)){test_predictions[[i]] <- all_predictions[[i]][[3]]$V2}
  test_predictions <- base::do.call(base::cbind, test_predictions)
  base::colnames(test_predictions) <- models_vector
  
  # ------------------------------------------------------------------------------
  # Actual:
  train_actual <- all_predictions[[1]][[1]]$actual_class
  validation_actual <- all_predictions[[1]][[2]]$actual_class
  test_actual <- all_predictions[[1]][[3]]$actual_class
  
  # ------------------------------------------------------------------------------
  # Change working directory to save all files in Ensemble_Model Binary Folder:
  base::setwd(cwd)
  
  # ------------------------------------------------------------------------------
  # Train results for single component models:
  train_default <- base::list()
  for (i in 1:base::length(models_vector)){
    Assessment_of_Classifier_Effectiveness <- Binary_Classifier_Verification(actual = train_actual,
                                                                             predicted = train_predictions[,i],
                                                                             cutoff = default_cutoff,
                                                                             type_info = base::paste(models_vector[i], "default_cutoff", dataset_types[1], sep = "_"),
                                                                             save = save_option,
                                                                             open = FALSE)[[3]]
    train_default[[i]] <- Assessment_of_Classifier_Effectiveness}
  
  train_default_summary <- tibble::tibble(Metric = train_default[[1]]$Metric)
  for (i in 1:base::length(models_vector)){train_default_summary <- dplyr::bind_cols(train_default_summary, train_default[[i]][5])}
  base::colnames(train_default_summary) <- base::c("Metric", models_vector)
  
  # ------------------------------------------------------------------------------
  # Validation results for single component models:
  validation_default <- base::list()
  for (i in 1:base::length(models_vector)){
    Assessment_of_Classifier_Effectiveness <- Binary_Classifier_Verification(actual = validation_actual,
                                                                             predicted = validation_predictions[,i],
                                                                             cutoff = default_cutoff,
                                                                             type_info = base::paste(models_vector[i], "default_cutoff", dataset_types[2], sep = "_"),
                                                                             save = save_option,
                                                                             open = FALSE)[[3]]
    validation_default[[i]] <- Assessment_of_Classifier_Effectiveness}
  
  validation_default_summary <- tibble::tibble(Metric = validation_default[[1]]$Metric)
  for (i in 1:base::length(models_vector)){validation_default_summary <- dplyr::bind_cols(validation_default_summary, validation_default[[i]][5])}
  base::colnames(validation_default_summary) <- base::c("Metric", models_vector)
  
  # ------------------------------------------------------------------------------
  # Test results for single component models:
  test_default <- base::list()
  for (i in 1:base::length(models_vector)){
    Assessment_of_Classifier_Effectiveness <- Binary_Classifier_Verification(actual = test_actual,
                                                                             predicted = test_predictions[,i],
                                                                             cutoff = default_cutoff,
                                                                             type_info = base::paste(models_vector[i], "default_cutoff", dataset_types[3], sep = "_"),
                                                                             save = save_option,
                                                                             open = FALSE)[[3]]
    test_default[[i]] <- Assessment_of_Classifier_Effectiveness}
  
  test_default_summary <- tibble::tibble(Metric = test_default[[1]]$Metric)
  for (i in 1:base::length(models_vector)){test_default_summary <- dplyr::bind_cols(test_default_summary, test_default[[i]][5])}
  base::colnames(test_default_summary) <- base::c("Metric", models_vector)
  
  # ------------------------------------------------------------------------------
  # Optimization dataset:
  if (optimization_dataset == "train"){
    actual_optimization = train_actual
    predictions_optimization = train_predictions}
  
  if (optimization_dataset == "validation"){
    actual_optimization = validation_actual
    predictions_optimization = validation_predictions}
  
  if (optimization_dataset == "train+validation"){
    actual_optimization = base::c(train_actual, validation_actual)
    predictions_optimization = base::rbind(train_predictions, validation_predictions)}
  
  # ------------------------------------------------------------------------------
  # Optimize cutoff and weights in binary ensemble model on selected data using simulation approach:
  ensemble_optimization <- Optimize_Binary_Ensemble_Cutoff_Model(actual = actual_optimization,
                                                                 predictions = predictions_optimization,
                                                                 cuts = cuts,
                                                                 weights = weights,
                                                                 key_metric = key_metric,
                                                                 key_metric_as_string = key_metric_as_string,
                                                                 ascending = ascending,
                                                                 top = top,
                                                                 seed = seed,
                                                                 summary_type = summary_type)
  ensemble_optimization_cutoff <- ensemble_optimization[[3]] %>%
    dplyr::pull()
  
  ensemble_optimization_weights <- ensemble_optimization[[4]] %>%
    tidyr::pivot_longer(cols = dplyr::everything()) %>%
    dplyr::select(value) %>%
    dplyr::pull()
  
  # ------------------------------------------------------------------------------
  # Ensemble model predictions:
  train_result <- mapply("*", base::as.data.frame(train_predictions), ensemble_optimization_weights) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(prediction = base::rowSums(.)) %>%
    dplyr::select(prediction) %>%
    dplyr::pull()
  
  validation_result <- mapply("*", base::as.data.frame(validation_predictions), ensemble_optimization_weights) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(prediction = base::rowSums(.)) %>%
    dplyr::select(prediction) %>%
    dplyr::pull()
  
  test_result <- mapply("*", base::as.data.frame(test_predictions), ensemble_optimization_weights) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(prediction = base::rowSums(.)) %>%
    dplyr::select(prediction) %>%
    dplyr::pull()
  
  ensemble_model_predictions <- base::list(train_result,
                                           validation_result,
                                           test_result)
  
  all_ensemble_model_predictions <- base::list(tibble::as_tibble(train_predictions) %>%
                                                 dplyr::mutate(Ensemble_Model = train_result),
                                               tibble::as_tibble(validation_predictions) %>%
                                                 dplyr::mutate(Ensemble_Model = validation_result),
                                               tibble::as_tibble(test_predictions) %>%
                                                 dplyr::mutate(Ensemble_Model = test_result))
  
  # ------------------------------------------------------------------------------
  # Ensemble model results on train data:
  train_dataset_ensemble_model_results <- Binary_Classifier_Verification(actual = train_actual,
                                                                         predicted = ensemble_model_predictions[[1]],
                                                                         cutoff = ensemble_optimization_cutoff,
                                                                         type_info = base::paste("Ensemble_Model", dataset_types[1], sep = "_"),
                                                                         save = save_option,
                                                                         open = FALSE)[[3]]
  
  train_dataset_ensemble_model_results <- train_dataset_ensemble_model_results %>%
    dplyr::select(Metric, Score) %>%
    dplyr::rename(Ensemble_Model = Score)
  
  if (save_option == TRUE){
    datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
    train_default_summary %>%
      dplyr::left_join(train_dataset_ensemble_model_results, by = "Metric") %>%
      readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Ensemble_Model_train_dataset_results_summary_comparison.csv", sep = "_"), sep = "/"))
    base::Sys.sleep(time = 1)}
  
  # ------------------------------------------------------------------------------
  # Ensemble model results on validation data:
  validation_dataset_ensemble_model_results <- Binary_Classifier_Verification(actual = validation_actual,
                                                                              predicted = ensemble_model_predictions[[2]],
                                                                              cutoff = ensemble_optimization_cutoff,
                                                                              type_info = base::paste("Ensemble_Model", dataset_types[2], sep = "_"),
                                                                              save = save_option,
                                                                              open = FALSE)[[3]]
  
  validation_dataset_ensemble_model_results <- validation_dataset_ensemble_model_results %>%
    dplyr::select(Metric, Score) %>%
    dplyr::rename(Ensemble_Model = Score)
  
  if (save_option == TRUE){
    datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
    validation_default_summary %>%
      dplyr::left_join(validation_dataset_ensemble_model_results, by = "Metric") %>%
      readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Ensemble_Model_validation_dataset_results_summary_comparison.csv", sep = "_"), sep = "/"))
    base::Sys.sleep(time = 1)}
  
  # ------------------------------------------------------------------------------
  # Ensemble model results on test data:
  test_dataset_ensemble_model_results <- Binary_Classifier_Verification(actual = test_actual,
                                                                        predicted = ensemble_model_predictions[[3]],
                                                                        cutoff = ensemble_optimization_cutoff,
                                                                        type_info = base::paste("Ensemble_Model", dataset_types[3], sep = "_"),
                                                                        save = save_option,
                                                                        open = FALSE)[[3]]
  
  test_dataset_ensemble_model_results <- test_dataset_ensemble_model_results %>%
    dplyr::select(Metric, Score) %>%
    dplyr::rename(Ensemble_Model = Score)
  
  if (save_option == TRUE){
    datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
    test_default_summary %>%
      dplyr::left_join(test_dataset_ensemble_model_results, by = "Metric") %>%
      readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Ensemble_Model_test_dataset_results_summary_comparison.csv", sep = "_"), sep = "/"))}
  
  # ------------------------------------------------------------------------------
  # Set the initial working directory:
  base::setwd(".."); base::setwd("..")
  
  # ------------------------------------------------------------------------------
  # Final summary of ensemble model results:
  ensemble_model_summary <- base::list(train_dataset_results = train_default_summary %>%
                                         dplyr::left_join(train_dataset_ensemble_model_results, by = "Metric"),
                                       validation_dataset_results = validation_default_summary %>%
                                         dplyr::left_join(validation_dataset_ensemble_model_results, by = "Metric"),
                                       test_dataset_results = test_default_summary %>%
                                         dplyr::left_join(test_dataset_ensemble_model_results, by = "Metric"))
  
  ensemble_model_summary %>%
    base::lapply(., knitr::kable)
  
  base::return(base::list(all_optimization_combinations = ensemble_optimization[[1]],
                          top_optimization_combinations = ensemble_optimization[[2]],
                          optimal_cutoff = ensemble_optimization_cutoff,
                          optimal_weights = ensemble_optimization_weights,
                          train_dataset_results = train_default_summary %>%
                            dplyr::left_join(train_dataset_ensemble_model_results, by = "Metric"),
                          validation_dataset_results = validation_default_summary %>%
                            dplyr::left_join(validation_dataset_ensemble_model_results, by = "Metric"),
                          test_dataset_results = test_default_summary %>%
                            dplyr::left_join(test_dataset_ensemble_model_results, by = "Metric"),
                          train_models_predictions = all_ensemble_model_predictions[[1]],
                          train_actual_class = train_actual,
                          validation_models_predictions = all_ensemble_model_predictions[[2]],
                          validation_actual_class = validation_actual,
                          test_models_predictions = all_ensemble_model_predictions[[3]],
                          test_actual_class = test_actual))
}

# ------------------------------------------------------------------------------
# Calculate final predictions for several categorical models with indicated partial weights:
Multiply_List_Values <- function(list, weights_vector){
  for(i in 1:base::length(list)){
    list[[i]] <- list[[i]] * weights_vector[i]}
  base::return(list)}

# ------------------------------------------------------------------------------
# Optimize Categorical Ensemble Model:
Optimize_Categorical_Ensemble_Cutoff_Model <- function(actual_class,
                                                       predictions, # list
                                                       weights,
                                                       labels,
                                                       models_vector,
                                                       key_metric = Accuracy,
                                                       key_metric_as_string = FALSE,
                                                       ascending = FALSE,
                                                       summary_type = "mean",
                                                       seed = 42,
                                                       top = 10){
  
  if (key_metric_as_string == FALSE){
    key_metric <- dplyr::enquo(key_metric) 
    key_metric_name <- dplyr::quo_name(key_metric)}
  
  if (key_metric_as_string == TRUE){
    key_metric <- rlang::sym(key_metric)
    key_metric <- dplyr::enquo(key_metric) 
    key_metric_name <- dplyr::quo_name(key_metric)}
  
  # Generate waights:
  base::set.seed(seed = seed)
  weights_ <- base::matrix(data = stats::runif(base::length(predictions) * weights, min = 0, max = 1),
                           nrow = weights,
                           ncol = base::length(predictions))
  
  results_list <- base::list()
  weights_list <- base::list()
  base::cat("\n", "Ensemble model optimization:", "\n")
  pb = txtProgressBar(min = 0, max = weights, initial = 0, style = 3) 
  
  for (j in 1:weights){
    weights_vector <- weights_[j,]/base::sum(weights_[j,])
    predictions_table <- Multiply_List_Values(predictions, weights_vector) %>%
      base::Reduce("+", .)
    
    results_list[[j]] <- Categorical_Classifier_Verification(actual = actual_class,
                                                             probabilities = predictions_table,
                                                             labels = labels,
                                                             save = FALSE,
                                                             open = FALSE,
                                                             type_info = "")[[2]]
    weights_list[[j]] <- weights_vector
    utils::setTxtProgressBar(pb, j)}
  
  base::cat("\n")
  
  results_list <- base::do.call(base::rbind, results_list) %>%
    dplyr::mutate(class = NULL,
                  type = NULL) %>%
    dplyr::rename_all(funs(stringr::str_to_title(.)))
  
  weights_table <- base::do.call(base::rbind, weights_list) %>%
    tibble::as_tibble() %>%
    magrittr::set_colnames(base::paste("Model", models_vector, sep = "_"))
  
  # Arrange according to selected metric:
  if(ascending == TRUE){
    dplyr::bind_cols(results_list, weights_table) %>%
      dplyr::arrange(!!key_metric) -> final_results
  } else {
    dplyr::bind_cols(results_list, weights_table) %>%
      dplyr::arrange(dplyr::desc(!!key_metric)) -> final_results
  }
  
  # Return results:
  if (summary_type == "mean"){
    base::return(base::list(all_results = final_results,
                            top_results = final_results %>% utils::head(top),
                            optimized_weights = final_results %>%
                              dplyr::select(dplyr::starts_with("Model")) %>%
                              utils::head(top) %>%
                              dplyr::summarise_all(base::mean)))}
  
  if (summary_type == "median"){
    base::return(base::list(all_results = final_results,
                            top_results = final_results %>% utils::head(top),
                            optimized_weights = final_results %>%
                              dplyr::select(dplyr::starts_with("Model")) %>%
                              utils::head(top) %>%
                              dplyr::summarise_all(stats::median)))}
}

# ------------------------------------------------------------------------------
# Build Ensemble model for categorical classification problem:
Categorical_Ensemble_Model <- function(models_vector,
                                       labels,
                                       optimization_dataset, # "train", "validation", "train+validation"
                                       save_option = FALSE,
                                       weights = 25,
                                       key_metric = "Accuracy",
                                       key_metric_as_string = TRUE,
                                       ascending = FALSE,
                                       top = 10,
                                       seed = 42,
                                       summary_type = "mean",
                                       cwd = models_store_dir,
                                       n = 3){
  
  train_pattern <- "train_categorical_probabilities"
  validation_pattern <- "validation_categorical_probabilities"
  test_dir <- "test_categorical_probabilities"
  
  dataset_types <- base::c("train_dataset", "validation_dataset", "test_dataset")
  
  all_predictions <- base::list()
  for (i in base::seq_along(models_vector)){
    model <- base::list()
    model[[1]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = train_pattern, full.names = TRUE))
    model[[2]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = validation_pattern, full.names = TRUE))
    model[[3]] <- readr::read_csv2(base::list.files(base::paste(base::getwd(), models_vector[i], model_type, sep = "/"), pattern = test_dir, full.names = TRUE))
    all_predictions[[i]] <- model}
  
  Display_List_Structure(all_predictions, n = n)
  
  # ------------------------------------------------------------------------------
  # Predictions:
  # Train:
  train_predictions <- base::list()
  for (i in 1:base::length(all_predictions)){train_predictions[[i]] <- all_predictions[[i]][[1]] %>%
    dplyr::select(dplyr::starts_with("V"))}
  
  # Validation:
  validation_predictions <- base::list()
  for (i in 1:base::length(all_predictions)){validation_predictions[[i]] <- all_predictions[[i]][[2]] %>%
    dplyr::select(dplyr::starts_with("V"))}
  
  # Test:
  test_predictions <- base::list()
  for (i in 1:base::length(all_predictions)){test_predictions[[i]] <- all_predictions[[i]][[3]] %>%
    dplyr::select(dplyr::starts_with("V"))}
  
  # ------------------------------------------------------------------------------
  # Actual:
  train_actual <- all_predictions[[1]][[1]]$actual_class
  validation_actual <- all_predictions[[1]][[2]]$actual_class
  test_actual <- all_predictions[[1]][[3]]$actual_class
  
  # ------------------------------------------------------------------------------
  # Change working directory to save all files in Ensemble_Model Categorical Folder:
  base::setwd(cwd)
  
  # ------------------------------------------------------------------------------
  # Train results for single component models:
  train_component_results <- base::list()
  for (i in 1:base::length(models_vector)){
    train_component_results[[i]] <- Categorical_Classifier_Verification(actual = train_actual,
                                                                        probabilities = train_predictions[[i]],
                                                                        labels = labels,
                                                                        save = save_option,
                                                                        open = FALSE,
                                                                        type_info = base::paste(models_vector[i], dataset_types[1], sep = "_"))[[3]] %>%
      dplyr::mutate(Model = models_vector[i])}
  
  train_component_overall_results <- base::list()
  for (i in 1:base::length(train_component_results)){
    train_component_overall_results[[i]] <- train_component_results[[i]] %>%
      dplyr::filter(Class == "Overall") %>%
      dplyr::mutate(Type = NULL,
                    Class = NULL) %>%
      tidyr::pivot_longer(cols = base::c("Records", "Correct", "Incorrect", "Accuracy", "Inaccuracy", "Precision", "Recall", "F1"),
                          names_to = "Metric",
                          values_to = base::paste(models_vector[i], "Score", sep = "_")) %>%
      dplyr::mutate(Model = NULL)}
  
  train_component_overall_results_2 <- tibble::tibble(Metric = train_component_overall_results[[1]]$Metric)
  for (i in 1:base::length(models_vector)){train_component_overall_results_2 <- dplyr::bind_cols(train_component_overall_results_2, train_component_overall_results[[i]][2])}
  
  # ------------------------------------------------------------------------------
  # Validation results for single component models:
  validation_component_results <- base::list()
  for (i in 1:base::length(models_vector)){
    validation_component_results[[i]] <- Categorical_Classifier_Verification(actual = validation_actual,
                                                                             probabilities = validation_predictions[[i]],
                                                                             labels = labels,
                                                                             save = save_option,
                                                                             open = FALSE,
                                                                             type_info = base::paste(models_vector[i], dataset_types[2], sep = "_"))[[3]] %>%
      dplyr::mutate(Model = models_vector[i])}
  
  validation_component_overall_results <- base::list()
  for (i in 1:base::length(validation_component_results)){
    validation_component_overall_results[[i]] <- validation_component_results[[i]] %>%
      dplyr::filter(Class == "Overall") %>%
      dplyr::mutate(Type = NULL,
                    Class = NULL) %>%
      tidyr::pivot_longer(cols = base::c("Records", "Correct", "Incorrect", "Accuracy", "Inaccuracy", "Precision", "Recall", "F1"),
                          names_to = "Metric",
                          values_to = base::paste(models_vector[i], "Score", sep = "_")) %>%
      dplyr::mutate(Model = NULL)}
  
  validation_component_overall_results_2 <- tibble::tibble(Metric = validation_component_overall_results[[1]]$Metric)
  for (i in 1:base::length(models_vector)){validation_component_overall_results_2 <- dplyr::bind_cols(validation_component_overall_results_2, validation_component_overall_results[[i]][2])}
  
  # ------------------------------------------------------------------------------
  # Test results for single component models:
  test_component_results <- base::list()
  for (i in 1:base::length(models_vector)){
    test_component_results[[i]] <- Categorical_Classifier_Verification(actual = test_actual,
                                                                       probabilities = test_predictions[[i]],
                                                                       labels = labels,
                                                                       save = save_option,
                                                                       open = FALSE,
                                                                       type_info = base::paste(models_vector[i], dataset_types[3], sep = "_"))[[3]] %>%
      dplyr::mutate(Model = models_vector[i])}
  
  test_component_overall_results <- base::list()
  for (i in 1:base::length(test_component_results)){
    test_component_overall_results[[i]] <- test_component_results[[i]] %>%
      dplyr::filter(Class == "Overall") %>%
      dplyr::mutate(Type = NULL,
                    Class = NULL) %>%
      tidyr::pivot_longer(cols = base::c("Records", "Correct", "Incorrect", "Accuracy", "Inaccuracy", "Precision", "Recall", "F1"),
                          names_to = "Metric",
                          values_to = base::paste(models_vector[i], "Score", sep = "_")) %>%
      dplyr::mutate(Model = NULL)}
  
  test_component_overall_results_2 <- tibble::tibble(Metric = test_component_overall_results[[1]]$Metric)
  for (i in 1:base::length(models_vector)){test_component_overall_results_2 <- dplyr::bind_cols(test_component_overall_results_2, test_component_overall_results[[i]][2])}
  
  # ------------------------------------------------------------------------------
  # Optimization dataset:
  if (optimization_dataset == "train"){
    actual_optimization = train_actual
    predictions_optimization = train_predictions}
  
  if (optimization_dataset == "validation"){
    actual_optimization = validation_actual
    predictions_optimization = validation_predictions}
  
  if (optimization_dataset == "train+validation"){
    actual_optimization = base::c(train_actual, validation_actual)
    predictions_optimization = base::list()
    for(i in 1:base::length(models_vector)){
      predictions_optimization[[i]] <- dplyr::bind_rows(train_predictions[[i]], validation_predictions[[i]])
    }}
  
  # ------------------------------------------------------------------------------
  # Optimize weights in ensemble categorical model on selected data using simulation approach:
  ensemble_optimization <- Optimize_Categorical_Ensemble_Cutoff_Model(actual_class = actual_optimization,
                                                                      predictions = predictions_optimization,
                                                                      weights = weights,
                                                                      labels = labels,
                                                                      models_vector = models_vector,
                                                                      key_metric_as_string = key_metric_as_string,
                                                                      key_metric = key_metric,
                                                                      seed = seed,
                                                                      ascending = ascending,
                                                                      summary_type = summary_type)
  
  ensemble_optimization_weights <- ensemble_optimization$optimized_weights %>%
    tidyr::pivot_longer(cols = dplyr::everything()) %>%
    dplyr::select(value) %>%
    dplyr::pull()
  
  # ------------------------------------------------------------------------------
  # Ensemble model predictions:
  train_result <- Multiply_List_Values(list = train_predictions, weights_vector = ensemble_optimization_weights) %>%
    base::Reduce("+", .)
  
  validation_result <- Multiply_List_Values(list = validation_predictions, weights_vector = ensemble_optimization_weights) %>%
    base::Reduce("+", .)
  
  test_result <- Multiply_List_Values(list = test_predictions, weights_vector = ensemble_optimization_weights) %>%
    base::Reduce("+", .)
  
  ensemble_model_predictions <- base::list(train_result,
                                           validation_result,
                                           test_result)
  
  # ------------------------------------------------------------------------------
  # Train results for single component models:
  train_dataset_ensemble_model_results <- Categorical_Classifier_Verification(actual = train_actual,
                                                                              probabilities = ensemble_model_predictions[[1]],
                                                                              labels = labels,
                                                                              type_info = base::paste("Ensemble_Model", dataset_types[1], sep = "_"),
                                                                              save = save_option,
                                                                              open = FALSE)
  
  train_dataset_ensemble_model_results_2 <- train_dataset_ensemble_model_results[[3]] %>%
    dplyr::filter(Class == "Overall") %>%
    dplyr::mutate(Type = NULL,
                  Class = NULL) %>%
    tidyr::pivot_longer(cols = base::c("Records", "Correct", "Incorrect", "Accuracy", "Inaccuracy", "Precision", "Recall", "F1"),
                        names_to = "Metric",
                        values_to = base::paste("Ensemble_Model", "Score", sep = "_"))
  
  if (save_option == TRUE){
    datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
    train_component_overall_results_2 %>%
      dplyr::left_join(train_dataset_ensemble_model_results_2, by = "Metric") %>%
      readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Ensemble_Model_train_dataset_results_summary_comparison.csv", sep = "_"), sep = "/"))
    base::Sys.sleep(time = 1)}
  
  # ------------------------------------------------------------------------------
  # Validation results for single component models:
  validation_dataset_ensemble_model_results <- Categorical_Classifier_Verification(actual = validation_actual,
                                                                                   probabilities = ensemble_model_predictions[[2]],
                                                                                   labels = labels,
                                                                                   type_info = base::paste("Ensemble_Model", dataset_types[2], sep = "_"),
                                                                                   save = save_option,
                                                                                   open = FALSE)
  
  validation_dataset_ensemble_model_results_2 <- validation_dataset_ensemble_model_results[[3]] %>%
    dplyr::filter(Class == "Overall") %>%
    dplyr::mutate(Type = NULL,
                  Class = NULL) %>%
    tidyr::pivot_longer(cols = base::c("Records", "Correct", "Incorrect", "Accuracy", "Inaccuracy", "Precision", "Recall", "F1"),
                        names_to = "Metric",
                        values_to = base::paste("Ensemble_Model", "Score", sep = "_"))
  
  if (save_option == TRUE){
    datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
    validation_component_overall_results_2 %>%
      dplyr::left_join(validation_dataset_ensemble_model_results_2, by = "Metric") %>%
      readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Ensemble_Model_validation_dataset_results_summary_comparison.csv", sep = "_"), sep = "/"))
    base::Sys.sleep(time = 1)}
  
  # ------------------------------------------------------------------------------
  # Test results for single component models:
  test_dataset_ensemble_model_results <- Categorical_Classifier_Verification(actual = test_actual,
                                                                             probabilities = ensemble_model_predictions[[3]],
                                                                             labels = labels,
                                                                             type_info = base::paste("Ensemble_Model", dataset_types[3], sep = "_"),
                                                                             save = save_option,
                                                                             open = FALSE)
  
  test_dataset_ensemble_model_results_2 <- test_dataset_ensemble_model_results[[3]] %>%
    dplyr::filter(Class == "Overall") %>%
    dplyr::mutate(Type = NULL,
                  Class = NULL) %>%
    tidyr::pivot_longer(cols = base::c("Records", "Correct", "Incorrect", "Accuracy", "Inaccuracy", "Precision", "Recall", "F1"),
                        names_to = "Metric",
                        values_to = base::paste("Ensemble_Model", "Score", sep = "_"))
  
  if (save_option == TRUE){
    datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
    test_component_overall_results_2 %>%
      dplyr::left_join(test_dataset_ensemble_model_results_2, by = "Metric") %>%
      readr::write_csv2(path = base::paste(models_store_dir, base::paste(datetime, "Ensemble_Model_test_dataset_results_summary_comparison.csv", sep = "_"), sep = "/"))
    base::Sys.sleep(time = 1)}
  
  # ------------------------------------------------------------------------------
  # Set the initial working directory:
  base::setwd(".."); base::setwd("..")
  
  # ------------------------------------------------------------------------------
  # Final summary of ensemble model results:
  ensemble_model_summary <- base::list(train_dataset_results = train_component_overall_results_2 %>%
                                         dplyr::left_join(train_dataset_ensemble_model_results_2, by = "Metric"),
                                       validation_dataset_results = validation_component_overall_results_2 %>%
                                         dplyr::left_join(validation_dataset_ensemble_model_results_2, by = "Metric"),
                                       test_dataset_results = test_component_overall_results_2 %>%
                                         dplyr::left_join(test_dataset_ensemble_model_results_2, by = "Metric"))
  
  ensemble_model_summary %>%
    base::lapply(., knitr::kable)
  
  base::return(base::list(all_optimization_combinations = ensemble_optimization[[1]],
                          top_optimization_combinations = ensemble_optimization[[2]],
                          optimal_weights = ensemble_optimization_weights,
                          train_dataset_results = train_component_overall_results_2 %>%
                            dplyr::left_join(train_dataset_ensemble_model_results_2, by = "Metric"),
                          validation_dataset_results = validation_component_overall_results_2 %>%
                            dplyr::left_join(validation_dataset_ensemble_model_results_2, by = "Metric"),
                          test_dataset_results = test_component_overall_results_2 %>%
                            dplyr::left_join(test_dataset_ensemble_model_results_2, by = "Metric"),
                          train_models_predictions = train_predictions,
                          train_ensemble_model_prediction = train_result %>%
                            tibble::as_tibble(),
                          train_actual_class = train_actual,
                          validation_models_predictions = validation_predictions,
                          validation_ensemble_model_prediction = validation_result %>%
                            tibble::as_tibble(),
                          validation_actual_class = validation_actual,
                          test_models_predictions = test_predictions,
                          test_ensemble_model_prediction = test_result %>%
                            tibble::as_tibble(),
                          test_actual_class = test_actual))
}




