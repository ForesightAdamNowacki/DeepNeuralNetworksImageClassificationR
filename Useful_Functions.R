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
  prediction <- model %>% keras::predict_generator(generator = generator, steps = 1)
  
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
                                                       cutoff = 0.5,
                                                       save_summary_files = TRUE,
                                                       save_correct_images = TRUE,
                                                       save_incorrect_images = TRUE){
  
  base::print(base::paste("Current working directory:", cwd))
  base::setwd(cwd)
  
  datetime <- stringr::str_replace_all(base::Sys.time(), ":", "-")
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
  
  if (base::isTRUE(save_summary_files)){
    readr::write_csv2(summary_data, base::paste(datetime, dataset_label, "all_classification.csv", sep = "_"))
    readr::write_csv2(summary_data_correct, base::paste(datetime, dataset_label, "correct_classification.csv", sep = "_"))
    readr::write_csv2(summary_data_incorrect, base::paste(datetime, dataset_label, "incorrect_classification.csv", sep = "_"))
    base::print(base::paste("File created:", base::paste(datetime, dataset_label, "all_classification.csv", sep = "_")))
    base::print(base::paste("File created:", base::paste(datetime, dataset_label, "correct_classification.csv", sep = "_")))
    base::print(base::paste("File created:", base::paste(datetime, dataset_label, "incorrect_classification.csv", sep = "_")))}
  
  # correct:
  if (base::isTRUE(save_correct_images)){
    correct_classification_folder <- base::paste(datetime, dataset_label, "correct_classification", sep = "_")
    base::unlink(correct_classification_folder, recursive = TRUE)
    base::dir.create(correct_classification_folder, recursive  = TRUE, showWarnings = FALSE)
    base::print(base::paste("Folder created:", correct_classification_folder))
    
    correct_classification_dir <- base::paste(base::getwd(), correct_classification_folder, sep = "/")
    base::file.copy(from = summary_data_correct$files,
                    to = base::paste(correct_classification_dir, base::basename(summary_data_correct$files), sep = "/"))}
  
  # incorrect:
  if (base::isTRUE(save_incorrect_images)){
    incorrect_classification_folder <- base::paste(datetime, dataset_label, "incorrect_classification", sep = "_")
    base::unlink(incorrect_classification_folder, recursive = TRUE)
    base::dir.create(incorrect_classification_folder, recursive  = TRUE, showWarnings = FALSE)
    base::print(base::paste("Folder created:", incorrect_classification_folder))
    
    incorrect_classification_dir <- base::paste(base::getwd(), incorrect_classification_folder, sep = "/")
    base::file.copy(from = summary_data_incorrect$files,
                    to = base::paste(incorrect_classification_dir, base::basename(summary_data_incorrect$files), sep = "/"))}
  
  if (base::isTRUE(save_summary_files)){
    base::invisible(base::list(all_files = summary_data,
                               correct_classification = summary_data_correct,
                               incorrect_classification = summary_data_incorrect))}
}

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
# Final version
# library(tidyverse)
# cuts <- 50 # i
# weights <- 50 # j
# 
# # Train:
# pred_1 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Inception_V3_train_binary_probabilities.csv") %>%
#   dplyr::select(V2) %>% 
#   dplyr::pull()
# pred_2 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/ResNet50_train_binary_probabilities.csv") %>%
#   dplyr::select(V2) %>% 
#   dplyr::pull()
# pred_3 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Xception_train_binary_probabilities.csv") %>%
#   dplyr::select(V2) %>% 
#   dplyr::pull()
# predictions <- base::cbind(pred_1, pred_2, pred_3); predictions
# 
# files <- count_files(path = "D:\\GitHub\\Datasets\\Cats_And_Dogs\\train")
# actual_class <- base::rep(base::c(0, 1), times = files$category_obs)
# actual_class <- base::factor(actual_class, levels = base::c(0, 1), labels = base::c(0, 1))
# 
# 
# 
# # Validation:
# pred_1 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Inception_V3_validation_binary_probabilities.csv") %>%
#   dplyr::select(V2) %>% 
#   dplyr::pull()
# pred_2 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/ResNet50_validation_binary_probabilities.csv") %>%
#   dplyr::select(V2) %>% 
#   dplyr::pull()
# pred_3 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Xception_validation_binary_probabilities.csv") %>%
#   dplyr::select(V2) %>% 
#   dplyr::pull()
# predictions <- base::cbind(pred_1, pred_2, pred_3); predictions
# 
# files <- count_files(path = "D:\\GitHub\\Datasets\\Cats_And_Dogs\\validation")
# actual_class <- base::rep(base::c(0, 1), times = files$category_obs)
# actual_class <- base::factor(actual_class, levels = base::c(0, 1), labels = base::c(0, 1))
# 
# # Optimization
# 
# abc <- Optimize_Ensemble_Cutoff_Model(actual_class = actual_class,
#                                predictions = predictions,
#                                cuts = 25,
#                                weights = 25,
#                                key_metric = ACC,
#                                )
# 




Optimize_Ensemble_Cutoff_Model <- function(actual_class,
                                           predictions,
                                           cuts,
                                           weights,
                                           key_metric = ACC,
                                           ascending = FALSE,
                                           seed = 42,
                                           top = 10,
                                           TN_cost = 0,
                                           FP_cost = 1,
                                           FN_cost = 1,
                                           TP_cost = 0){
  
  # Libraries:
  if (!require(tidyverse)){utils::install.packages('tidyverse'); require('tidyverse')}  
  
  # Key metric:
  key_metric <- dplyr::enquo(key_metric) 
  key_metric_name <- dplyr::quo_name(key_metric)
  
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
    utils::setTxtProgressBar(pb,i, title = "tytul", label = "abc")}
  
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
  base::return(base::list(all_results = final_results,
                          top_results = final_results %>% utils::head(top),
                          optimized_cutoff = final_results %>%
                            utils::head(top) %>%
                            dplyr::select(CUTOFF) %>%
                            dplyr::summarise_all(mean),
                          optimized_weights = final_results %>%
                            utils::head(top) %>%
                            dplyr::select(dplyr::starts_with("model_")) %>%
                            dplyr::summarise_all(mean)))}

# ------------------------------------------------------------------------------
# Plot predictions distribution in division to binary classes:
Visualize_Predictions_Distribution <- function(actual,
                                               predicted,
                                               bins = 10,
                                               text_size = 7,
                                               title_size = 9){
  
  tibble::tibble(actual = base::factor(actual),
                 predicted = predicted) %>%
    dplyr::mutate(cut = ggplot2::cut_interval(predicted, length = 1/bins)) %>%
    dplyr::group_by(cut, actual) %>%
    dplyr::summarise(n = dplyr::n()) %>%
    dplyr::ungroup() %>%
    tidyr::complete(cut, actual, fill = base::list(n = 0)) %>%
    ggplot2::ggplot(data = ., mapping = ggplot2::aes(x = cut, y = n, label = n)) +
    ggplot2::geom_bar(stat = "identity", col = "black") +
    ggplot2::geom_label() +
    ggplot2::facet_grid(actual~.) +
    ggplot2::labs(x = "Prediction value",
                  y = "Count",
                  title = "Predictions distribution per binary class") +
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
                   strip.text = element_text(size = text_size, face = "bold"))}


# ------------------------------------------------------------------------------
# BINARY MODEL EVALUATION
# Function to verify the predictive and classification capabilities of the binary model.
# ------------------------------------------------------------------------------
# Environment:
base::library(reticulate)
reticulate::use_condaenv(condaenv = "GPU_ML_2", required = TRUE)
base::library(Metrics)
base::library(tidyverse)
base::library(tibble)
base::library(knitr)
base::library(gridExtra)
base::library(gt)

# ------------------------------------------------------------------------------
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
               filename = stringr::str_replace_all(base::paste0(sys_time, " Binary model evaluation metrics ", type_info, ".png"), ":", "-"),
               vwidth = 900,
               vheight = 1600,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Binary model evaluation metrics ", type_info, ".png"), ":", "-"))
    }
  }
  
  gt_table %>% base::print(.)
  
  base::invisible(base::list("Confusion_Matrix_Explanation" = result_1,
                             "Confusion_Matrix_Result" = result_2,
                             "Assessment_of_Classifier_Effectiveness" = result_3))
}

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
  
  key_metric <- dplyr::enquo(key_metric) 
  key_metric_name <- dplyr::quo_name(key_metric)
  
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
               filename = stringr::str_replace_all(base::paste0(sys_time, " Binary model cut-off value optimization ", type_info, ".png"), ":", "-"),
               vwidth = 1600,
               vheight = 900,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Binary model cut-off value optimization ", type_info, ".png"), ":", "-"))
    }
  }
  
  gt_table %>% base::print(.)
  base::invisible(df)
}

# ------------------------------------------------------------------------------
# Categorical model evaluation:
Categorical_Model_Evaluation <- function(actual,
                                         probabilities,
                                         labels,
                                         type_info = "",
                                         cutoff = 0.5,
                                         FN_cost = 1,
                                         FP_cost = 1,
                                         TN_cost = 0,
                                         TP_cost = 0,
                                         save = FALSE,
                                         open = TRUE){
  
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
    dplyr::rename(class = actual) -> stats_2
  
  stats_2 %>%
    dplyr::mutate(class = "overall") %>%
    dplyr::group_by(class) %>%
    dplyr::summarise(records = base::sum(records),
                     correct = base::sum(correct),
                     incorrect = records - correct,
                     accuracy = correct/records,
                     inaccuracy = incorrect/records,
                     type = "All categories") -> stats_3
  
  dplyr::bind_rows(stats_2, stats_3) %>%
    dplyr::rename(Class = class,
                  Records = records,
                  Correct = correct,
                  Incorrect = incorrect, 
                  Accuracy = accuracy,
                  Inaccuracy = inaccuracy,
                  Precision = precision,
                  Recall = recall,
                  F1 = f1) -> stats_4
  
  probabilities %>%
    base::as.data.frame() %>%
    dplyr::mutate(id = dplyr::row_number()) %>%
    tidyr::pivot_longer(cols = dplyr::starts_with("V"), names_to = "predicted_class", values_to = "probability") %>%
    dplyr::mutate(predicted_class = base::as.numeric(stringr::str_sub(predicted_class, 2, -1))) %>%
    dplyr::left_join(tibble::tibble(id = 1:base::length(actual),
                                    actual = actual), by = base::c("id")) %>%
    dplyr::mutate(ground_true = base::ifelse(predicted_class == actual, 1, 0)) %>%
    dplyr::select(ground_true, probability) -> stats_5
  
  stats_4 %>%
    dplyr::ungroup() %>%
    gt::gt(rowname_col = "Class", groupname_col = "type") %>%
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
  
  Binary_Classifier_Verification_2 <- function(actual,
                                               predicted,
                                               type_info = "",
                                               cutoff = 0.5,
                                               FN_cost = 1,
                                               FP_cost = 1,
                                               TN_cost = 0,
                                               TP_cost = 0,
                                               save = FALSE,
                                               open = TRUE){
    
    # Confusion matrix explanation:
    result_1 <- tibble::tibble("Confusion Matrix" = base::c("Actual Negative (0)", "Actual Positive (1)"),
                               "Predicted Negative (0)" = base::c("True Negative (TN)", "False Negative (FN)"),
                               "Predicted Positive (1)" = base::c("False Positive (FP)", "True Positive (TP)"))
    
    probability <- predicted
    if(base::length(base::unique(predicted)) > 2){predicted <- base::ifelse(predicted < cutoff, 0, 1)}
    
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
    
    result_3 %>%
      dplyr::mutate(Group = base::ifelse(Metric %in% base::c("Number of Observations", "True Negative",
                                                             "False Positive", "False Negative",
                                                             "True Positive", "Condition Positive",
                                                             "Condition Negative"), 
                                         "Confusion Matrix Result", "Assessment of Classifier Effectiveness")) %>%
      gt::gt(rowname_col = "ID", groupname_col = "Group") %>%
      gt::tab_header(title = gt::md(base::paste("Model's evaluation metrics", sys_time)),
                     subtitle = gt::md("Binary-Categorical classification model")) %>%
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
    
    base::print(gt_table)
    
    if (save == TRUE){
      gt::gtsave(data = gt_table,
                 filename = stringr::str_replace_all(base::paste0(sys_time, " Binary-categorical model evaluation metrics ", type_info, ".png"), ":", "-"),
                 vwidth = 900,
                 vheight = 1600,
                 expand = 5)
      if (open == TRUE){
        rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Binary-categorical model evaluation metrics ", type_info, ".png"), ":", "-"))
      }
    }
    
    base::invisible(base::list("Confusion_Matrix_Explanation" = result_1,
                               "Confusion_Matrix_Result" = result_2,
                               "Assessment_of_Classifier_Effectiveness" = result_3))
  }
  
  base::print(gt_table)
  
  if (save == TRUE){
    gt::gtsave(data = gt_table,
               filename = stringr::str_replace_all(base::paste0(sys_time, " Categorical model evaluation metrics ", type_info, ".png"), ":", "-"),
               vwidth = 900,
               vheight = 600,
               expand = 5)
    if (open == TRUE){
      rstudioapi::viewer(stringr::str_replace_all(base::paste0(sys_time, " Categorical model evaluation metrics ", type_info, ".png"), ":", "-"))
    }
  }
  
  binary_results <- Binary_Classifier_Verification_2(actual = stats_5$ground_true,
                                                     predicted = stats_5$probability,
                                                     type_info = type_info,
                                                     save = save,
                                                     open = open,
                                                     cutoff = cutoff,
                                                     FN_cost = FN_cost,
                                                     FP_cost = FP_cost,
                                                     TN_cost = TN_cost,
                                                     TP_cost = TP_cost)
  
  base::invisible(base::list("Categorical classification" = stats_4,
                             "Binary classification" = binary_results))
}

