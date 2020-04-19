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
library(tidyverse)
cuts <- 50 # i
weights <- 50 # j

# Train:
pred_1 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Inception_V3_train_binary_probabilities.csv") %>%
  dplyr::select(V2) %>% 
  dplyr::pull()
pred_2 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/ResNet50_train_binary_probabilities.csv") %>%
  dplyr::select(V2) %>% 
  dplyr::pull()
pred_3 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Xception_train_binary_probabilities.csv") %>%
  dplyr::select(V2) %>% 
  dplyr::pull()
predictions <- base::cbind(pred_1, pred_2, pred_3); predictions

files <- count_files(path = "D:\\GitHub\\Datasets\\Cats_And_Dogs\\train")
actual_class <- base::rep(base::c(0, 1), times = files$category_obs)
actual_class <- base::factor(actual_class, levels = base::c(0, 1), labels = base::c(0, 1))



# Validation:
pred_1 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Inception_V3_validation_binary_probabilities.csv") %>%
  dplyr::select(V2) %>% 
  dplyr::pull()
pred_2 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/ResNet50_validation_binary_probabilities.csv") %>%
  dplyr::select(V2) %>% 
  dplyr::pull()
pred_3 <- readr::read_csv("D:/GitHub/DeepNeuralNetworksRepoR_Models_Store/Xception_validation_binary_probabilities.csv") %>%
  dplyr::select(V2) %>% 
  dplyr::pull()
predictions <- base::cbind(pred_1, pred_2, pred_3); predictions

files <- count_files(path = "D:\\GitHub\\Datasets\\Cats_And_Dogs\\validation")
actual_class <- base::rep(base::c(0, 1), times = files$category_obs)
actual_class <- base::factor(actual_class, levels = base::c(0, 1), labels = base::c(0, 1))

# Optimization
cuts_ <- stats::runif(n = cuts, min = 0, max = 1)
cuts_ <- base::sort(x = cuts_, decreasing = FALSE)

weights_ <- base::matrix(data = stats::runif(base::ncol(predictions) * weights, min = 0, max = 1),
                         nrow = weights,
                         ncol = base::ncol(predictions))

TN_cost <- 0
FP_cost <- 1
FN_cost <- 1
TP_cost <- 0

results <- base::list()
pb = txtProgressBar(min = 0, max = cuts, initial = 0, style = 3, title = "tytul", label = "abc") 

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
    df$COST[j] <- TN_cost * df$TN[j] + FP_cost * df$FP[j] + FN_cost * df$FN[j] + TP_cost * df$TP[j]
    
    
  }
  results[[i]] <- df
  utils::setTxtProgressBar(pb,i, title = "tytul", label = "abc")
}

final_results <- base::do.call(bind_rows, results) %>%
  tidyr::separate(col = WEIGHTS, sep = ", ", into = base::paste("model", 1:base::ncol(predictions), sep = "_"), convert = TRUE)
final_results %>%
  dplyr::arrange(dplyr::desc(ACC)) 


require(svMisc)
install.packages("svMisc")


svMisc::

pred_1_class <- ifelse(pred_1 < 0.5, 0, 1)
pred_1_class <- table(actual_class, pred_1_class) 
sum(diag(pred_1_class))/sum(pred_1_class)
sum(pred_1_class) - sum(diag(pred_1_class))








# -------------------------
library(tidyverse)
n <- 100 # observations
cuts <- 50 # i
weights <- 50 # j
pred_1 <- runif(n, min = 0, max = 1)
pred_2 <- runif(n, min = 0, max = 1)
pred_3 <- runif(n, min = 0, max = 1)
actual_class <- base::sample(0:1, n, replace = TRUE)
actual_class <- base::factor(actual_class, levels = base::c(0, 1), labels = base::c(0, 1))
predictions <- base::round(base::cbind(pred_1, pred_2, pred_3), 4)

cuts_ <- stats::runif(n = cuts, min = 0, max = 1)
cuts_ <- base::sort(x = cuts_, decreasing = FALSE)

weights_ <- base::matrix(data = stats::runif(base::ncol(predictions) * weights, min = 0, max = 1),
                         nrow = weights,
                         ncol = base::ncol(predictions))

TN_cost <- 0
FP_cost <- 1
FN_cost <- 1
TP_cost <- 0

i <- 1
j <- 1

results <- base::list()
for (i in 1:base::length(cuts_)){
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
    df$COST[j] <- TN_cost * df$TN[j] + FP_cost * df$FP[j] + FN_cost * df$FN[j] + TP_cost * df$TP[j]
    
    
  }
  results[[i]] <- df
  base::print(i/cuts)
  
}

final_results <- base::do.call(bind_rows, results) %>%
  tidyr::separate(col = WEIGHTS, sep = ", ", into = base::paste("model", 1:base::ncol(predictions), sep = "_"), convert = TRUE)
final_results %>%
  dplyr::select(CUTOFF, dplyr::starts_with("model_"), ACC) %>%
  tidyr::pivot_longer(cols = dplyr::starts_with("model_"), names_to = "MODEL", values_to = "WEIGHT") %>%
  ggplot2::ggplot(data = ., mapping = aes(x = CUTOFF, y = WEIGHT, color = ACC)) +
  ggplot2::facet_wrap(.~MODEL) +
  ggplot2::geom_point() +
  ggplot2::scale_color_gradient(low = "white", high = "black")
  ggplot2::scale_color_continuous()

  
testit <- function(x = sort(runif(20)), ...)
  {
    pb <- txtProgressBar(...)
    for(i in c(0, x, 1)) {Sys.sleep(0.5); setTxtProgressBar(pb, i)}
    Sys.sleep(1)
    close(pb)
  }
testit()
testit(runif(10))


