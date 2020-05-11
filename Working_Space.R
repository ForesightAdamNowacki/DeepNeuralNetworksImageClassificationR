# Create k folds

data_dir <- "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\data"
data_to <- "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds"
folds <- 5
seed = 42

Create_KFolds_Directories(data_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\data",
                          target_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds",
                          seed = 42)

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
  
  readr::write_csv(files, "Cross_Validation_Splits.csv")
}

Create_KFolds_Directories(data_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\data",
                          target_dir = "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds",
                          seed = 42)






data_dir <- "D:\\GitHub\\Datasets\\Cats_And_Dogs_Small\\K-folds"

setwd("..")
getwd()

#folds <- base::list.files(data_dir)[base::grepl("fold_", list.files(data_dir))]
#steps <- base::paste("step", 1:base::length(folds), sep = "_")
#classes <- base::c("cats", "dogs") # from above function

i <- 1
j <- train_dir
j <- validation_dir
k <- 1

for (i in base::seq_along(steps)){
  
  step_dir <- base::paste(data_dir, steps[i], sep = "/")
  train_dir <- base::paste(step_dir, "train", sep = "/")
  validation_dir <- base::paste(step_dir, "validation", sep = "/")
  base::dir.create(path = step_dir, showWarnings = FALSE, recursive = TRUE)
  base::dir.create(path = train_dir, showWarnings = FALSE, recursive = TRUE)
  base::dir.create(path = validation_dir, showWarnings = FALSE, recursive = TRUE)
  
  
}

classes
step
folds







