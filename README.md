![alt text](Images//MD_Head.png?raw=true)
----------
# DeepNeuralNetworksImageClassificationR

Deep Neural Networks architectures based on convolutional neural networks using Keras and TensorFlow-GPU in R programming language. The repository is devoted to binary and multi-class image classification.

----------
# Scripts description:

The scripts collected on GitHub are divided into 3 main sections depending on the purpose and the characteristics of the main task: 
* **binary classification**, 
* **multi-class classification**,
* **auxiliary files and useful functions**.

----------
# Binary classification:
Files intended for implementation and optimization of deep neural networks for binary classification contain "Binary_Classification" in the script name. Below is attached brief description and purpose of each script:
* **Binary_Classification_CNN.R** - implementation of own deep neural network architecture for binary classification of images from the scratch,
* **Binary_Classification_CNN_Hyperparameters_Optimization_1.R** - model's hyperparameters optimization for binary image classification (requires implemented model in Binary_Classification_CNN_Hyperparameters_Optimization_2.R script),
* **Binary_Classification_CNN_Hyperparameters_Optimization_2.R** - model's architecture for hyperparameters optimization for binary image classification,
* **Binary_Classification_Cross_Validation.R** - implementation of deep neural network architecture for binary image classification from the scratch with cross-validation method,
* **Binary_Classification_DenseNet121.R** - implementation of the DenseNet121 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_DenseNet169.R** - implementation of the DenseNet169 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_DenseNet201.R** - implementation of the DenseNet201 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_Ensemble_Model.R** - implementation of the Meta-Classifier model for binary image classification using previously trained component architectures with additional simulation (random grid search) optimization of the hyperparameters of weights / contribution of partial models and the probability cutoff point.
* **Binary_Classification_Inception_ResNet_V2.R** - implementation of the Inception ResNet V2 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_Inception_V3.R** - implementation of the Inception V3 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_MobileNet_V2.R** - implementation of the MobileNet V2 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_NASNetMobile_V2.R** - implementation of the NASNetMobile model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_ResNet50.R** - implementation of the ResNet50 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_VGG16_1st_Stage.R** - implementation of the VGG16 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization with partially frozen model's weights (1st step),
* **Binary_Classification_VGG16_2nd_Stage.R** - implementation of the VGG16 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization of model weights with frostbitten layers (2nd step),
* **Binary_Classification_VGG16_Pipe_Runner.R** - script sequentially compiling Binary_Classification_VGG16_1st_Stage.R and Binary_Classification_VGG16_2nd_Stage.R files,
* **Binary_Classification_VGG19_1st_Stage.R** - implementation of the VGG19 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization with partially frozen model's weights (1st step),
* **Binary_Classification_VGG19_2nd_Stage.R** - implementation of the VGG19 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization of model weights with frostbitten layers (2nd step),
* **Binary_Classification_VGG19_Pipe_Runner.R** - script sequentially compiling Binary_Classification_VGG19_1st_Stage.R and Binary_Classification_VGG19_2nd_Stage.R files,
* **Binary_Classification_Xception.R** - implementation of the Xception model for binary image classification using pretrained weights on the ImageNet dataset.

----------
# Categorical classification:
Files intended for implementation and optimization of deep neural networks for categorical classification contain "Categorical_Classification" in the script name. Below is attached brief description and purpose of each script:
* **Categorical_Classification_CNN.R** - implementation of own deep neural network architecture for categorical classification of images from the scratch,
* **Categorical_Classification_CNN_Hyperparameters_Optimization_1.R** - model's hyperparameters optimization for categorical image classification (requires implemented model in Categorical_Classification_CNN_Hyperparameters_Optimization_2.R script),
* **Categorical_Classification_CNN_Hyperparameters_Optimization_2.R** - model's architecture for hyperparameters optimization for categorical image classification,
* **Categorical_Classification_Cross_Validation.R** - implementation of deep neural network architecture for binary image classification from the scratch with cross-validation method,
* **Categorical_Classification_DenseNet121.R** - implementation of the DenseNet121 model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_DenseNet169.R** - implementation of the DenseNet169 model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_DenseNet201.R** - implementation of the DenseNet201 model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_Ensemble_Model.R** - implementation of the Meta-Classifier model for categorical image classification using previously trained component architectures with additional simulation (random grid search) optimization of the hyperparameters of weights / contribution of partial models.
* **Categorical_Classification_Inception_ResNet_V2.R** - implementation of the Inception ResNet V2 model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_Inception_V3.R** - implementation of the Inception V3 model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_MobileNet_V2.R** - implementation of the MobileNet V2 model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_NASNetMobile_V2.R** - implementation of the NASNetMobile model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_ResNet50.R** - implementation of the ResNet50 model for categorical image classification using pretrained weights on the ImageNet dataset,
* **Categorical_Classification_VGG16_1st_Stage.R** - implementation of the VGG16 model for categorical image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization with partially frozen model's weights (1st step),
* **Categorical_Classification_VGG16_2nd_Stage.R** - implementation of the VGG16 model for categorical image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization of model weights with frostbitten layers (2nd step),
* **Categorical_Classification_VGG16_Pipe_Runner.R** - script sequentially compiling Categorical_Classification_VGG16_1st_Stage.R and Categorical_Classification_VGG16_2nd_Stage.R files,
* **Categorical_Classification_VGG19_1st_Stage.R** - implementation of the VGG19 model for categorical image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization with partially frozen model's weights (1st step),
* **Categorical_Classification_VGG19_2nd_Stage.R** - implementation of the VGG19 model for categorical image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization of model weights with frostbitten layers (2nd step),
* **Categorical_Classification_VGG19_Pipe_Runner.R** - script sequentially compiling Categorical_Classification_VGG19_1st_Stage.R and Categorical_Classification_VGG19_2nd_Stage.R files,
* **Categorical_Classification_Xception.R** - implementation of the Xception model for categorical image classification using pretrained weights on the ImageNet dataset.

----------
# Auxiliary files and useful functions:
* **Data_Augmentation.R** - manual testing and checking all settings and combinations of hyperparameters available in the Keras library used during data augmentation,
* **Requirements.R** - list of packages used in this repository,
* **Useful_Functions.R** - collection of useful and accelerating functions for building, optimizing and testing models used in binary and multi-class classification files briefly described above,
* **Visualisation_Heatmaps.R** - heatmap visualisation script for optimized deep convolutional neural network model,
* **Visualisation_Convolutional_Activations.R** - visualisation of convolutional layers activations in optimized deep convolutional neural network model,
* **Visualization_Convolutional_Filters.R** - visualization of what type of patterns each filter detects in convolutional layers of deep convolutional neural network model,
* **Visualisation_Pooling_Activations.R** - visualisation of pooling layers activations in optimized deep convolutional neural network model,
* **Working_Space.R** - workspace for creating and testing new functionalities.

----------
# Setup Anaconda environments:
## 1. Environment for TensorFlow 2.0 (or higher):
1. **Install Anaconda Python 3.7 version:**
* Download from https://www.anaconda.com/distribution/
* Open Anaconda Prompt
2. **Create and activate new anaconda environment (e.g. GPU_ML_2):**
```
conda create -n GPU_ML_2 python=3.7
conda activate GPU_ML_2
```
3. **Install Python kernell:**
```
pip install ipykernel
python -m ipykernel install --user --name GPU_ML_2 --display-name "GPU_ML_2"
```
4. **Install Tensorflow-GPU:**
```
conda install tensorflow-gpu==2.0.0
```
5. **Install jupyter:**
```
conda install jupyter
```
6. **Install keras:**
```
pip install keras
```
7. **Install R:**
* Download R from https://cran.r-project.org/
8. **Install RStudio IDE:**
* Download RStudio from https://rstudio.com/
9. **Open RStudio and install Keras and Tensorflow with GPU support:**
```
utils::install.packages("reticulate")
base::library(reticulate)
reticulate::use_condaenv("GPU_ML_2", required = TRUE)
utils::install.packages("tensorflow")
utils::install.packages("keras")
base::library(tensorflow)
base::library(keras)
keras::install_keras(tensorflow = "gpu")
```

## 2. Environment for TensorFlow 1.15:
1. **Install Anaconda Python 3.7 version:**
* Download from https://www.anaconda.com/distribution/
* Open Anaconda Prompt
2. **Create and activate new anaconda environment (e.g. GPU_ML_1):**
```
conda create -n GPU_ML_1 python=3.7
conda activate GPU_ML_1
```
3. **Install Python kernell:**
```
pip install ipykernel
python -m ipykernel install --user --name GPU_ML_1 --display-name "GPU_ML_1"
```
4. **Install Tensorflow-GPU:**
```
conda install tensorflow-gpu==1.15
```
5. **Install jupyter:**
```
conda install jupyter
```
6. **Install keras:**
```
pip install keras
```
7. **Install R:**
* Download R from https://cran.r-project.org/
8. **Install RStudio IDE:**
* Download RStudio from https://rstudio.com/
9. **Open RStudio and install Keras and Tensorflow with GPU support:**
```
utils::install.packages("reticulate")
base::library(reticulate)
reticulate::use_condaenv("GPU_ML_1", required = TRUE)
utils::install.packages("tensorflow")
utils::install.packages("keras")
base::library(tensorflow)
base::library(keras)
keras::install_keras(tensorflow = "1.15-gpu")
```
----------