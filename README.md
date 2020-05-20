![alt text](Images//MD_Head.png?raw=true)
----------
# DeepNeuralNetworksRepoR

Deep Neural Networks architectures based on convolutional neural networks using Keras and TensorFlow-GPU in R. The repository is devoted to binary and multi-class image classification.

----------
# Scripts description:

The scripts collected on GitHub are divided into 3 main sections depending on the purpose and the characteristics of the task: 
* **binary classification**, 
* **multi-class classification**,
* **auxiliary files and functions**.

----------
# Binary classification:
Files intended for implementation and optimization of deep neural networks for binary classification contain "Binary_Classification" in the script name. Below is attached brief description and purpose of each script:
* **Binary_Classification_CNN.R** - implementation of own deep neural network architecture for binary classification of images from the scratch,
* **Binary_Classification_DenseNet121.R** - implementation of the DenseNet121 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_DenseNet169.R** - implementation of the DenseNet169 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_DenseNet201.R** - implementation of the DenseNet201 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_Ensemble_Model.R** - implementation of the Meta-Classifier model for binary classification using previously trained component architectures with additional simulation (random grid search) optimization of the hyperparameter of weights / contribution of partial models and the probability cutoff point.
* **Binary_Classification_Inception_ResNet_V2.R** - implementation of the Inception ResNet V2 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_Inception_V3.R** - implementation of the Inception V3 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_K-Fold_Cross_Validation.R** - implementation of own deep neural network architecture for binary image classification from the scratch with cross-validation method,
* **Binary_Classification_MobileNet_V2.R** - implementation of the MobileNet V2 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_NASNetMobile_V2.R** - implementation of the NASNetMobile model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_ResNet50.R** - implementation of the ResNet50 model for binary image classification using pretrained weights on the ImageNet dataset,
* **Binary_Classification_VGG16_1st_Stage.R** - implementation of the VGG16 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization with partially frozen model weights (1st step),
* **Binary_Classification_VGG16_2nd_Stage.R** - implementation of the VGG16 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization of model weights with frostbitten layers (2nd step),
* **Binary_Classification_VGG16_Pipe_Runner.R** - executable script sequentially compiling Binary_Classification_VGG16_1st_Stage.R and Binary_Classification_VGG16_2nd_Stage.R files,
* **Binary_Classification_VGG19_1st_Stage.R** - implementation of the VGG19 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization with partially frozen model weights (1st step),
* **Binary_Classification_VGG19_2nd_Stage.R** - implementation of the VGG19 model for binary image classification using pretrained weights on the ImageNet dataset - feature extraction and fine-tunning optimization of model weights with frostbitten layers (2nd step),
* **Binary_Classification_VGG19_Pipe_Runner.R** - executable script sequentially compiling Binary_Classification_VGG19_1st_Stage.R and Binary_Classification_VGG19_2nd_Stage.R files,



* **Binary_Categorical_Model_Evaluation.R:**
	* **Binary_Classifier_Verification** - function calculates all most important metrics in binary classification problems for provided actual classes and predicted probabilities. Available key metrics: True Negative, 
False Positive, False Negative, True Positive, Condition Positive, Condition Negative, Accuracy, Balanced Accuracy, Area Under Curve, Bias, Classification Error, True Positive Rate, True Negative Rate, Positive Predictive 
Value, Negative Predictive Value, False Negative Rate, False Positive Rate, False Discovery Rate, False Omission Rate, Threat Score, F1 Score, Bookmaker Informedness, Markedness, Gini Index, Cost.
	* **Binary_Classifier_Cutoff_Optimization** - function optimizes cutoff level for chosen key metric: True Negative, False Positive, False Negative, True Positive, Condition Positive, Condition Negative, Accuracy,
Balanced Accuracy, Area Under Curve, Bias, Classification Error, True Positive Rate, True Negative Rate, Positive Predictive Value, Negative Predictive Value, False Negative Rate, False Positive Rate, False Discovery Rate,
False Omission Rate, Threat Score, F1 Score, Bookmaker Informedness, Markedness, Gini Index, Cost.
	* **Categorical_Model_Evaluation** - function calculates Accuracy, Precision, Recall and F1 Score per classes and overall metrics calculated without division into classes for all provided observations. Additionaly
function converts multi-class classification into binary classification problem for each available class separately and computes metrics such as: True Negative, False Positive, False Negative, True Positive, Condition Positive,
Condition Negative, Accuracy, Balanced Accuracy, Area Under Curve, Bias, Classification Error, True Positive Rate, True Negative Rate, Positive Predictive Value, Negative Predictive Value, False Negative Rate, False 
Positive Rate, False Discovery Rate, False Omission Rate, Threat Score, F1 Score, Bookmaker Informedness, Markedness, Gini Index, Cost.

* **CNN_Binary_Model_Implementation.R:**
	* will be updated soon

* **Data_Augmentation.R:**
	* **Image_Augmentation** - function gives the opportunity to inspect possible ways of image data augmentation techniques and adapt random image modification settings for a specific classification problem.

* **Inseption_ResNet_V2_Binary_Classification.R:**
	* will be updated soon

* **Inception_V3_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using GoogLeNet Inception V3 pretrained model on ImageNet dataset.

* **Inception_V3_Categorical_Classification.R:**
	* Complete pipeline for multi-classification problem in images using GoogLeNet Inception V3 pretrained model on ImageNet dataset.

* **ResNet50_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using ResNet50 pretrained model on ImageNet dataset.

* **ResNet50_Categorical_Classification.R:**
	* Complete pipeline for multi-classification problem in images using ResNet50 pretrained model on ImageNet dataset.

* **Useful_Functions.R:**
	* will be updated soon

* **VGG16_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using VGG16 pretrained model on ImageNet dataset.

* **VGG16_Categorical_Classification.R:**
	* Complete pipeline for multi-classification problem in images using VGG16 pretrained model on ImageNet dataset.

* **VGG19_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using VGG19 pretrained model on ImageNet dataset.

* **VGG19_Categorical_Classification.R:**
	* Complete pipeline for multi-classification problem in images using VGG19 pretrained model on ImageNet dataset.

* **Xception_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using Xception pretrained model on ImageNet dataset.

* **Xception_Categorical_Classification.R:**
	* Complete pipeline for multi-classification problem in images using Xception pretrained model on ImageNet dataset.

----------
# Setup Anaconda environment
1. **Install Anaconda Python 3.7 version:**
* Download from https://www.anaconda.com/distribution/
* Open Anaconda Prompt
2. **Create and activate new anaconda environment (e.g. GPU_ML_2):**
```
conda create -n GPU_ML_2 python=3.6
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
pip install keras==2.3.1
```
6. **Install R:**
* Download R from https://cran.r-project.org/
7. **Install RStudio IDE:**
* Download RStudio from https://rstudio.com/
8. **Open RStudio and install Keras and Tensorflow with GPU support:**
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
----------






