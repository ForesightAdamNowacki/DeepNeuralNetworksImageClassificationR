
![Alt text](D:/GitHub/Datasets/GitHub_Images/Keras.png?raw=true "Title")

----------
# DeepNeuralNetworksRepoR

Deep Neural Networks using Keras and TensorFlow-GPU in R. The repository is devoted to binary and multi-class object classification issues in images data.

----------
# Scripts description

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

* **Data_Augmentation.R:**
	* **Image_Augmentation** - function gives the opportunity to inspect possible ways of image data augmentation techniques and adapt random image modification settings for a specific classification problem.

* **Inception_V3_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using GoogLeNet Inception V3 pretrained model on ImageNet dataset.

* **Inception_V3_Categorical_Classification.R:**
	* Complete pipeline for multi-classification problem in images using GoogLeNet Inception V3 pretrained model on ImageNet dataset.

* **ResNet50_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using ResNet50 pretrained model on ImageNet dataset.

* **ResNet50_Categorical_Classification.R:**
	* Complete pipeline for multi-classification problem in images using ResNet50 pretrained model on ImageNet dataset.

* **Xception_Binary_Classification.R:**
	* Complete pipeline for binary classification problem in images using Xception pretrained model on ImageNet dataset.

----------
# Setup Anaconda environment
1. **Install Anaconda Python 3.7 version:**
* Download from https://www.anaconda.com/distribution/
* Open Anaconda Prompt
2. **Create and activate new anaconda environment (e.g. GPU_ML_2) in Anaconda Prompt:**
* conda create -n GPU_ML_2 python=3.6
* conda activate GPU_ML_2
3. **Install Python kernell in Anaconda Prompt:**
* pip install ipykernel
* python -m ipykernel install --user --name GPU_ML_2 --display-name "GPU_ML_2"
4. **Install Tensorflow-GPU in Anaconda Prompt:**
* conda install tensorflow-gpu==2.0.0
5. **Install jupyter in Anaconda Prompt:**
* conda install jupyter
* pip install keras==2.3.1 (use 'pip install' instead of 'conda install' what can downgrade Tensorflow-GPU package and may cause any problems) 
6. **Install R:**
* Download R from https://cran.r-project.org/
7. **Install RStudio IDE:**
* Download RStudio from https://rstudio.com/
8. **Open RStudio and install Keras and Tensorflow with GPU support:**
* utils::install.packages("reticulate")
* base::library(reticulate)
* reticulate::use_condaenv("GPU_ML_2", required = TRUE)
* utils::install.packages("tensorflow")
* utils::install.packages("keras")
* base::library(tensorflow)
* base::library(keras)
* keras::install_keras(tensorflow = "gpu")

----------






