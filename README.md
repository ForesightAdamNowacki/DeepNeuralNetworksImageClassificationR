----------
# DeepNeuralNetworksRepoR

Deep Neural Networks using Keras and Tensorflow-GPU in R. A repository devoted to binary and multi-class classification issues in recognizing objects in images.

----------
# Setup anaconda environment
1. Install Anaconda Python 3.7 version:
* Download from https://www.anaconda.com/distribution/
* Open Anaconda Prompt
2. Create and activate new anaconda environment (e.g. GPU_ML_2) in Anaconda Prompt:
* conda create -n GPU_ML_2 python=3.6
* conda activate GPU_ML_2
3. Install Python kernell in Anaconda Prompt:
* pip install ipykernel
* python -m ipykernel install --user --name GPU_ML_2 --display-name "GPU_ML_2"
4. Install Tensorflow-GPU in Anaconda Prompt:
* conda install tensorflow-gpu==2.0.0
5. Install jupyter in Anaconda Prompt:
* conda install jupyter
* pip install keras==2.3.1 (use 'pip install' instead of 'conda install' what can downgrade Tensorflow-GPU package and may cause any problems) 
6. Install R:
* Download R from https://cran.r-project.org/
7. Install RStudio IDE:
* Download RStudio from https://rstudio.com/
8. Open RStudio and install Keras and Tensorflow with GPU support:
	utils::install.packages("reticulate")
	base::library(reticulate)
* reticulate::use_condaenv("GPU_ML_2", required = TRUE)
* utils::install.packages("tensorflow")
* utils::install.packages("keras")
* base::library(tensorflow)
* base::library(keras)
* keras::install_keras(tensorflow = "gpu")

----------
# cdn





