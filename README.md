- [Overview](#overview)
- [System requirements](#system-requirements)
  - [Hardware requirements](#hardware-requirements)
  - [Software requirements](#software-requirements)
- [Installation guide](#installation-guide)
  - [Using the Expansion Model](#using-the-expansion-model)
  - [Entire Repository](#entire-repository)
- [Information about the datasets](#information-about-the-datasets)
- [License](#license)
- [References](#references)


# Overview
![The expansion model architecture](docs/model.png)

The expansion model is a learning-free convolutional neural network based on compressession in the spatial domain and expansion in the feature domain. To use the model for your own data, please follow the steps incuded in section 1. To reproduce the results from our bioarxiv preprint, refer to section 2. 


# System requirements

## Hardware requirements
 The code requires only a standard computer with enough CPU and GPU compute power to support all operations. The scripts for replicating the main results use about ~18 GB GPU RAM at peak. 


## Software requirements

### OS requirements
The code has been tested on Fedora Linux 39 as well as RHEL 9.3. 

### Python version
The code has been tested on Python==3.10.14

### Python dependencies
The following is a list of python libraries with version numbers required to run all scripts:
```
pillow==10.3.0
opencv-python==4.10.0.84
loguru==0.7.2
matplotlib==3.9.0
numpy==2.0.0
pandas==2.2.2
scipy==1.13.1
seaborn==0.13.2
scikit-learn==1.5.0
timm==1.0.7
torch==2.3.1
torchmetrics==1.4.0.post0
torchvision==0.18.1
tqdm==4.66.4
xarray==2024.6.0
netCDF4==1.7.1
cupy-cuda12x==13.2.0
python-dotenv==1.0.1
```

# Installation guide

## Using the Expansion model

- Please download (only) the folder ```/code/model_activations/models```. This is easily done using https://download-directory.github.io/.
  

- Navigate to the repo folder, then install the requirements:
```
conda create -n expansion_model python==3.10.14
conda activate expansion_model
pip install -r requirements.txt 
```

Below is an example of how to use the expansion model. Alternatively, you can navigate to 'main.ipynb` for an example.

1. Import the model
```python
from expansion import Expansion5L
```

2. Import (preprocessed) images as a torch tensor. Below is a random image tensor of size NxCxHxW. Where N = number of images, C = number of channels, H = height of the image, W = width of the image.
   
```python
X = troch.Tensor(1,3,224,224)
```

3. Instantiate the model
```python
expansion_model = Expansion5L(filters_5 = 3000, # number of filters in the last convolution layer of the model
                              device='cuda').Build()

```

4. Extract image features
```python
features = expansion_model(X)
```

The ouput is a tensor of size NxP, where N = the number of image and P = the number of features in the last layer of the model.

## Entire repository

### Initial setup

First, download the data used in all analysis [here](https://www.dropbox.com/scl/fo/ow0v17ldsax4iddtp82aj/AEwkme4Crdi0d80hv2zigC8?rlkey=ne57t05d1dkgotkymtxxyigzm&st=pbuh26zf&dl=0) and unzip each dataset folder. The download may take a while (~15 minutes) given the size of the datasets (~30 GB). To run only the demo, download the majajhong and places folders.


Clone this repository and navigate to the repository folder.
```
git clone https://github.com/akazemian/untrained_models_of_visual_cortex.git
cd untrained_models_of_visual_cortex
```

In the root directory, open ```.env``` and set the path for the ```CACHE``` and ```DATA``` (where the data was downloaded) folders. 

Install required packages (1-2 minutes)
```
conda create -n expansion_project python==3.10.14
conda activate expansion_project
pip install .
```

## Running the demo
There are 2 datasets used for the demo. For analyses involving neural data, a subset of the majajhong dataset (50 images) is used. For analyses relating to image classification, a subset of the Places train set (500 images) and validation set (500 images) is used. Further, the smallest version of each untrained model is considered for all parts of the demo. Each demo has a separate notebook in the folder ```demo_notebooks```, where the output can be visualized. Alternatively, the scripts for generating demo results can be run as shown below. Running all demo files takes 4-5 minutes.  

To generate brain similarity scores for the untrained models and alexnet (total time: 7-8 seconds): 
```
python demo/main_results.py 
```

To generate the PCA results (total time: 14-15 seconds):
```
python demo/pca_analysis.py
```

To generate model ablation results (total time: 80 seconds):
```
python demo/ablation_studies.py
```

To generate image classification results using the places dataset (total time: 2-3 minutes):
```
python demo/classification.py 
```

### Replicating results

Navigate to the project directory and make sure to specify the dataset (```majajhong``` or ```naturalscenes```) and the device (```cuda``` or ```cpu```) when running the following. 

To generate brain similarity score for the untrained models and alexnet: 
```
python code/main_results.py --dataset majajhong --device cuda
```

To generate the PCA results:
```
python code/pca_analysis.py --dataset majajhong --device cuda
```

To generate model ablation results:
```
python code/ablation_studies.py --dataset majajhong --device cuda
```

To generate image classification results using the places dataset:
```
python code/classification.py --device cuda
```
### Dealing with memory issues

If there are any memory issues when running the above, try:
- running the script again 
- changing the batch size with --batchsize.
- changing the device between cpu and gpu.
  
### 3. Generating figures

Navigate to the ```notebooks``` folder. Here you will find notebooks for generating each figure individually. These are saved in the ```figures``` folder.

# Information about the datasets:

When generating the results, the stimuli and preprocessed neural data are downloaded using the DATA path set earlier. This is done to increase efficiency. Alternatively, information about how to download and process the raw data manually is described below. The code used for preprocessing the neural data can be found in ```tools/neural_data_processing```

## The Majajhong dataset 

**Data**: The monkey electrophysiology dataset (Majaj et al., 2015) is available as part of the [Brain-score GitHub package](https://github.com/brain-score):. 

**Preprocessing**: We use the average response to stimuli across repetitions. 

## The Naturalscenes dataset:

Data: The Natural Scenes Dataset human fMRI dataset (Allen et al., 2022) can be downloaded [here](https://naturalscenesdataset.org/). 

Preprocessing: We use the NSD single-trial betas, preprocessed in 1.8-mm volume space and denoised using the GLMdenoise technique (version 3; “betas_fithrf_GLMdenoise_RR”) (Kay et al., 2013). We converted the betas to z-scores within each scanning session and computed the average betas for each NSD stimulus across repetitions. 


## The Places dataset:

Data: The Places dataset (Zhou et al, 2018) can be downloaded [here](http://places2.csail.mit.edu/)

# License

This project is covered under the MIT License.

# References

Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle, L. T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J. B., Naselaris, T., & Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. Nature Neuroscience, 25(1), 116–126. https://doi.org/10.1038/s41593-021-00962-x 

Majaj, N. J., Hong, H., Solomon, E. A., & DiCarlo, J. J. (2015). Simple Learned Weighted Sums of Inferior Temporal Neuronal Firing Rates Accurately Predict Human Core Object Recognition Performance. Journal of Neuroscience, 35(39), 13402–13418. https://doi.org/10.1523/JNEUROSCI.5181-14.2015 

Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A. (2018). Places: A 10 Million Image Database for Scene Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(6), 1452–1464. https://doi.org/10.1109/TPAMI.2017.2723009 



  
  
