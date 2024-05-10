# High performing untrained models of primate visual cortex

![The expansion model architecture](model.png)

The expansion model is a learning-free convolutional neural network based on compressession in the spatial domain and expansion in the feature domain. To use the model for your own data, please follow the steps incuded in section 1.  


# 1. The (feature) Expansion Model

- Please download (only) the folder models (/model_features/models).
  

- Navigate to the repo folder and install requirements:
```
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


# 2. Predicting Human fMRI Responses to Natural Scenes

Data:
The Natural Scenes Dataset human fMRI dataset (Allen et al., 2022) can be downloaded [here](https://naturalscenesdataset.org/). 

Preprocessing:
We use the NSD single-trial betas, preprocessed in 1.8-mm volume space and denoised using the GLMdenoise technique (version 3; “betas_fithrf_GLMdenoise_RR”) (Kay et al., 2013). We converted the betas to z-scores within each scanning session and computed the average betas for each NSD stimulus across repetitions. 

 
# 3. Predicting Macaque Single Cell Responses to Objects 

Data:
The monkey electrophysiology dataset (Majaj et al., 2015) is available as part of the [Brain-score GitHub package](https://github.com/brain-score):. 

Preprocessing:
We use the average response to stimuli across repetitions. 


# 4. Image Classification 

Data:
The Places dataset (Zhou et al, 2018) can be downloaded [here](http://places2.csail.mit.edu/)


  
  
