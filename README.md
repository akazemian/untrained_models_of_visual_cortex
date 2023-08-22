# Modeling the brain using an untrained CNN with a rich representational space

The expansion model is a learning-free convolutional neural network based on high-dimensional random sampling. To use the model as an encoding model of visual cortex for your own data, please follow the steps incuded in the Setup sectionfollowed by those in section: The (dimensionality) Expansion Model. 


# Setup

- Clone this repository to a local directrory. 
```
git clone https://github.com/akazemian/random_models_of_visual_cortex.git
```
- Install requieremnts
```
pip install -r requirements.txt 
```
- Add the the follwoing line to your list of environment variables:
```
export BONNER_ROOT_PATH='<local path to this repo>'
```
- Navigate to the local repository, in the config.py file, set the paths for the neural data and their corresponing stimuli.

# The (dimensionality) Expansion Model
Below is an example of how to use it the expansion model.

1. Import the model
```python
from model_features.models.expansion_3_layers import Expansion
```

2. Import preprocessed images as a torch tensor. Below is a random tensor of size NxCxHxW.
```python
X = troch.Tensor(1,3,224,224)
```

3. Instantiate the model
```python
expansion = Expansion()
model = expansion.Build()
```

4. Extract model features
```python
output = model(X)
```

The ouput is a tensor of size NxP, where N is the number of image and P is the number of features in the last layer of the model.


# Predicting Neural Data
- Navigate to the local repo directory 
- In ```model_evaluation/predicting_brain_data/compute_encoding_score.py```, choose the neural dataset and the model for which encoding perofrmance will be obtained.
- Run the following code to extract model activations and use them to predict neural responses:
```model_evaluation/predicting_brain_data/compute_encoding_score.py```
- The results can be viewed in ```model_evaluation/results/predicting_brain_data/encoding_performance.ipynb```

# Image Classification 
- Navigate to the local repo directory 
- Eun the following code:
```
python model_evaluation/image_classification/classification.py
```
- View the results in ```model_evaluation/results/classification/classification_results.ipynb```
  
  
