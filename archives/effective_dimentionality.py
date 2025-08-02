import numpy as np
import os
import xarray as xr
import pickle

from code_.model_activations.configs import model_cfg
from code_.model_activations.loading import load_full_identifier
from config import CACHE, FIGURES_ADDITIONAL


import numpy as np
from sklearn.decomposition import PCA

def participation_ratio(X):
    """
    Computes the Participation Ratio (effective dimensionality) using a random sample
    of rows and randomized PCA for memory efficiency.
    
    Parameters:
      X : numpy array of shape (n_samples, n_features)
          The full data matrix.
      sample_size : int, number of rows (images) to sample. If X has fewer rows than sample_size,
                    the entire dataset is used.
      seed : int, random seed for reproducibility.
    
    Returns:
      pr : float
          The participation ratio computed on the sampled data.
    """

    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Use randomized PCA (this is more memory efficient for large feature dimensions)
    pca = PCA(svd_solver='randomized', random_state=42)
    pca.fit(X_centered)
    
    # The eigenvalues of the covariance matrix are given by explained_variance_
    eigvals = pca.explained_variance_
    # Clip small negative values due to numerical issues
    eigvals = np.clip(eigvals, a_min=0, a_max=None)
    
    # Compute Participation Ratio (PR)
    pr = (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)
    return pr



DATASET = 'naturalscenes'
models = ['expansion','fully_connected','vit']
model_dict = {}

np.random.seed(42)
indices = np.random.choice(73000, 30000, replace=False)

for model_name in models:
        
    model_dict[model_name] = []
    features_list = model_cfg[DATASET]['models'][model_name]['features']
    
    for feature in features_list:
        
        identifier = load_full_identifier(model_name=model_name, 
                                    features=feature, 
                                    layers=5, 
                                    dataset=DATASET)
    
        model_features = xr.open_dataarray(os.path.join(CACHE,'activations',identifier),engine='netcdf4')
        model_features_sample = model_features.values[indices,:]
        print(model_features_sample.shape)
        ed = participation_ratio(model_features_sample)
        model_dict[model_name].append(ed)
        print(model_name, feature, 'ed:',ed)

with open('/home/akazemi3/Desktop/untrained_models_of_visual_cortex/results/additional_analysis/ed','wb') as f:
    pickle.dump(model_dict,f)
    
