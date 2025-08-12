import os
from tqdm import tqdm 

import numpy as np
import xarray as xr
from scipy.stats import spearmanr, pearsonr
from dotenv import load_dotenv

from code_.encoding_score.benchmarks.majajhong import load_majaj_data, load_activations
from code_.encoding_score.benchmarks.nsd import load_nsd_data, filter_activations

import numpy as np
load_dotenv()
CACHE = os.getenv("CACHE")

class RDM:
    """
    Representational Dissimilarity Matrix.
    Converts an xarray dataset of stimulus x voxels (or activations) into a `stimulus x stimulus` RDM.

    Kriegeskorte et al., 2008 https://doi.org/10.3389/neuro.06.004.2008
    """  
      
    def __call__(self, data):
        dissimilarity = 1 - np.corrcoef(data)
        return dissimilarity

        
class RSA:
    def __init__(self, metric:str):
        self.metric = metric
        
    def __call__(self, rdm_1, rdm_2):
        triu1 = self._triangulars(rdm_1)
        triu2 = self._triangulars(rdm_2)
        match self.metric:
            case 'pearsonr':
                corr, p = pearsonr(triu1, triu2)
            case 'spearmanr':
                corr, p = spearmanr(triu1, triu2)
        return corr

    def _triangulars(self, values):
        # ensure diagonal is zero
        diag = np.diag(values)
        diag = np.nan_to_num(diag, nan=0, copy=True)  # we also accept nans in the diagonal from correlating zeros
        np.testing.assert_almost_equal(diag, 0)
        # index and retrieve upper triangular
        triangular_indices = np.triu_indices(values.shape[0], k=1)
        return values[triangular_indices]


def compute_rsa_nsd(iden, region):
    """
    Computes similarity between neural data and activations for each feature in features_list.
    
    Args:
    - features_list: List of feature values to be used in activations identifier.
    - neural_data: The neural data for which the RDM is calculated.
    
    Returns:
    - sim_expansions: List of rsa results for each feature.
    """    
    # Loop over each feature and compute the similarity
        # Load the activations using the feature value
    SUBJECTS = [i for i in range(8)]
    activations_data = xr.open_dataarray(os.path.join(CACHE,'activations',iden),engine='netcdf4')
    
    rdm = RDM()
    rsa = RSA(metric='pearsonr')
    
    rsa_all = []
    for subject in SUBJECTS:
        # Compute RDM for neural data once, since it doesn't change across features
        filetered_ids, neural_data = load_nsd_data(mode ='all', subject = subject, region = region)
        RDM_neural = rdm(neural_data.beta.values)
    
        activations = filter_activations(data = activations_data, ids = filetered_ids)  
        # Compute the RDM for the activations
        RDM_expansion = rdm(activations)
        
        # Compute the similarity between neural RDM and activations RDM
        rsa_ = rsa(RDM_neural, RDM_expansion)
        rsa_all.append(rsa_)

    return sum(rsa_all)/len(rsa_all) # mean accross subjects


def compute_rsa_majajhong(iden, region, demo=False):
    """
    Computes similarity between neural data and activations for each feature in features_list.
    
    Args:
    - features_list: List of feature values to be used in activations identifier.
    - neural_data: The neural data for which the RDM is calculated.
    
    Returns:
    - sim_expansions: List of rsa results for each feature.
    """    
    # Loop over each feature and compute the similarity
        # Load the activations using the feature value
    SUBJECTS = ['Chabo','Tito']
    if demo:
        activations = load_activations(activations_identifier=iden, mode='train_demo')
    else:
        activations = load_activations(activations_identifier=iden, mode='all')
    
    rdm = RDM()
    rsa = RSA(metric='pearsonr')
    
    rsa_all = []
    for subject in tqdm(SUBJECTS):
        # Compute RDM for neural data once, since it doesn't change across features
        if demo:
            neural_data = load_majaj_data(mode= 'train_demo', subject = subject, region= region)
        else:
            neural_data = load_majaj_data(mode= 'all', subject = subject, region= region)
        RDM_neural = rdm(neural_data)
    
        # Compute the RDM for the activations
        RDM_expansion = rdm(activations)
        
        # Compute the similarity between neural RDM and activations RDM
        rsa_ = rsa(RDM_neural, RDM_expansion)
        rsa_all.append(rsa_)

    return sum(rsa_all)/len(rsa_all) # mean accross subjects
