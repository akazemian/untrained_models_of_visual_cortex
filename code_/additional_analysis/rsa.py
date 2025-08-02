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


def compute_rsa_majajhong(iden, region):
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
    activations = load_activations(
        activations_identifier=iden, 
        mode='all')
    
    rdm = RDM()
    rsa = RSA(metric='pearsonr')
    
    rsa_all = []
    for subject in tqdm(SUBJECTS):
        # Compute RDM for neural data once, since it doesn't change across features
        neural_data = load_majaj_data(mode= 'all', subject = subject, region= region)
        RDM_neural = rdm(neural_data)
    
        # Compute the RDM for the activations
        RDM_expansion = rdm(activations)
        
        # Compute the similarity between neural RDM and activations RDM
        rsa_ = rsa(RDM_neural, RDM_expansion)
        rsa_all.append(rsa_)

    return sum(rsa_all)/len(rsa_all) # mean accross subjects


# class RDM:
#     """
#     Optimized Representational Dissimilarity Matrix (RDM).
#     Converts a dataset of stimulus x voxels (or activations) into a `stimulus x stimulus` RDM.
    
#     This version is optimized to handle large matrices.
#     """
#     def __init__(self, use_gpu: bool = False):
#         """
#         Args:
#         - use_gpu (bool): Whether to use GPU for computations (requires PyTorch).
#         """
#         self.use_gpu = use_gpu and torch.cuda.is_available()

#     def __call__(self, data):
#         """
#         Args:
#         - data (numpy.ndarray): Input data (stimulus x voxels).

#         Returns:
#         - dissimilarity (numpy.ndarray): Stimulus x stimulus RDM.
#         """
#         # Use GPU if available and requested
#         if self.use_gpu:
#             return self._compute_rdm_gpu(data)
#         else:
#             return self._compute_rdm_cpu(data)

#     def _compute_rdm_cpu(self, data):
#         """
#         Compute RDM on CPU using memory-efficient techniques.
#         """
#         # Compute pairwise distances (1 - Pearson correlation)
#         data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
#         dissimilarity = 1 - squareform(pdist(data, metric="correlation"))
        
#         # Ensure diagonal is zero
#         np.fill_diagonal(dissimilarity, 0)
#         return dissimilarity

#     def _compute_rdm_gpu(self, data):
#         """
#         Compute RDM on GPU using PyTorch for acceleration.
#         """
#         # Convert to PyTorch tensor
#         data = torch.tensor(data, dtype=torch.float32, device="cuda")
        
#         # Normalize data
#         data = (data - data.mean(dim=1, keepdim=True)) / data.std(dim=1, keepdim=True)
        
#         # Compute pairwise correlations
#         similarity = torch.matmul(data, data.T) / data.shape[1]
#         dissimilarity = 1 - similarity.cpu().numpy()
        
#         # Ensure diagonal is zero
#         np.fill_diagonal(dissimilarity, 0)
#         return dissimilarity


# import numpy as np
# from scipy.stats import pearsonr, spearmanr
# import torch

# class RDMSimilarity:
#     """
#     Optimized class for computing similarity between two RDMs.
#     """
#     def __init__(self, metric: str, use_gpu: bool = False):
#         """
#         Args:
#         - metric (str): Similarity metric ('pearsonr' or 'spearmanr').
#         - use_gpu (bool): Whether to use GPU for computations (requires PyTorch).
#         """
#         self.metric = metric.lower()
#         self.use_gpu = use_gpu and torch.cuda.is_available()

#     def __call__(self, rdm_1, rdm_2):
#         """
#         Compute similarity between two RDMs.
#         """
#         # Ensure RDMs have the same shape
#         assert rdm_1.shape == rdm_2.shape, "RDMs must have the same dimensions."
        
#         # Extract upper triangular parts
#         triu1 = self._triangulars(rdm_1)
#         triu2 = self._triangulars(rdm_2)
        
#         # Compute similarity
#         if self.metric == 'pearsonr':
#             return self._compute_pearson(triu1, triu2)
#         elif self.metric == 'spearmanr':
#             return self._compute_spearman(triu1, triu2)
#         else:
#             raise ValueError(f"Unsupported metric: {self.metric}. Use 'pearsonr' or 'spearmanr'.")

#     def _triangulars(self, values):
#         """
#         Extract upper triangular part of a matrix, excluding the diagonal.
#         """
#         # Ensure diagonal is zero
#         diag = np.diag(values)
#         diag = np.nan_to_num(diag, nan=0, copy=True)  # Handle NaNs in diagonal
#         np.testing.assert_almost_equal(diag, 0, err_msg="Diagonal must be zero.")
        
#         # Extract upper triangular indices
#         triangular_indices = np.triu_indices(values.shape[0], k=1)
#         return values[triangular_indices]

#     def _compute_pearson(self, triu1, triu2):
#         """
#         Compute Pearson correlation between two flattened triangular parts.
#         """
#         if self.use_gpu:
#             # Use PyTorch for GPU acceleration
#             triu1_tensor = torch.tensor(triu1, dtype=torch.float32, device="cuda")
#             triu2_tensor = torch.tensor(triu2, dtype=torch.float32, device="cuda")
#             corr = torch.corrcoef(torch.stack([triu1_tensor, triu2_tensor]))[0, 1]
#             return corr.item()
#         else:
#             # Use CPU with scipy
#             corr, _ = pearsonr(triu1, triu2)
#             return corr

#     def _compute_spearman(self, triu1, triu2):
#         """
#         Compute Spearman correlation between two flattened triangular parts.
#         """
#         if self.use_gpu:
#             # Spearman correlation is not natively available in PyTorch, so fallback to CPU
#             print("Spearman correlation is computed on CPU. GPU support is limited.")
#         corr, _ = spearmanr(triu1, triu2)
#         return corr

