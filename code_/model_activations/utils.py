import os
import functools
import pickle
import torch
import gc

from dotenv import load_dotenv
load_dotenv()
    
# env paths
CACHE = os.getenv("CACHE")

def register_pca_hook(x: torch.Tensor, pca_file_name: str, n_components, 
                      device) -> torch.Tensor:
    """
    Applies a PCA transformation to the tensor x using precomputed PCA parameters.

    Args:
        x (torch.Tensor): The input tensor for which PCA should be applied.
        pca_file_name (str): The file name where PCA parameters are stored.
        n_components (int, optional): Number of principal components to keep. If None, all components are used.
        device (str): Device to perform the computations on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The transformed tensor after applying PCA.
    """
    pca_path = os.path.join(CACHE, 'pca', pca_file_name)

    with open(pca_path, 'rb') as file:
        _pca = pickle.load(file)
    
    _mean = torch.Tensor(_pca.mean_).to(device)
    _eig_vec = torch.Tensor(_pca.components_.transpose()).to(device)
    
    x = x.squeeze()
    x -= _mean
    
    if n_components is not None:
        pcs = x @ _eig_vec[:, :n_components]
        print('pcs shape', pcs.shape)
        return pcs
    else:
        pcs = x @ _eig_vec
        print('pcs shape', pcs.shape)
        return pcs

def cache(file_name_func):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            if os.path.exists(cache_path):
                print('activations are already saved in cache')
                return 
            
            result = func(self, *args, **kwargs)
            result.to_netcdf(cache_path, engine='netcdf4')
            gc.collect()
            return 

        return wrapper
    return decorator
   