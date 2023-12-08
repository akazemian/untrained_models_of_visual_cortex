import sys
import os
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)

from config import CACHE 
import functools
import pickle
import torch
import gc



def register_pca_hook(x, PCA_FILE_NAME, n_components=1000, device='cuda'):
    
    with open(PCA_FILE_NAME, 'rb') as file:
        _pca = pickle.load(file)
    _mean = torch.Tensor(_pca.mean_).to(device)
    _eig_vec = torch.Tensor(_pca.components_.transpose()).to(device)
    x = x.squeeze()
    x -= _mean
    
    return x @ _eig_vec[:, :n_components]




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
            result.to_netcdf(cache_path)
            gc.collect()
            return 

        return wrapper
    return decorator
