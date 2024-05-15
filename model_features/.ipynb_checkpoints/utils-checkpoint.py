from config import CACHE 
import functools
import pickle
import torch
import gc

def register_pca_hook(x, pca_file_name, n_components=None, device='cuda'):
    
    pca_file_name = pca_file_name.split('_principal_components')[0]
    
    with open(PCA_FILE_NAME, 'rb') as file:
        _pca = pickle.load(file)
    _mean = torch.Tensor(_pca.mean_).to(device)
    _eig_vec = torch.Tensor(_pca.components_.transpose()).to(device)
    x = x.squeeze()
    x -= _mean
    
    if n_components is not None:
        return x @ _eig_vec[:, :n_components]
    else:
        return x @ _eig_vec

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
   