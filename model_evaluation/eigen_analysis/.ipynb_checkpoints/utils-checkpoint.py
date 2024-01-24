import os
import sys
import xarray as xr
from sklearn.decomposition import PCA
import functools
import pickle
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE
import torch



def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                print('pca results are already saved in cache')
                return 
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result, f,  protocol=4)
            return 
        
        return wrapper
    return decorator




class _PCA:
    
    def __init__(self,
                 device:str = 'cuda'):
        
        self.device = device
        
        if not os.path.exists(os.path.join(CACHE,'pca')):
            os.mkdir(os.path.join(CACHE,'pca'))
     
        
    @staticmethod
    def cache_file(iden, X):
        return os.path.join('pca',iden)

    
    @cache(cache_file)
    def _fit(self, iden, X):  
   
        X = torch.Tensor(X)
        pca = PCA(n_components=1000)
        pca.fit(X)
        
        return pca
