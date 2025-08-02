import os
import functools
import pickle
import logging
from tqdm import tqdm
import gc

import torch
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from dotenv import load_dotenv

# from config import setup_logging

# setup_logging()
load_dotenv()

CACHE = os.getenv("CACHE")

def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                logging.info('pca results are already saved in cache')
                return 
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result, f,  protocol=4)
            return 
        
        return wrapper
    return decorator


class _PCA:
    def __init__(self,
                 n_components:int=None,
                 device:str = 'cuda'):
        self.n_components = n_components
        self.device = device
        
        if not os.path.exists(os.path.join(CACHE,'pca')):
            os.mkdir(os.path.join(CACHE,'pca'))
        
    @staticmethod
    def cache_file(iden, X, batch_size=None):
        return os.path.join('pca',iden)

    @cache(cache_file)
    def _fit(self, iden, X, batch_size=None):  
        iden += f'_principal_components={self.n_components}'
        X = torch.Tensor(X)
        pca = PCA(n_components=self.n_components)
        pca.fit(X)
        return pca


# import os
# import sys
# import xarray as xr
# from sklearn.decomposition import PCA
# import functools
# import pickle
# import torch
# from config import CACHE



# def cache(file_name_func):

#     def decorator(func):
        
#         @functools.wraps(func)
#         def wrapper(self, *args, **kwargs):

#             file_name = file_name_func(*args, **kwargs) 
#             cache_path = os.path.join(CACHE, file_name)
            
#             if os.path.exists(cache_path):
#                 print('pca results are already saved in cache')
#                 return 
            
#             result = func(self, *args, **kwargs)
#             with open(cache_path,'wb') as f:
#                 pickle.dump(result, f,  protocol=4)
#             return 
        
#         return wrapper
#     return decorator




# class _PCA:
    
#     def __init__(self,
#                  n_components:int=None,
#                  device:str = 'cuda'):
        
#         self.n_components = n_components
#         self.device = device
        
#         if not os.path.exists(os.path.join(CACHE,'pca')):
#             os.mkdir(os.path.join(CACHE,'pca'))
     
        
#     @staticmethod
#     def cache_file(iden, X):
#         return os.path.join('pca',iden)

    
#     @cache(cache_file)
#     def _fit(self, iden, X):  
   
#         X = torch.Tensor(X)
#         pca = PCA(n_components=self.n_components)
#         pca.fit(X)
        
#         return pca
