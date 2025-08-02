import os
import functools
import pickle
import logging

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
# from cuml.decomposition import PCA

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
            
            print(cache_path)
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
                 n_components:int=None,
                 device:str = 'cuda'):
        self.n_components = n_components
        self.device = device
        
        if not os.path.exists(os.path.join(CACHE,'pca')):
            os.mkdir(os.path.join(CACHE,'pca'))
        
    @staticmethod
    def cache_file(iden, X, batch_size=None):
        return os.path.join('pca',iden)

    def set_components(self, X):
        dims = X.shape
        if isinstance(self.n_components, int) and self.n_components > min(dims):
            self.n_components = min(dims)
        print('dims:',dims, 'n components:', self.n_components)
        return 
        
    @cache(cache_file)
    def _fit(self, iden, X, incremental=False, batch_size=2000):  
        # X = torch.Tensor(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # X = X.astype('float32')
        print('test',self.n_components)
        # pca = PCA(n_components=self.n_components)#, svd_solver='full')
        print('features',X.shape[-1])
        # if self.n_components > min(X.shape[-1],X.shape[0]):
        # self.n_components = min(X.shape[-1],X.shape[0])
        self.set_components(X)
        if incremental:
            print('incremental')
            pca = IncrementalPCA(n_components=self.n_components, batch_size=self.n_components)
        else:
            print('not incremental')
            pca = PCA(n_components=self.n_components)

        print('starting fit')
        pca.fit(X)
        print('finished fitting')
        return pca

        # pca_incremental = IncrementalPCA(self.n_components)

        # for i in range(0, X.shape[0], batch_size):
        #     X_batch = X[i:i + batch_size]
        #     pca_incremental.partial_fit(X_batch)
        # return pca_incremental
    

