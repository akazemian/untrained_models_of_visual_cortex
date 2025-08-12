import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

import os
import functools
import pickle
import logging

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from config import CACHE


def num_pcs_required(variance_ratio, explained_variance=0.99):    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(variance_ratio)
    # Find the number of components required to explain at least 99% of variance
    num_components = np.argmax(cumulative_variance >= explained_variance) + 1
    return num_components
    
def powerlaw_exponent(eigspec: np.ndarray) -> float:
    start, end = 0, np.log10(len(eigspec))
    eignum = np.logspace(start, end, num=50).round().astype(int)
    eigspec = eigspec[eignum - 1]
    logeignum = np.log10(eignum)
    logeigspec = np.log10(eigspec)

    # remove infs when eigenvalues are too small
    filter_ = ~np.isinf(logeigspec)
    logeignum = logeignum[filter_]
    logeigspec = logeigspec[filter_]
    linear_fit = LinearRegression().fit(logeignum.reshape(-1, 1), logeigspec)
    alpha = -linear_fit.coef_.item()
    return alpha, linear_fit.intercept_

def plot_eigspec(data, label, color, log_scale=True):
    a, y = powerlaw_exponent(data)
    sns.lineplot(x=np.arange(1,len(data)+1),y=data/(10**y),label=label,c=color) #, alpha = {round(a,2)}
    #plt.bar(np.arange(1,len(data)+1),data,label=f'alpha = {round(a,2)}')
    plt.xscale('log')
    if log_scale:
        plt.yscale('log')
    plt.legend()

def plot_ref(data):
    a, y = powerlaw_exponent(data)
    idx = np.arange(0,len(data))
    sns.lineplot(x=idx, y=1/idx, label='reference') 
    plt.xscale('log')
    plt.yscale('log')  
    plt.legend()

def rescale_pca_variance(principal_components):
    """
    Rescales the variance of principal components to decay as a power law with a -1 index.

    Args:
    principal_components (numpy.ndarray): A 2D array where each column represents a principal component.

    Returns:
    numpy.ndarray: A 2D array of rescaled principal components.
    """
    # Number of components
    num_components = principal_components.shape[1]

    # Calculate the original variances
    original_variances = np.var(principal_components, axis=0)

    # Determine the constant C as the variance of the first component
    C = original_variances[0]

    # Calculate the scaling factors for each component
    scaling_factors = np.sqrt(C / np.arange(1, num_components + 1))

    # Rescale each principal component
    rescaled_components = principal_components * scaling_factors

    return rescaled_components


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
    

