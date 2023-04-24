import torch
import pickle
import os
from torch import nn

PATH_TO_PCA = '/data/atlas/pca'


class SpatialPCA(nn.Module):
    
    def __init__(self, _pca, n_components=1000):
        
        super().__init__()
                
        
        self.n_components = n_components
        self._mean = torch.Tensor(_pca.mean_)
        self._eig_vec = torch.Tensor(_pca.components_.transpose())
        
   
    def forward(self, X):

        X = X.cpu()
        
        
        SHAPE = X.shape
        N = SHAPE[0]
        C = SHAPE[1]
        K = SHAPE[-1]
        

        
        spatial_eig_vec = self._eig_vec[:,:self.n_components]
        print('spatial_eig_vec',spatial_eig_vec.shape)
        spatial_mean = self._mean
        print('spatial_mean',spatial_mean.shape)
        
        
        X = torch.clone(torch.Tensor(X))
        X = X.view(N,-1)
        
        ind_orig = 0
        num_orig = C
        
        ind_pca = 0
        num_pca = self.n_components
        
        X_all_s = torch.zeros(N,num_pca*K**2)
        
        while ind_orig < num_orig*K**2:
            X_s = X[:,ind_orig:ind_orig+num_orig]
            X_s -= spatial_mean
            X_s = X_s @ spatial_eig_vec
            X_all_s[:,ind_pca:ind_pca+num_pca] = X_s[:,:]
            ind_orig += num_orig
            ind_pca += num_pca
                        
        
        
        return X_all_s.reshape(N,self.n_components,K,K).cuda()
    



class NormalPCA(nn.Module):

    def __init__(self, _pca, n_components=1000):
        
        super().__init__()
                
        
        self.n_components = n_components
        self._mean = torch.Tensor(_pca.mean_)
        self._eig_vec = torch.Tensor(_pca.components_.transpose())
        
   
    def forward(self, X):

        
        N = X.shape[0]
        X = X.cpu()
        X = X.view(N,-1)#X = torch.clone(torch.Tensor(X))
        X -= self._mean

        return X @ self._eig_vec[:, :self.n_components]