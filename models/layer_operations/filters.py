import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import random





def make_random_filters(out_channels,in_channels,kernel_size):
    """
    Creates random filters from a uniform distribution  
    """
    torch.manual_seed(27)
    w = torch.rand(out_channels,in_channels,kernel_size,kernel_size)
    w -= w.mean(dim = [2,3],keepdim=True) # mean centering
        
    return w




    
class CurvatureModel(nn.Module):
  
    
    def __init__(self,
                 n_ories=16,
                 in_channels=1,
                 curves=np.logspace(-2, -0.1, 5),
                 gau_sizes=(5,), filt_size=9, fre=[1.2], gamma=1, sigx=1, sigy=1):
        super().__init__()

        self.n_ories = n_ories
        self.curves = curves
        self.gau_sizes = gau_sizes
        self.filt_size = filt_size
        self.fre = fre
        self.gamma = gamma
        self.sigx = sigx
        self.sigy = sigy
        self.in_channels = in_channels

    def forward(self):
        i = 0
        ories = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_ories)
        w = torch.zeros(size=(len(ories) * len(self.curves) * len(self.gau_sizes) * len(self.fre), self.in_channels, self.filt_size, self.filt_size))
        for curve in self.curves:
            for gau_size in self.gau_sizes:
                for orie in ories:
                    for f in self.fre:
                        w[i, 0, :, :] = banana_filter(gau_size, f, orie, curve, self.gamma, self.sigx, self.sigy, self.filt_size)
                        i += 1
        return w        

    
    

def banana_filter(s, fre, theta, cur, gamma, sigx, sigy, sz):
    # Define a matrix that used as a filter
    xv, yv = np.meshgrid(np.arange(np.fix(-sz/2).item(), np.fix(sz/2).item() + sz % 2),
                         np.arange(np.fix(sz/2).item(), np.fix(-sz/2).item() - sz % 2, -1))
    xv = xv.T
    yv = yv.T

    # Define orientation of the filter
    xc = xv * np.cos(theta) + yv * np.sin(theta)
    xs = -xv * np.sin(theta) + yv * np.cos(theta)

    # Define the bias term
    bias = np.exp(-sigx / 2)
    k = xc + cur * (xs ** 2)

    # Define the rotated Guassian rotated and curved function
    k2 = (k / sigx) ** 2 + (xs / (sigy * s)) ** 2
    G = np.exp(-k2 * fre ** 2 / 2)

    # Define the rotated and curved complex wave function
    F = np.exp(fre * k * 1j)

    # Multiply the complex wave function with the Gaussian function with a constant and bias
    filt = gamma * G * (F - bias)
    filt = np.real(filt)
    filt -= filt.mean()

    filt = torch.from_numpy(filt).float()
    return filt





def make_onebyone_filters(out_channels,in_channels):
    """
    Creates 1x1 convolution filters used for random projection
    """
    torch.manual_seed(27)
    w = torch.randn(out_channels,in_channels, 1, 1) 
    w = w/np.sqrt(out_channels)
    return w





def filters(filter_type,out_channels=None,in_channels=None,curv_params = None,kernel_size=None):

    """
    Returns the filters based on filter type
    
    Arguments
    ----------  
    
    filter_type:
    out_channels:
    in_channels:
    curv_params:
    kernel_size:
 
    """
        
    assert filter_type in ['random','1x1','curvature'], "filter should be one of 'random', '1x1' or 'curvature'"
    
    if filter_type == 'random':
        return make_random_filters(out_channels,in_channels,kernel_size)

    elif filter_type == '1x1':
        return make_onebyone_filters(out_channels,in_channels)

    elif filter_type == 'curvature':

        curve = CurvatureModel(
            in_channels=in_channels,
            n_ories=curv_params['n_ories'],
            gau_sizes=curv_params['gau_sizes'],
            curves=np.logspace(-2, -0.1, curv_params['n_curves']),
            fre = curv_params['spatial_fre'],
            filt_size=kernel_size)
        return curve()