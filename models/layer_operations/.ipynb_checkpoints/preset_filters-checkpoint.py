import torch
from torch import nn
import numpy as np

import numpy as np
import cv2

    
class CurvatureFilters(nn.Module):


    def __init__(self,
                 n_ories=16,
                 in_channels=1,
                 curves=np.logspace(-2, -0.1, 5),
                 gau_sizes=(5,), 
                 filt_size=9, 
                 fre=[1.2], 
                 gamma=1, 
                 sigx=1, 
                 sigy=1):
        
        
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





class GaborFilters(nn.Module):
    
    def __init__(self,
                 n_ories=12,
                 in_channels=1,
                 filt_size=9,
                 num_scales=1, 
                 min_scale=5,
                 max_scale=5,
                 ):
        
        
        super().__init__()

        self.n_ories = n_ories
        self.filt_size = filt_size
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.min_scale = min_scale
        self.max_scale = max_scale

        
    def forward(self):    
        
        orientations = np.linspace(0, np.pi, self.n_ories, endpoint=False)
        scales = np.linspace(self.min_scale, self.max_scale, self.num_scales)
        w = torch.zeros(size=(self.n_ories * self.num_scales, self.in_channels, self.filt_size, self.filt_size))

        i = 0
        for scale in scales:
            for orientation in orientations:
                sigma = scale
                theta = orientation
                lambda_ = scale * 2 / np.pi
                psi = 0
                gamma = 1

                w[i, 0, :, :] = torch.Tensor(cv2.getGaborKernel((self.filt_size, self.filt_size), sigma, theta, lambda_, gamma, psi))
                i += 1
    
        return w



    
    
def filters(filter_type:str,
            in_channels:int = None,
            curv_params:dict = None,
            kernel_size:int = None):

    """
    Returns the filters given the filter type
    
    Arguments
    ----------  
    
    filter_type: 
        The type of filters to use. There is currently only one type (curvature) but others, such as gabor filters can be added.
    

    in_channels:
        number of input channels

    curv_params:
        parameters for the curvature model

    kernel_size:
        size of the kernels
 
    """
        
    assert filter_type in ['curvature', 'gabor'], "the only available filter type is curvature"

    if filter_type == 'curvature':

        curve = CurvatureFilters(
            in_channels=in_channels,
            n_ories=curv_params['n_ories'],
            gau_sizes=curv_params['gau_sizes'],
            curves=np.logspace(-2, -0.1, curv_params['n_curves']),
            fre = curv_params['spatial_fre'],
            filt_size=kernel_size)
        return curve()
    
    elif filter_type == 'gabor':
        gabor = GaborFilters(
                 in_channels=in_channels,
                 n_ories=curv_params['n_ories'],
                 filt_size=kernel_size)
        return gabor()
        