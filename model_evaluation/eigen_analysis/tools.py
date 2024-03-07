import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import seaborn as sns



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



def plot_eigspec(data, label, color):
      
    a, y = powerlaw_exponent(data)
    sns.lineplot(x=np.arange(1,len(data)+1),y=data/(10**y),label=label,c=color) #, alpha = {round(a,2)}
    #plt.bar(np.arange(1,len(data)+1),data,label=f'alpha = {round(a,2)}')
    plt.xscale('log')
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

