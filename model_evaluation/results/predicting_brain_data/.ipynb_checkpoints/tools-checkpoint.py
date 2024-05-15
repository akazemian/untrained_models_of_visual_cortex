
from image_tools.loading import load_image_paths, get_image_labels
from config import CACHE, NSD_SAMPLE_IMAGES
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import torch
import pickle
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import gc

from model_evaluation.predicting_brain_data.benchmarks.majajhong import load_majaj_data
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
from model_evaluation.predicting_brain_data.benchmarks.nsd import filter_activations
from model_features.models.utils import load_full_iden
from model_evaluation.predicting_brain_data.regression.regression import pearson_r
from config import CACHE, NSD_NEURAL_DATA      



SHARED_IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_shared')
SHARED_IDS = pickle.load(open(SHARED_IDS_PATH, 'rb'))
SHARED_IDS = [image_id.strip('.png') for image_id in SHARED_IDS]
PREDS_PATH = '/data/atlas/.cache/beta_predictions_new'
TEST_PATH = '/data/atlas/.cache/test_data'

BOOTSTRAP_RESULTS_PATH = '/home/akazemi3/Desktop/untrained_models_of_visual_cortex/model_evaluation/results/predicting_brain_data/bootstrap_data'




superscript_map = {
    "0": "\u2070",
    "1": "\u00B9",
    "2": "\u00B2",
    "3": "\u00B3",
    "4": "\u2074",
    "5": "\u2075",
    "6": "\u2076",
    "7": "\u2077",
    "8": "\u2078",
    "9": "\u2079",
}


def compute_similarity_matrix(features):
    """
    Compute the similarity matrix (using Pearson correlation) for a set of features.
    """
    # Compute the pairwise distances (using correlation) and convert to similarity
    return 1 - squareform(pdist(features, 'correlation'))


def rsa(features1, features2):
    """
    Perform Representational Similarity Analysis between two sets of features.
    """
    # Compute similarity matrices for both sets of features
    sim_matrix_1 = compute_similarity_matrix(features1)
    sim_matrix_2 = compute_similarity_matrix(features2)

    # Flatten the upper triangular part of the matrices
    upper_tri_indices = np.triu_indices_from(sim_matrix_1, k=1)
    sim_matrix_1_flat = sim_matrix_1[upper_tri_indices]
    sim_matrix_2_flat = sim_matrix_2[upper_tri_indices]

    # Compute the Spearman correlation between the flattened matrices
    correlation, p_value = spearmanr(sim_matrix_1_flat, sim_matrix_2_flat)

    return correlation, p_value


def pearson_r_(x, y):
    """
    Compute Pearson correlation coefficients for batches of bootstrap samples.

    Parameters:
    x (torch.Tensor): A 3D tensor of shape (n_bootstraps, n_samples, n_features).
    y (torch.Tensor): A 3D tensor of shape (n_bootstraps, n_samples, n_features).

    Returns:
    torch.Tensor: 1D tensor of Pearson correlation coefficients for each bootstrap.
    """
    # Ensure the input tensors are of the same shape
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")

    # Mean-centering the data
    x_mean = torch.mean(x, dim=2, keepdim=True)
    y_mean = torch.mean(y, dim=2, keepdim=True)
    x = x - x_mean
    y = y - y_mean

    # Calculating Pearson Correlation Coefficient
    sum_sq_x = torch.sum(x ** 2, axis=2)
    sum_sq_y = torch.sum(y ** 2, axis=2)
    sum_coproduct = torch.sum(x * y, axis=2)
    denominator = torch.sqrt(sum_sq_x * sum_sq_y)

    # Avoid division by zero
    denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))

    r_values = sum_coproduct / denominator

    # Average across the samples in each bootstrap
    mean_r_values = torch.mean(r_values, axis=1)

    return mean_r_values



def to_superscript(number):
    return ''.join(superscript_map.get(char, char) for char in str(number))



def write_powers(power):

    base = 10
    
    if power < 0:
        # For negative powers, use the superscript minus sign (⁻) followed by the power
        power_str = "\u207B" + to_superscript(abs(power))
    else:
        power_str = to_superscript(power)
    
    return f"{base}{power_str}"






def get_bootstrap_data(model_name, features, layers, subjects, dataset, region, all_sampled_indices, file_name,
                       init_types=['kaiming_uniform'], non_linearities=['relu'], principal_components=[None],
                       batch_size=3, n_bootstraps=1000, device='cuda'):
    
    data_dict = {'model': [], 'features': [], 'pcs': [], 'init_type': [], 'nl_type': [],
                 'score': [], 'lower': [], 'upper': []}
    
    for feature in features:
        for component in principal_components:
            for non_linearity in non_linearities:
                for init_type in init_types:
                    try:
                        identifier = load_full_iden(model_name, feature, layers, dataset,
                                                                 component, non_linearity, init_type)
                        bootstrap_dist = compute_bootstrap_distribution(identifier, subjects, region,
                                                                                        all_sampled_indices, batch_size,
                                                                                        n_bootstraps, dataset, device)
                        update_data_dict(data_dict, model_name, feature, component, init_type, non_linearity, bootstrap_dist)

                        del bootstrap_dist, identifier
                        gc.collect()
                                
                    except FileNotFoundError:
                        print(f'File not found: {identifier}, region: {region}')

    df = pd.DataFrame.from_dict(data_dict)
    save_results(df, file_name, dataset, region)
    return df



def compute_bootstrap_distribution(identifier, subjects, region, all_sampled_indices, batch_size, n_bootstraps, dataset, device):
    score_sum = torch.zeros(n_bootstraps).to(device)
    for subject in tqdm(subjects):
        preds, test = load_data(identifier, region, subject, dataset)
        all_sampled_preds = preds[all_sampled_indices]
        all_sampled_tests = test[all_sampled_indices]
        score_sum += batch_pearson_r(all_sampled_tests, all_sampled_preds, batch_size, n_bootstraps, device)
        
        del preds, test, all_sampled_preds, all_sampled_tests
        gc.collect()
        
    return score_sum / len(subjects)



def batch_pearson_r(all_sampled_tests, all_sampled_preds, batch_size, n_bootstraps, device):
    #r_values = []
    r_values = torch.Tensor([])
    i = 0
    while i < n_bootstraps:
        # Compute Pearson r for all bootstraps at once
        mean_r_values = pearson_r_(all_sampled_tests[i:i + batch_size, :, :].to(device),
                                   all_sampled_preds[i:i + batch_size, :, :].to(device))
        r_values = torch.concat((r_values.to(device), mean_r_values))
        #r_values.extend(mean_r_values.tolist())
        i += batch_size
        
        del mean_r_values
        gc.collect()
        
    return r_values



def load_data(identifier, region, subject, dataset):
    with open(os.path.join(PREDS_PATH, f'{identifier}_{region}_{subject}.pkl'), 'rb') as file:
        preds = torch.Tensor(pickle.load(file))

    if 'naturalscenes' in dataset:
        _, neural_data_test, var_name_test = load_nsd_data(mode='shared', subject=subject, region=region)
        test = torch.Tensor(neural_data_test[var_name_test].values)
    else:
        test = load_majaj_data(subject, region, 'test')
    return preds, test

def update_data_dict(data_dict, model_name, feature, component, init_type, non_linearity, bootstrap_dist):
    data_dict['model'].append(model_name)
    data_dict['features'].append(str(feature))
    data_dict['pcs'].append(str(component))
    data_dict['init_type'].append(init_type)
    data_dict['nl_type'].append(non_linearity)
    data_dict['score'].append(torch.mean(bootstrap_dist))
    data_dict['lower'].append(percentile(bootstrap_dist, 2.5))
    data_dict['upper'].append(percentile(bootstrap_dist, 97.5))
    
    
def save_results(df, file_name, dataset, region):
    with open(os.path.join(BOOTSTRAP_RESULTS_PATH, f'bootstrap-results-{file_name}-{dataset}-{region}-df.pkl'), 'wb') as file:
        pickle.dump(df, file)

def percentile(t, q):

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

