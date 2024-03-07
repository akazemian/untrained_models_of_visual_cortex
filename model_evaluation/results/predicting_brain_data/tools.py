import os
import sys
ROOT = os.getenv('BONNER_ROOT_PATH')
sys.path.append(ROOT)
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


from model_evaluation.predicting_brain_data.benchmarks.majajhong import load_majaj_data
from model_evaluation.predicting_brain_data.benchmarks.nsd import load_nsd_data
from model_evaluation.predicting_brain_data.benchmarks.nsd import filter_activations
from model_features.models.models import load_full_iden
from model_evaluation.predicting_brain_data.regression.regression import pearson_r
from config import CACHE, NSD_NEURAL_DATA      



SHARED_IDS_PATH = os.path.join(ROOT, 'image_tools','nsd_ids_shared')
SHARED_IDS = pickle.load(open(SHARED_IDS_PATH, 'rb'))
SHARED_IDS = [image_id.strip('.png') for image_id in SHARED_IDS]
PREDS_PATH = '/data/atlas/.cache/beta_predictions'
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





def get_bootstrap_data(models, features, layers, subjects, dataset, region, all_sampled_indices, file_name,
                       init_types=['kaiming_uniform'], nl_types=['relu'], principal_components=[None],
                       l1_random_filters=[None], batch_size=50, n_bootstraps=1000):
    
    data_dict = {'model': [], 'features': [], 'l1_random_filters': [], 'pcs': [], 'init_type': [], 'nl_type': [],
                 'score': [], 'lower': [], 'upper': []}
    
    for feature in features:
        for random_filter in l1_random_filters:
            for component in principal_components:
                for model_name in models:
                    for nonlinearity in nl_types:
                        for initializer in init_types:
                            try:
                                identifier = load_full_iden(model_name, feature, layers, random_filter, dataset,
                                                                 component, nonlinearity, initializer)
                                bootstrap_distribution = compute_bootstrap_distribution(identifier, subjects, region,
                                                                                        all_sampled_indices, batch_size,
                                                                                        n_bootstraps, dataset)
                                update_data_dict(data_dict, model_name, feature, random_filter, component, initializer,
                                                 nonlinearity, bootstrap_distribution)
                            except FileNotFoundError:
                                print(f'File not found: {identifier}, region: {region}')

    df = pd.DataFrame.from_dict(data_dict)
    save_results(df, file_name, dataset, region)
    return df



def compute_bootstrap_distribution(identifier, subjects, region, all_sampled_indices, batch_size, n_bootstraps, dataset):
    score_sum = np.zeros(n_bootstraps)
    for subject in tqdm(subjects):
        preds, test = load_data(identifier, region, subject, dataset)
        all_sampled_preds = preds[all_sampled_indices]
        all_sampled_tests = test[all_sampled_indices]
        score_sum += batch_pearson_r(all_sampled_tests, all_sampled_preds, batch_size, n_bootstraps)
    return score_sum / len(subjects)



def batch_pearson_r(all_sampled_tests, all_sampled_preds, batch_size, n_bootstraps):
    r_values = []
    i = 0
    while i < n_bootstraps:
        # Compute Pearson r for all bootstraps at once
        mean_r_values = pearson_r_(all_sampled_tests[i:i + batch_size, :, :].cuda(),
                                   all_sampled_preds[i:i + batch_size, :, :].cuda())
        r_values.extend(mean_r_values.tolist())
        i += batch_size
    return np.array(r_values)


def load_data(identifier, region, subject, dataset):
    with open(os.path.join(PREDS_PATH, f'{identifier}_{region}_{subject}.pkl'), 'rb') as file:
        preds = torch.Tensor(pickle.load(file))

    if 'naturalscenes' in dataset:
        _, neural_data_test, var_name_test = load_nsd_data(mode='shared', subject=subject, region=region)
        test = torch.Tensor(neural_data_test[var_name_test].values)
    else:
        test = load_majaj_data(subject, region, 'test')
    return preds, test

def update_data_dict(data_dict, model_name, feature, random_filter, component, initializer, nonlinearity, bootstrap_dist):
    data_dict['model'].append(model_name)
    data_dict['features'].append(str(feature))
    data_dict['l1_random_filters'].append(random_filter)
    data_dict['pcs'].append(str(component))
    data_dict['init_type'].append(initializer)
    data_dict['nl_type'].append(nonlinearity)
    data_dict['score'].append(np.mean(bootstrap_dist))
    data_dict['lower'].append(np.percentile(bootstrap_dist, 2.5))
    data_dict['upper'].append(np.percentile(bootstrap_dist, 97.5))
    
    
def save_results(df, file_name, dataset, region):
    with open(os.path.join(BOOTSTRAP_RESULTS_PATH, f'bootstrap-results-{file_name}-{dataset}-{region}-df.pkl'), 'wb') as file:
        pickle.dump(df, file)

        
# def get_bootstrap_data(models, features,  layers, subjects,
#                        dataset, region, all_sampled_indices, file_name, 
#                        init_types=['kaiming_uniform'], nl_types=['relu'],
#                        principal_components = [None], l1_random_filters=[None], batch_size=50, n_bootstraps=1000):

    
#     data_dict = {'model':[],'features':[],'l1_random_filters':[],'pcs':[], 'init_type':[], 'nl_type':[],
#                  'score':[], 'lower':[],'upper':[]}
#     mean_scores = []
#     lower_bound = []
#     upper_bound = []              
        
    
#     for f in features:

#         for r_f in l1_random_filters:

#             for c in principal_components:
            
#                 for model_name in models:

#                     for nl in nl_types:

#                         for init in init_types:
        
#                             try:
                                    
#                                 activations_identifier = load_iden(model_name=model_name, features=f, layers=layers, random_filters=r_f, dataset=dataset)
#                                 if c is not None:
#                                     activations_identifier = activations_identifier + f'_principal_components={c}'
                                
#                                 if nl != 'relu':
#                                     activations_identifier = activations_identifier + '_' + nl
                                
#                                 if init != 'kaiming_uniform':
#                                     activations_identifier = activations_identifier + '_' + init
                                    
#                                 print(activations_identifier)
#                                 score_sum = np.zeros(n_bootstraps)
                
#                                 for s in tqdm(subjects):
                
#                                         # load preds and y_true
#                                         r_values = []
#                                         with open(os.path.join(PREDS_PATH,f'{activations_identifier}_{region}_{s}.pkl'), 'rb') as file:
#                                             preds = torch.Tensor(pickle.load(file))
                
#                                         if 'naturalscenes' in dataset:
#                                             ids_test, neural_data_test, var_name_test = load_nsd_data(mode ='shared', subject = s, region = region)           
#                                             test = torch.Tensor(neural_data_test[var_name_test].values)
            
#                                         else:
#                                             test = load_majaj_data(s, region, 'test')
                                            
                                        
#                                         # Vectorized bootstrapping
#                                         all_sampled_preds = preds[all_sampled_indices]
#                                         all_sampled_tests = test[all_sampled_indices]
                
#                                         i = 0
#                                         batch = batch_size
#                                         while i < n_bootstraps:
#                                             # Compute Pearson r for all bootstraps at once
#                                             mean_r_values = pearson_r_(all_sampled_tests[i:i+batch,:,:].cuda(), 
#                                                                        all_sampled_preds[i:i+batch,:,:].cuda())
#                                             r_values.extend(mean_r_values.tolist())
#                                             i += batch
                
#                                         score_sum += r_values
                
#                                 bootstrap_dist = score_sum/len(subjects)
                
#                                 data_dict['model'].append(model_name)
#                                 data_dict['features'].append(str(f))
#                                 data_dict['l1_random_filters'].append(r_f)
#                                 data_dict['pcs'].append(str(c))
#                                 data_dict['init_type'].append(init)
#                                 data_dict['nl_type'].append(nl)
#                                 data_dict['score'].append(np.mean(bootstrap_dist))
#                                 data_dict['lower'].append(np.percentile(bootstrap_dist, 2.5))
#                                 data_dict['upper'].append(np.percentile(bootstrap_dist, 97.5))
            
#                             except FileNotFoundError:
#                                 print('FILE NOT FOUND',activations_identifier,region)
#                                 pass
        
#     df = pd.DataFrame.from_dict(data_dict)
        
#     with open(os.path.join(BOOTSTRAP_RESULTS_PATH,f'bootstrap-results-{file_name}-{dataset}-{region}-df.pkl'), 'wb') as file:
#         pickle.dump(df,file)
            
#     return df