import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import torchvision 
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import os 
import sys
sys.path.append(os.getenv('BONNER_ROOT_PATH'))

from image_tools.processing import *
from image_tools.loading import *
from model_evaluation.utils import get_activations_iden
from config import CACHE



def make_pandas_df(data_dict, dataset, regions, subjects):
    
    df = pd.DataFrame()
    index = 0
    
    for model_name, model_info in data_dict.items():
        
            activations_iden = get_activations_iden(model_info, dataset)

            for region in regions:

                scores_iden = activations_iden + '_' + region
                data = xr.open_dataset(os.path.join(CACHE,'encoding_scores',scores_iden), engine='h5netcdf')

                for subject in subjects:
                    subject_data = data.where(data.subject == subject, drop=True)
                    mean_r = np.mean(subject_data.r_value.values)

                    df_tmp =  pd.DataFrame({'score':mean_r,
                                            'model':model_name,
                                            'iden':model_info['iden'],
                                            'n_layers':model_info['num_layers'],
                                            'num_features':model_info['num_features'],
                                            'region':region,
                                            'subject':subject},index=[index])
                    df = pd.concat([df,df_tmp])
                    index+=1

    return df



    

def scores_vs_num_features(df, x_axis, palette, width):

    return sns.barplot(x = df[x_axis], 
                       y = df['score'], 
                       hue=df.iden, 
                       palette = palette, 
                       errorbar="sd",  
                       width=width)

    

def compare_models(df, color, palette, width):

    
    if color is not None:
        return sns.barplot(x = df.iden, 
                           y = df['score'],
                           errorbar="sd",
                           width=width,
                           color=color
                          )
    else:    
        return sns.barplot(x = df.iden, 
                           y = df['score'], 
                           hue = df.iden, 
                           errorbar="sd",  
                           width=width, 
                           palette=palette, 
                           dodge=False)           
        
            

    

    
def plot_results(data_dict, plot_type, dataset, regions, 
                 ylim, width, 
                 x_axis=None, 
                 palette=None,
                 color = None,
                 show_legend= True, 
                 params = (6,4), 
                 name_dict= None, 
                 file_name=None):    
    
    assert plot_type in ['scores_vs_num_features','compare_models'], f"choose one of {['scores_vs_num_features','compare_models']} as the plot type"
    
    plt.clf()
    
    if dataset == 'naturalscenes':
        subjects = [i for i in range(8)]
    
    elif dataset == 'majajhong':
        subjects = ['Tito','Chabo']
        
    sns.set_context(context='talk')    
    
    rcParams['figure.figsize'] = params        
    
    df = make_pandas_df(data_dict, dataset, regions, subjects)
    
    if name_dict is not None:
        df['iden'] = df['iden'].map(name_dict)
    
    if x_axis is not None:
        df[x_axis] = df[x_axis].apply(lambda x: str(x))
    
    match plot_type:
        case 'scores_vs_num_features':
            ax1 = scores_vs_num_features(df, x_axis, palette, width)
        case 'compare_models':
            ax1 = compare_models(df, color, palette, width)
            
            
    if show_legend:
        ax1.legend(fontsize=30,loc='upper left')
    else:
        ax1.get_legend().remove()
        
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16) 
    plt.ylabel('Correlation (Pearson r)', fontsize=18)
    plt.xlabel('')
    plt.ylabel(size=25,ylabel='Correlation (Pearson r)')    
    plt.xticks(size=25)
    plt.yticks(size=25)
    plt.ylim(ylim)
    
    if file_name is not None:
        plt.savefig(f'{file_name}.png', bbox_inches='tight', dpi=300, transparent=True) 
        

    
