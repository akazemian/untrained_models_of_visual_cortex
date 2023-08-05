import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import torchvision 
import pandas as pd
import seaborn as sns
import os 

ROOT = os.getenv('MB_ROOT_PATH')
sys.path.append(ROOT)
DATA = os.getenv('MB_DATA_PATH')
MODEL_SCORES_PATH = os.path.join(DATA,'model_scores_final')
from tools.processing import *
from tools.loading import *
from analysis.encoding_model_analysis.tools.utils import get_activations_iden, get_scores_iden
from matplotlib import rcParams


# figure size in inches

def get_data(data_dict, dataset, regions, mode, subjects):
    
    df = pd.DataFrame()
    index = 0
    
    for model_name, model_info in data_dict.items():
        
            activations_iden = get_activations_iden(model_info, dataset, mode)

            for region in regions:

                scores_iden = get_scores_iden(model_info, activations_iden, region, dataset, mode)
                data = xr.open_dataset(os.path.join(MODEL_SCORES_PATH,activations_iden,scores_iden), engine='netcdf4')

                for subject in subjects:
                    subject_data = data.where(data.subject == subject, drop=True)
                    mean_r = np.mean(subject_data.r_value.values)

                    df_tmp =  pd.DataFrame({'score':mean_r,
                                            'model':model_name,
                                            'iden':model_info['iden'],
                                            'n_dims':model_info['n_dims'],
                                            'n_layers':model_info['num_layers'],
                                            'num_features':model_info['num_features'],
                                            'region':region,
                                            'subject':subject},index=[index])
                    df = pd.concat([df,df_tmp])
                    index+=1

    return df






def plot_subject_means(data_dict, dataset, regions, mode, palette, model_nums, opacity, ylim, height, aspect, size, params = (6,4), name_dict = None, file_name = None, show_legend = False):
  

    plt.clf()
    rcParams['figure.figsize'] = params
    
    if dataset == 'naturalscenes':
        subjects = [i for i in range(8)]
    elif dataset == 'majajhong':
        subjects = ['Tito','Chabo']
        
    
    df = get_data(data_dict, dataset, regions, mode, subjects)
    
    if name_dict is not None:
        df['iden'] = df['iden'].map(name_dict)
    
    if model_nums is None:
        df['model_num'] = 0
    else:
        df['model_num'] = df['iden'].map(model_nums)
    df['jitter'] = df['model_num'].apply(lambda x:  np.random.normal(x, 0.06))

    ax = sns.relplot(data = df, x = 'jitter',y = 'score', hue = 'region', alpha = opacity, s = size, height = height, aspect = aspect, palette = palette)
    ax = sns.boxplot(data = df, x = 'iden', y = 'score', color='white', showfliers = False)

    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=16) 
    plt.ylabel('Correlation (Pearson r)', fontsize=18)
    plt.ylim(ylim)
    ax.set(xlabel=None)
    #sns.set(rc={'figure.figsize':(12,12)})

    if file_name is not None:
        plt.savefig(f'{file_name}.png', dpi=300)

        
        
    

    
    

# figure size in inches

def plot_data_means_vs_features(data_dict, dataset, regions, mode, x_axis,  palette, ylim,
                                width, show_legend= True, error_bars = False, params = (6,4), 
                                name_dict= None, xlog=False, ylog=False, file_name=None):


    plt.clf()
    if dataset == 'naturalscenes':
        subjects = [i for i in range(8)]
    elif dataset == 'majajhong':
        subjects = ['Tito','Chabo']
        
    sns.set_context(context='talk')    
    
    rcParams['figure.figsize'] = params
        
        
    
    df = get_data(data_dict, dataset, regions, mode, subjects)
    
    if name_dict is not None:
        df['iden'] = df['iden'].map(name_dict)
    
    df[x_axis] = df[x_axis].apply(lambda x: str(x))



    #df[x_axis] = df[x_axis].apply(lambda x: str(x))
    ax1 = sns.barplot(x = df[x_axis], y = df['score'], hue=df.iden, palette = palette, 
                errorbar="sd",  width=width
                    )
                
        
    
    if show_legend:
        ax1.legend(fontsize=15,loc='upper left')
    else:
        ax1.get_legend().remove()
    
    if xlog:
        plt.xscale("log")
        
    if ylog:
        plt.yscale("log")
        
    plt.rc('xtick', labelsize=26) 
    plt.rc('ytick', labelsize=26) 
    plt.ylabel('Correlation (Pearson r)', fontsize=26)
    plt.xlabel('Number of Random Features', fontsize=26)
    plt.ylabel(ylabel='Correlation (Pearson r)')    
    
    plt.ylim(ylim)
    
    if file_name is not None:
        plt.savefig(f'{file_name}.png', dpi=300)
    

    
    

# figure size in inches

def plot_data_means_vs_regions(data_dicts, dataset, mode, x_axis,  palette, ylim,
                                width, show_legend= True, error_bars = False,
                                params = (6,4), name_dict= None, xlog=False, ylog=False, file_name=None):


    plt.clf()
    if dataset == 'naturalscenes':
        subjects = [i for i in range(8)]
    elif dataset == 'majajhong':
        subjects = ['Tito','Chabo']
        
    sns.set_context(context='talk')    
    
    rcParams['figure.figsize'] = params
    
    
    df = pd.DataFrame()
    for i in range(2):
        
        data_dict = data_dicts['data_dict'][i]
        regions = data_dicts['regions'][i]
        df_tmp = get_data(data_dict, dataset, regions, mode, subjects)
        df = pd.concat([df, df_tmp])
    
    
    
    df['region'] = df['region'].map({'V1':'V1-V4',
                                    'V2':'V1-V4',
                                    'V3':'V1-V4',
                                    'V4':'V1-V4',
                                    'general':'general'})
    
    df['iden'] = df['iden'].apply(lambda x: x.split('_conv')[0])
    df['iden'] = df['iden'].map(name_dict)


    ax1 = sns.barplot(x = df[x_axis], y = df['score'], hue=df['iden'], palette = palette, 
                errorbar="sd",  width=width
                    )
                    
    
    if show_legend:
        ax1.legend(fontsize=15,loc='upper right')
    else:
        ax1.get_legend().remove()
    
    if xlog:
        plt.xscale("log")
        
    if ylog:
        plt.yscale("log")
        
    plt.rc('xtick', labelsize=30) 
    plt.rc('ytick', labelsize=26) 
    plt.ylabel('Correlation (Pearson r)', fontsize=26)
    plt.xlabel('')
    plt.ylabel(ylabel='Correlation (Pearson r)')    
    
    plt.ylim(ylim)
    
    if file_name is not None:
        plt.savefig(f'{file_name}.png', dpi=300)
    

    
    


# figure size in inches

def plot_diff_models(data_dict, dataset, regions, mode, x_axis, ylim,
                                width, color,  palette=None, title = None, show_legend= True, error_bars = False, 
                                params = (6,4), name_dict= None, xlog=False, ylog=False, file_name=None):


    plt.clf()
    if dataset == 'naturalscenes':
        subjects = [i for i in range(8)]
    elif dataset == 'majajhong':
        subjects = ['Tito','Chabo']
        
    sns.set_context(context='talk')    
    rcParams['figure.figsize'] = params
            
    
    df = get_data(data_dict, dataset, regions, mode, subjects)
    if name_dict is not None:
        df['iden'] = df['iden'].map(name_dict)
    df[x_axis] = df[x_axis].apply(lambda x: str(x))



    if color is not None:
        ax1 = sns.barplot(x = df.iden, y = df['score'], 
                    errorbar="sd",  width=width, color=color,
                    )
    else:    
        ax1 = sns.barplot(x = df.iden, y = df['score'], hue = df.iden, errorbar="sd",  width=width, palette=palette, dodge=False)            
    
    
    if show_legend:
        ax1.legend(fontsize=20,loc='upper center',ncol=3)
    else:
        ax1.legend().remove()
    
    if xlog:
        plt.xscale("log")
        
    if ylog:
        plt.yscale("log")
        
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16) 
    plt.ylabel('Correlation (Pearson r)', fontsize=18)
    plt.xlabel('')
    plt.ylabel(size=25, ylabel='Correlation (Pearson r)')    
    plt.xticks(size=25)
    plt.yticks(size=25)
    plt.ylim(ylim)
    if title is not None:
        plt.title(title, fontsize=25)
    
    if file_name is not None:
        plt.savefig(f'{file_name}.png', dpi=300)
    

    
    


def plot_diff_layers(data_dict, dataset, region, mode, x_axis, ylim,
                                width, show_legend= True, error_bars = False,
                                params = (6,4), baseline_dict = None, baseline_palette = None,
                                name_dict= None, xlog=False, ylog=False, file_name=None):


    plt.clf()
    if dataset == 'naturalscenes':
        subjects = [i for i in range(8)]
    elif dataset == 'majajhong':
        subjects = ['Tito','Chabo']
        
    sns.set_context(context='talk')    
    
    rcParams['figure.figsize'] = params
    
        
        
    
    df = get_data(data_dict, dataset, regions, mode, subjects)
    if name_dict is not None:
        df['iden'] = df['iden'].map(name_dict)
    df[x_axis] = df[x_axis].apply(lambda x: str(x))

    ax1 = sns.barplot(x = df.n_layers, y = df['score'], errorbar="sd",  width=width
                    )
        
            
    if show_legend:
        ax1.legend(fontsize=15,loc='upper left')
    else:
        ax1.get_legend().remove()
    
    if xlog:
        plt.xscale("log")
        
    if ylog:
        plt.yscale("log")
        
    plt.rc('xtick', labelsize=16) 
    plt.rc('ytick', labelsize=16) 
    plt.ylabel('Correlation (Pearson r)', fontsize=18)
    plt.xlabel('')
    plt.ylabel(size=25,ylabel='Correlation (Pearson r)')    
    plt.xticks(size=25,rotation=15)
    plt.yticks(size=25)
    plt.ylim(ylim)
    
    if file_name is not None:
        plt.savefig(f'{file_name}.png', dpi=300)
    

    
    

    
