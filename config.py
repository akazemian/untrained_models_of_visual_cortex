import os
import logging
from pathlib import Path
import os
import pickle
from pathlib import Path
# ----------------------------- SET THESE PATHS ----------------------------- 
DATA = '/data/atlas/repo_data'
# CACHE = '/data/atlas/expansion_cache'
CACHE = '/data/atlas/.repo_cache_10'

#---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PREDS_PATH = os.path.join(CACHE,'neural_preds')
RESULTS = os.path.join(ROOT, 'results')
PCA_PATH = os.path.join(CACHE,'pca')
FIGURES = os.path.join(ROOT, 'figures', 'main figures')

# ----------------------------- Dataset Paths  ----------------------------- 

# monkey ephys (majajhong) 
MAJAJ_DATA = os.path.join(DATA,'majajhong')
MAJAJ_RAW_DATA = os.path.join(MAJAJ_DATA, 'assy_dicarlo_MajajHong2015_public.nc') # full data with repeats for noise ceiling calculations
MAJAJ_TRAIN_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids'), "rb"))
MAJAJ_TEST_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids'), "rb"))
TRAIN_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids_demo'), "rb"))
TEST_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids_demo'), "rb"))
MAJAJ_IMAGES = os.path.join(MAJAJ_DATA,'image_dicarlo_hvm-public')

# naturalscenes fMRI (nsd)
NSD_NEURAL_DATA = os.path.join(DATA,'naturalscenes')
NSD_ANNOTS = os.path.join(NSD_NEURAL_DATA, 'annotations.nc')
NSD_NC_DATA = os.path.join(NSD_NEURAL_DATA, 'noise_ceilings', 'fithrf_GLMdenoise_RR') 
NSD_SAMPLE_IMAGES = os.path.join(NSD_NEURAL_DATA,'sample_ids.pkl')
NSD_IMAGES = os.path.join(NSD_NEURAL_DATA,'images')

# THINGS fMRI
THINGS_DATA = DATA
THINGS_IMAGES = os.path.join(DATA,'hebart2019.things', 'images')
THINGS_TRAIN_IDS = pickle.load(open(os.path.join(DATA,'things_train_ids'), "rb"))
THINGS_TEST_IDS = pickle.load(open(os.path.join(DATA,'things_test_ids'), "rb"))


# Places dataset for image classification 
PLACES_IMAGES = '/data/atlas/datasets/places' # places 

# ----------------------------- Variables  ----------------------------- 

# lapha range for ridge regression
ALPHA_RANGE = [10**i for i in range(-10, 10)]




