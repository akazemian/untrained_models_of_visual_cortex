import os
import logging
from pathlib import Path
import pickle
from dotenv import load_dotenv
load_dotenv()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
# env paths
DATA = os.getenv("DATA")
CACHE = os.getenv("CACHE")

# root path
ROOT = Path.cwd()

# results
PREDS_PATH = os.path.join(CACHE,'neural_preds')
RESULTS_PATH = ROOT / 'results'

# neural data
MAJAJ_DATA = os.path.join(DATA,'majajhong')
MAJAJ_TRAIN_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids'), "rb"))
MAJAJ_TEST_IDS =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids'), "rb"))
TRAIN_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_train_ids_demo'), "rb"))
TEST_IDS_DEMO =  pickle.load(open(os.path.join(MAJAJ_DATA,'majaj_test_ids_demo'), "rb"))

NSD_NEURAL_DATA = os.path.join(DATA,'naturalscenes')

# benchmark variables
ALPHA_RANGE = [10**i for i in range(-10, 10)]




# # where everything will be cached
# CACHE = '/data/atlas/.repo_cache_10'
# DATA = '/data/atlas/repo_data'
# PREDS_PATH = os.path.join(CACHE,'neural_preds')


# # naturalscenes data
# NSD_SAMPLE_IMAGES = '/data/atlas/neural_data/naturalscenes/sample_ids.pkl'
# NSD_IMAGES = os.path.join(DATA,'naturalscenes','images')
# NSD_NEURAL_DATA = os.path.join(DATA,'naturalscenes')#NSD_NEURAL_DATA = '/data/shared/for_atlas'

            
# #majaj neural data
# MAJAJ_IMAGES = os.path.join(DATA,'majajhong','image_dicarlo_hvm-public')
# MAJAJ_DATA = os.path.join(DATA,'majajhong')
# MAJAJ_NAME_DICT = os.path.join(DATA,'majajhong','image_dicarlo_hvm-public.csv')

# # places dataset for image classification
# PLACES_IMAGES = '/data/atlas/datasets/places' # places 




