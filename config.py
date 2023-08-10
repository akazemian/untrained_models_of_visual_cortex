
import os

# naturalscenes data
NSD_IMAGES = '/data/shared/datasets/allen2021.natural_scenes/images'
NSD_NEURAL_DATA = '/data/shared/for_atlas'

# places dataset for image classification
PLACES_IMAGES = '/data/atlas/datasets/places' # places 



CACHE_DIR = '/data/atlas'

CACHE = os.path.join(CACHE_DIR,'.cache')
#PROCESSED_IMAGES_PATH = os.path.join(CACHE_PATH,'processed_images')
ACTIVATIONS_PATH = os.path.join(CACHE,'activations')
ENCODING_SCORES_PATH = os.path.join(CACHE,'encoding_scores')
