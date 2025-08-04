import numpy as np
import os
import argparse
import os
import xarray as xr
import pickle

from config import RESULTS
from code_.tools.loading import load_places_cat_labels, load_image_paths, get_image_labels
from code_.model_activations.activation_extractor import Activations
from code_.model_activations.loading import load_model, load_full_identifier
from code_.image_classification.tools import get_Xy, cv_performance_demo
from code_.model_activations.configs import analysis_cfg as cfg     
from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.tools.utils import timeit, setup_logging
from dotenv import load_dotenv

load_dotenv()

CACHE = os.getenv("CACHE")

train_data = 'places_train_demo'
val_data = 'places_val_demo'

@timeit
def main():
        parser = argparse.ArgumentParser(
        description="Compute encoding scores for all regions in a dataset"
        )
        parser.add_argument(
        "--batch_size",
        type=str,
        default=16,
        )        
        parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., 'cuda' or 'cpu')"
        )
        args = parser.parse_args()

        TOTAL_COMPONENTS = n_components = 1000  

        models = ['expansion', 'alexnet_trained']

        for model_name in models:
        
                pca_identifier = load_full_identifier(model_name=model_name, 
                                                        features=cfg[train_data]['models'][model_name]['features'], 
                                                        layers=cfg[train_data]['models'][model_name]['layers'], 
                                                        dataset=train_data,
                                                        principal_components = TOTAL_COMPONENTS) 

                # compute model PCs using the train set
                if not os.path.exists(os.path.join(CACHE,'pca',pca_identifier)):
                        compute_model_pcs(model_name = model_name, 
                              features=cfg[train_data]['models'][model_name]['features'],  
                              layers=cfg[train_data]['models'][model_name]['layers'], 
                              dataset = train_data, 
                              components = TOTAL_COMPONENTS, 
                              device = args.device,
                              batch_size=int(args.batch_size))
            
                activations_identifier = load_full_identifier(model_name=model_name, 
                                                                features=cfg[train_data]['models'][model_name]['features'], 
                                                                layers=cfg[train_data]['models'][model_name]['layers'], 
                                                                dataset=train_data,
                                                                principal_components = n_components)   

                model = load_model(model_name=model_name, 
                                                features=cfg[val_data]['models'][model_name]['features'], 
                                                layers=cfg[val_data]['models'][model_name]['layers'],
                                                device=args.device)
                
                Activations(model=model,
                        layer_names=['last'],
                        dataset=val_data,
                        device= args.device,
                        hook='pca',
                        pca_iden = pca_identifier,
                        n_components=n_components,
                        batch_size = int(args.batch_size)).get_array(activations_identifier) 

                data = xr.open_dataset(os.path.join(CACHE,'activations',activations_identifier))
                data = data.set_xindex('stimulus_id')                

                # load demo subset labels
                cat_labels = load_places_cat_labels()
                image_paths = load_image_paths(val_data)
                demo_labels = get_image_labels(val_data, image_paths)
                demo_cat_labels = {image: cat for image, cat in cat_labels.items() if image in demo_labels} 
                
                X, y = get_Xy(data)
                score = cv_performance_demo(X, y, cat_labels=demo_cat_labels)
                print(activations_identifier, ':', score)

                with open(os.path.join(RESULTS, f'classification-{activations_identifier}'),'wb') as f:
                        pickle.dump(score,f)


if __name__ == "__main__":
        main()
