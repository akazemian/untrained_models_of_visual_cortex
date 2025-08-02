from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.model_activations.loading import load_model, load_full_identifier
from code_.eigen_analysis.utils import _PCA
from code_.model_activations.activation_extractor import Activations
from config import setup_logging
import logging
import argparse
from code_.tools.utils import timeit, setup_logging

setup_logging()


@timeit
def main():
    parser = argparse.ArgumentParser(
        description="Compute encoding scores for all regions in a dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='expansion',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'majajhong')"
    )
    parser.add_argument(
        "--batch_size",
        default=5,
        type=str,
        help="Batch size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu')"
    )
    args = parser.parse_args()
    TOTAL_COMPONENTS = 1000
    N_COMPONENTS = [10, 100, 1000]
    FEATURES = 30000
    LAYERS = 5

    compute_model_pcs(model_name=args.model, 
                    features=FEATURES, 
                    layers=LAYERS, 
                    batch_size=args.batch_size,
                    dataset=args.dataset, 
                    components=TOTAL_COMPONENTS, 
                    device=args.device)

    # project activations onto the computed PCs 
    for n_components in N_COMPONENTS:
        
        pca_identifier = load_full_identifier(model_name=args.model, 
                                                        features=FEATURES, 
                                                        layers=LAYERS, 
                                                        dataset=args.dataset,
                                                        principal_components = TOTAL_COMPONENTS)
        
        activations_identifier = load_full_identifier(model_name=args.model, 
                                                features=FEATURES, 
                                                layers=LAYERS, 
                                                dataset=args.device,
                                                principal_components = n_components)            
        
        
        logging.info(f"Extracting activations and projecting onto the first {n_components} PCs")
        
        #load model
        model = load_model(model_name=args.model, 
                        features=FEATURES, 
                            layers=LAYERS,
                            device=args.device)

        # compute activations and project onto PCs
        Activations(model=model, 
                    dataset=args.dataset, 
                    pca_iden = pca_identifier,
                    n_components = n_components, 
                    batch_size = args.batch_size,
                    device= args.device).get_array(activations_identifier)  


if __name__ == "__main__":
    main()


            
            