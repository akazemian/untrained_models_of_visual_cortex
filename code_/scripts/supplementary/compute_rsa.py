import os
import pickle
import argparse
from code_.encoding_score.rsa import compute_rsa_majajhong, compute_rsa_nsd
from code_.model_activations.loading import load_full_identifier
from config import FIGURES, RESULTS
from code_.model_activations.configs import model_cfg as cfg
from code_.tools.utils import timeit
import argparse 


@timeit
def main():
    parser = argparse.ArgumentParser(
        description="Compute encoding scores for all regions in a dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'majajhong')"
    )

    parser.add_argument(
        "--rsa_metric",
        type=str,
        default='pearsonr',
    )

    parser.add_argument(
        "--rdm_metric",
        type=str,
        default='euclidean',
    )

    args = parser.parse_args()


    for region in cfg[args.dataset]['regions']:

        rsa_dict = {}
        models = ['expansion']#,'fully_connected','vit']

        for model_name in models:
            print(f'computing RSA for model: {model_name}')
            rsa_dict[model_name] = []
            model_idens = []
            for features in cfg[args.dataset]['models'][model_name]['features']:
                activation_iden = load_full_identifier(model_name=model_name, 
                                    features = features, 
                                    layers=cfg[args.dataset]['models'][model_name]['layers'], 
                                    dataset=args.dataset)
                model_idens.append(activation_iden) 
            match args.dataset:
                case 'majajhong':
                    for iden in model_idens:
                        rsa_dict[model_name].append(compute_rsa_majajhong(iden, region, args.rdm_metric, args.rsa_metric, demo=False))
                case 'naturalscenes':
                    for iden in model_idens:
                        rsa_dict[model_name].append(compute_rsa_nsd(iden, region, args.rdm_metric, args.rsa_metric))

        with open(os.path.join(RESULTS, f'rsa_rdm_metric={args.rdm_metric}_{args.dataset}_{region}'), 'wb') as f:
            pickle.dump(rsa_dict, f)
    

if __name__ == "__main__":
    main()

            
            


