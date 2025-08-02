from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.model_activations.configs import model_cfg as cfg


models = ['vit','fully_connected','expansion']
dataset = 'naturalscenes'
dataset = 'majajhong'
device = 'cuda'
components = 2000

for model_name in models:

    # if model_name == 'vit':
    #     layers = None
    # elif model_name == 'expansion':
    #     layers = 5
    # else:
    #      layers = 5
    for f in cfg[dataset]['models'][model_name]['features']:
        print(model_name, f)
        compute_model_pcs(model_name=model_name,
                        features=f, 
                        layers=cfg[dataset]['models'][model_name]['layers'], 
                        batch_size=10,
                        dataset=dataset, 
                        components=components, 
                        device=device,
                        incremental=False)