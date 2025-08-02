from code_.eigen_analysis.compute_pcs import compute_model_pcs
from code_.model_activations.models.configs import model_cfg as cfg


models = ['vit']
dataset = 'majajhong'
device = 'cuda'

for model_name in models:
    
    for f in cfg[dataset]['models'][model_name]['features']:
        if f*(6**2) < 1000:
            components = 100
        else:
            components = 2000
        
        compute_model_pcs(model_name=model_name,
                      features=f, 
                      layers=5, 
                      batch_size=10,
                      dataset=dataset, 
                      components=components, 
                      device=device)