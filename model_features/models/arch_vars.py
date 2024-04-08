import torch
import os
import sys
sys.path.append(os.getenv('MODELS_ROOT_PATH'))
from expansion import Expansion5L


FILTERS_1_DICT = {
                'curvature':
                    {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                'gabor':
                    {'n_ories':12,'num_scales':3},
                'random':
                    {'filters':30}
               }
                



device = 'cuda'

#layer 1 vars
filters_1_type = 'curvature' # can do one of ['curvature','gabor','random']
filters_1_params = FILTERS_1_DICT[filters_1_type]

#layer 2 vars
filters_2=1000
#layer 3 vars
filters_3=3000
#layer 4 vars
filters_4=5000
#layer 5 vars
filters_5=[3, 30,3000,30000] # VARIABLE

# other vars
init_type = ['kaiming_uniform', 'uniform'] # VARIABLE
non_linearity= ['relu', 'elu','none'] # VARIABLE
gpool=False # if there is global max pooling on for the model output, default (False) makes use of spatial information


expansion_model = Expansion5L(filters_1_type=filters_1_type,
                              filters_1_params= filters_1_params,
                              filters_2=filters_2,
                              filters_3=filters_3,
                              filters_4=filters_4,
                              filters_5=filters_5,
                              init_type= init_type,
                              gpool=gpool,
                              non_linearity=non_linearity,
                              device=device).Build()

image = torch.rand(1,3,224,224)
features = expansion_model(image.to(device))
print(features.shape)