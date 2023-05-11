
import warnings
warnings.warn('my warning')
from collections import OrderedDict
import xarray as xr
import numpy as np
import tables
SUBMODULE_SEPARATOR = '.'
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
from tools.loading import *
from torch import nn
import pickle
from models.layer_operations.convolution import RandomProjections

ROOT = os.getenv('MB_DATA_PATH')
PATH_TO_PCA = os.path.join(ROOT,'pca')






class PytorchWrapper:
    def __init__(self, model,forward_kwargs=None): #preprocessing=None, identifier=None,  *args, **kwargs

        #logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #logger.debug(f"Using device {self._device}")
        self._model = model
        self._model = self._model.to(self._device)
        self._forward_kwargs = forward_kwargs or {}


    def get_activations(self, images, layer_names):

        images = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        with torch.no_grad():
            self._model(images, **self._forward_kwargs)
        for hook in hooks:
            hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        if layer_name == 'logits':
            return self._output_layer()
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def _output_layer(self):
        module = self._model
        while module._modules:
            module = module._modules[next(reversed(module._modules))]
        return module

    @classmethod
    def _tensor_to_numpy(cls, output):
        try:
            return output.cpu().data.numpy()
        except AttributeError:
            return output
            

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = PytorchWrapper._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def __repr__(self):
        return repr(self._model)
    
    

    
    
    
    
   
    

    
def batch_activations(model: nn.Module, 
                      layer_names: list, 
                      images: torch.Tensor,
                      image_labels: list,
                      max_pool: bool,
                      random_proj: bool,
                      pca_type: str,
                     n_components: int) -> xr.Dataset:

        
        activations_dict = model.get_activations(images = images, layer_names = layer_names)
        activations_final = []
    
        
        for layer in layer_names:
                             
            
            activations_b = activations_dict[layer]
            
            # if max_pool :
            #     mp = nn.MaxPool2d(activations_dict[layer].shape[-1]) 
            #     activations_b = mp(torch.Tensor(activations_b))
                
            # if pca_type == ['spca']:
            #     ds = xr.Dataset(
            #         data_vars=dict(x=(["presentation", "channels", "h", "w"], np.array(activations_b))),
            #         coords={'stimulus_id': (['presentation'], image_labels)})
            
            #else:
            activations_b = activations_b.reshape(activations_dict[layer].shape[0],-1)
            ds = xr.Dataset(
            data_vars=dict(x=(["presentation", "features"], np.array(activations_b))),
            coords={'stimulus_id': (['presentation'], image_labels)})
            
            activations_final.append(ds)     
        
        
        activations_final_all = xr.concat(activations_final,dim='presentation') 
        
        return activations_final_all

    

    
    
    
    
        
class Activations:
    
    def __init__(self,
                 model: nn.Module,
                 layer_names: list,
                 dataset: str,
                 preprocess,
                 mode: str,
                 max_pool: bool = False,
                 pca_type: str = None,
                 n_components: int = None,
                 random_proj: bool = False,
                 batch_size: int = 100):
        
        self.model = model
        self.layer_names = layer_names
        self.dataset = dataset
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.max_pool = max_pool 
        self.pca_type = pca_type
        self.random_proj = random_proj
        self.mode = mode
        self.n_components = n_components
     
        
    def get_array(self,path,identifier):
        
        if os.path.exists(os.path.join(path,identifier)):
            print(f'array is already saved in {path} as {identifier}')
        
        else:
        
            wrapped_model = PytorchWrapper(self.model)
            image_paths = LoadImagePaths(name = self.dataset, mode = self.mode)
            labels = get_image_labels(self.dataset, image_paths)  
            processed_images = self.preprocess(image_paths, self.dataset) 
            
    
            print('extracting activations')
            
            i = 0   
            ds_list = []
            pbar = tqdm(total = len(image_paths)//self.batch_size)
            
            while i < len(image_paths):
            
                batch_data_final = batch_activations(wrapped_model,
                                                     self.layer_names,
                                                     processed_images[i:i+self.batch_size],
                                                     labels[i:i+self.batch_size],
                                                     max_pool = self.max_pool,
                                                     pca_type = self.pca_type,
                                                     n_components = self.n_components,
                                                    random_proj = self.random_proj)
                    
                ds_list.append(batch_data_final)    
                i += self.batch_size
                pbar.update(1)
        
            pbar.close()
            
            data = xr.concat(ds_list,dim='presentation')
            data.to_netcdf(os.path.join(path,identifier))
            print(f'array is now saved in {path} as {identifier}')
    
            
        
        
        
        
            
            
def write_to_array(x, file_path):

    f = tables.open_file(file_path, mode='w')
    atom = tables.Float64Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0,) + x.shape[1:])

    array_c.append(x)
    f.close()
    
    return






def append_to_array(x, file_path):

    f = tables.open_file(file_path, mode='a')
    f.root.data.append(x)
    f.close()
    
    return
    
    






