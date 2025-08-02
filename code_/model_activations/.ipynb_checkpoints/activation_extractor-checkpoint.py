import warnings
warnings.warn('my warning')
from collections import OrderedDict
from typing import Optional
import xarray as xr
import numpy as np
SUBMODULE_SEPARATOR = '.'
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
import pickle
import sys
import functools

from code_.tools.loading import load_image_paths, get_image_labels
from code_.tools.processing import ImageProcessor
from config import CACHE 
from code_.model_activations.utils import cache, register_pca_hook

PATH_TO_PCA = os.path.join(CACHE,'pca')



class PytorchWrapper:
    def __init__(self, model, identifier, device, forward_kwargs=None): 
        
        self._device = device
        self._model = model
        self._model.to(self._device)
        self._forward_kwargs = forward_kwargs or {}
        self.identifier = identifier


    def get_activations(self, images, layer_names, n_components, pca_iden):

        images = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results, n_components=n_components, pca_iden= pca_iden)
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
            
    def register_hook(self, layer: nn.Module, layer_name: str, target_dict: OrderedDict, 
                      n_components: Optional[int], pca_iden: Optional[str]) -> torch.utils.hooks.RemovableHandle:
        """
        Registers a forward hook on a layer to capture or transform its outputs during the forward pass.

        Args:
        layer (nn.Module): The layer to which the hook will be attached.
        layer_name (str): The name of the layer.
        target_dict (OrderedDict): Dictionary to store layer outputs.
        n_components (Optional[int]): Number of PCA components.
        pca_iden (Optional[str]): Identifier for where to get the PCs from.

        Returns:
        torch.utils.hooks.RemovableHandle: A handle that can be used to remove the hook.
        """
        def hook_function(_layer: nn.Module, _input: torch.Tensor, output: torch.Tensor, name: str = layer_name):    
            if pca_iden is not None:
                target_dict[name] = register_pca_hook(x=output, pca_file_name=pca_iden,
                                                      n_components=n_components, device=self._device)
            else:
                target_dict[name] = output

        hook = layer.register_forward_hook(hook_function)
        return hook 

# def register_hook(self, layer, layer_name, target_dict, _hook, n_components, pca_iden):
#     def hook_function(_layer, _input, output, _hook = _hook, name=layer_name):
        
#         if _hook is None:
#             target_dict[name] = output
            
#         elif _hook == 'pca':
#             target_dict[name] = register_pca_hook(output, os.path.join(PATH_TO_PCA, pca_iden), n_components=n_components)

#     hook = layer.register_forward_hook(hook_function)
#     return hook 


    
    def __repr__(self):
        return repr(self._model)    
    

    


def batch_activations(model: nn.Module, 
                      image_labels: list,
                      layer_names:list, 
                      pca_iden,
                      n_components:int,
                      device=str,
                      dataset=None,
                      image_paths: list=None,
                      images: torch.Tensor=None,
                      batch_size:int=None,) -> xr.Dataset:
    

            
        if image_paths is not None:
            images = ImageProcessor(device=device, batch_size=batch_size).process_batch(image_paths=image_paths,
                                                                                        dataset=dataset,
                                                                                       )

        
    
        activations_dict = model.get_activations(images = images, 
                                                 layer_names = layer_names, 
                                                n_components=n_components,
                                                pca_iden = pca_iden)
        activations_final = []
    
                             
        for layer in layer_names:
            activations_b = activations_dict[layer]
            activations_b = torch.Tensor(activations_b.reshape(activations_dict[layer].shape[0],-1))
            ds = xr.Dataset(
            data_vars=dict(x=(["presentation", "features"], activations_b.cpu())),
            coords={'stimulus_id': (['presentation'], image_labels)})

            activations_final.append(ds)     
        
        
        activations_final_all = xr.concat(activations_final,dim='presentation') 
        
        return activations_final_all

    

    
    
    
    
        
class Activations:
    
    def __init__(self,
                 model: nn.Module,
                 dataset: str,
                 layer_names: list=['last'],
                 pca_iden= None,
                 hook:str = None,
                 n_components=None,
                 device:str= 'cuda',
                 batch_size: int = 64,
                 compute_mode:str='fast',
                 subject_images=None):
        
        
        self.model = model
        self.layer_names = layer_names
        self.dataset = dataset
        self.batch_size = batch_size
        self.hook = hook
        self.pca_iden = pca_iden
        self.n_components = n_components
        self.device = device
        self.compute_mode = compute_mode
        self.subject_images = subject_images 
        
        assert self.compute_mode in ['fast','slow'], "invalid compute mode, please choose one of: 'fast', 'slow'"

        if not os.path.exists(os.path.join(CACHE,'activations')):
            os.mkdir(os.path.join(CACHE,'activations'))
     
        
    @staticmethod
    def cache_file(iden):
        return os.path.join('activations',iden)

    
    @cache(cache_file)
    def get_array(self,iden):       
                        
        print('extracting activations...')
        
        pytorch_model = PytorchWrapper(model = self.model, identifier = iden, device=self.device)
        images, labels = load_image_data(dataset_name=self.dataset, device=self.device)

        
        i = 0   
        ds_list = []
        pbar = tqdm(total = len(images)//self.batch_size)

        
        while i < len(images):

            batch_data_final = batch_activations(model=pytorch_model,
                                                images=images[i:i+self.batch_size, :],
                                                image_labels=labels[i:i+self.batch_size],
                                                layer_names = self.layer_names,
                                                pca_iden = self.pca_iden,
                                                n_components=self.n_components,
                                                device=self.device,
                                                batch_size=self.batch_size
                                                )

            ds_list.append(batch_data_final)    
            i += self.batch_size
            pbar.update(1)

        pbar.close()

            
        
        data = xr.concat(ds_list,dim='presentation')
        
        print('model activations are saved in cache')
        return data
    

    
    

def load_image_data(dataset_name:str, device:str):
    """
    Loads image paths and their corresponding labels for a given dataset.

    Args:
    dataset_name (str): The name of the dataset from which to load images. 

    Returns:
    tuple: A tuple containing:
           - image_paths (list[str]): A list of file paths corresponding to images.
           - image_labels (list[str]): A list of labels associated with the images. 
    """
    image_paths = load_image_paths(dataset_name=dataset_name)
    images = ImageProcessor(device=device).process(image_paths=image_paths, dataset=dataset_name)
    labels = get_image_labels(dataset_name = dataset_name, image_paths=image_paths)

    return images, labels

    
    
    
    