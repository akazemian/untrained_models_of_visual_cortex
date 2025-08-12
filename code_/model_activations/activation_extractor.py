import warnings
warnings.warn('my warning')
from collections import OrderedDict
from typing import Optional
import xarray as xr
import gc

import os
import time
import torch
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
    
# env paths
from config import CACHE
from code_.tools.loading import load_image_data
from code_.tools.processing import ImageProcessor
from code_.model_activations.utils import cache, register_pca_hook

SUBMODULE_SEPARATOR = '.'


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
            print(activations_b.shape)
            ds = xr.Dataset(
            data_vars=dict(x=(["presentation", "features"], activations_b.cpu())),
            coords={'stimulus_id': (['presentation'], image_labels)})

            activations_final.append(ds)     
            del ds
            gc.collect()
            torch.cuda.empty_cache()
        
        
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

        # start_time = time.time()
        # batch_times = []
        while i < len(images):
            
            # batch_start = time.time()
            batch_data_final = batch_activations(model=pytorch_model,
                                                images=images[i:i+self.batch_size, :],
                                                image_labels=labels[i:i+self.batch_size],
                                                layer_names = self.layer_names,
                                                pca_iden = self.pca_iden,
                                                n_components=self.n_components,
                                                device=self.device,
                                                batch_size=self.batch_size
                                                )
            # batch_end = time.time()
            # batch_duration = batch_end - batch_start
            # batch_times.append(batch_duration)
            # Calculate average time per batch

            # Print timing information
            # print(f"Batch {i} completed in {batch_duration:.2f} seconds")

            ds_list.append(batch_data_final)    
            i += self.batch_size
            pbar.update(1)
            
            del batch_data_final
            gc.collect()
            torch.cuda.empty_cache()

        del images, labels
        gc.collect()
        torch.cuda.empty_cache()
        pbar.close()
        # Print total elapsed time
        # average_time_per_batch = (time.time() - start_time)/len(batch_times)
        # print(f"\n average processingtime: {average_time_per_batch:.2f} seconds.")

        data = xr.concat(ds_list,dim='presentation')
        return data