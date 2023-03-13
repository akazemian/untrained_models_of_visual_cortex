from models.engineered_model import Model
from models.layer_operations.convolution import StandardConvolution,RandomProjections
from models.layer_operations.output import Output

class EngineeredModel:
    
    """
    Used to Initialize the Engineered Model
    
    Attributes
    ----------
    curv_params : dict
        the parameters used for creating the gabor filters. The number of filters in this layer = n_ories x n_curves x number of frequencies
    
    filters_2 : str
        number of random filters used in conv layer 2
    
    batches_2 : str 
        the number of batches used to apply conv layer 2 filters. Can be used for larger number of filters to avoid memory issues 
    """
    
    def __init__(self, curv_params = {'n_ories':8,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=20000,batches_2=1):
    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.batches_2 = batches_2
        
    
    
    
    def Build(self):
    
        c1 = StandardConvolution(filter_size=45,filter_type='curvature',pooling=('max',6),curv_params=self.curv_params)     
        c2 = StandardConvolution(out_channels=self.filters_2,filter_size=9,filter_type='random',pooling=('max',8))
        last = Output()

        return Model(c1,c2,self.batches_2,last)  
    