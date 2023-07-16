from models.layer_operations.convolution import Convolution
from models.layer_operations.output import Output
from models.layer_operations.blurpool import BlurPool
from models.layer_operations.nonlinearity import NonLinearity
import torch
from torch import nn
from models.layer_operations.pca import NormalPCA
import os
import pickle
torch.manual_seed(0)
torch.cuda.manual_seed(0)


ROOT_DATA = os.getenv('MB_DATA_PATH')
PATH_TO_PCA = os.path.join(ROOT_DATA,'pca')
#IDEN = 'model_abs_3x3_bp_224_ap_mp_pca_5000_naturalscenes'                     
IDEN = 'expansion_model_pca_mp_pca_2000_imagenet21k_gs'

def load_pca_file(identifier):

    file = open(os.path.join(PATH_TO_PCA,identifier), 'rb')
    _pca = pickle.load(file)  
    file.close()
    return _pca




                         


class Model(nn.Module):
    
    
    def __init__(self,
                conv1: nn.Module,
                bpool1: nn.Module,
                pool1: nn.Module,
                conv2: nn.Module,
                bpool2: nn.Module,
                pool2: nn.Module,
                conv3: nn.Module,
                bpool3: nn.Module,
                pool3: nn.Module,
                batches_3: int,
                nl: nn.Module,
                gpool: bool,
                pca: nn.Module,
                last: nn.Module,
                print_shape: bool = True
                ):
        
        super(Model, self).__init__()
        
        
        self.conv1 = conv1 
        self.bpool1 = bpool1 
        self.pool1 = pool1
        
        self.conv2 = conv2
        self.bpool2 = bpool2
        self.pool2 = pool2
        
        self.conv3 = conv3
        self.bpool3 = bpool3
        self.pool3 = pool3
        self.batches_3 = batches_3
        
        self.pca = pca
        self.nl = nl
        self.gpool = gpool
        self.last = last
        self.print_shape = print_shape
        
        
    def forward(self, x:nn.Module): 
        
        
        #layer 1 
        x = self.conv1(x)  # conv 
        x = self.nl(x) # non linearity 
        x = self.bpool1(x) # anti-aliasing blurpool                
        x = self.pool1(x) # pool

            
        #layer 2
        x = self.conv2(x)  
        x = self.nl(x) 
        x = self.bpool2(x) 
        x = self.pool2(x)             
            
            
        #layer 3
        conv_3 = []
        for i in range(self.batches_3):
            conv_3.append(self.conv3(x))  
        x = torch.cat(conv_3) 
        x = self.bpool3(x) 
        x = self.pool3(x) 

        
        if self.gpool: # global pool
            H = x.shape[-1]
            gmp = nn.AvgPool2d(H) 
            x = gmp(x)

        
        x = self.pca(x)
        x = self.last(x) # final layer
        
        return x    
  

    
class ExpansionModelPCA:
        
    
    """
    Attributes
    ----------
    curv_params  
        pre-set curvature filter parameters 
    
    filters_2
        number of random filters in layer 2

        
    filters_3 
        number of random filters in layer 3
    
    batches_3
        number of batches used for layer 3 convolution. Used in case of memory issues 
        to perform convolution in batches. The number of output channles is equal to 
        filters_3 x batches_3
    
    bpool_filter_size
        kernel size for the anti aliasing operation (blurpool)

    gpool:
        whether global pooling is performed on the output of layer 3 
    """
    



    def __init__(self, 
                 curv_params:dict = {'n_ories':12,'n_curves':3,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2:int=2000,
                 filters_3:int=10000,
                 batches_3:int = 1,
                 init_type:str = 'kaiming_uniform',
                 bpool_filter_size:int=4,
                 gpool:bool=False,
                 n_components=5000):    
        
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
        self.filters_3 = filters_3
        self.batches_3 = batches_3
        self.init_type = init_type
        self.bpool_filter_size = bpool_filter_size
        self.gpool = gpool
        self.n_components = n_components
        self._pca = load_pca_file(IDEN)    
    
    def Build(self):        
        
        # layer 1
        conv1 = Convolution(filter_size=15,filter_type='curvature',curv_params=self.curv_params)     
        bpool1 = BlurPool(36, filt_size=self.bpool_filter_size, stride=2)
        pool1 = nn.AvgPool2d(kernel_size=3)

        
        # layer 2
        conv2 = nn.Conv2d(36, self.filters_2, kernel_size=(9, 9))
        #initialize_conv_layer(conv2, self.init_type)
        bpool2 = BlurPool(self.filters_2, filt_size=self.bpool_filter_size, stride=2)
        pool2 = nn.AvgPool2d(kernel_size=2,stride=1)
        
        
        # layer 3
        conv3 = nn.Conv2d(self.filters_2, self.filters_3, kernel_size=(7,7))
        #initialize_conv_layer(conv3, self.init_type)
        bpool3 = BlurPool(self.filters_3*self.batches_3, filt_size=self.bpool_filter_size, stride=2)
        pool3 = nn.AvgPool2d(kernel_size=2,stride=1)

        # non lineairy function
        nl = NonLinearity('abs')
        pca = NormalPCA(_pca = self._pca, n_components = self.n_components)

        #readout layer
        last = Output()
        

        
        return Model(
                conv1 = conv1,
                bpool1 = bpool1,
                pool1 = pool1,
                conv2 = conv2,
                bpool2 = bpool2,
                pool2 = pool2,
                conv3 = conv3,
                bpool3 = bpool3,
                pool3 = pool3,
                batches_3 = self.batches_3,
                nl = nl,
                pca = pca,
                gpool = self.gpool,
                last = last
        )
    



def initialize_conv_layer(conv_layer, initialization):
    if initialization == 'kaiming_uniform':
        nn.init.kaiming_uniform_(conv_layer.weight)
        nn.init.uniform_(conv_layer.bias)
    elif initialization == 'kaiming_normal':
        nn.init.kaiming_normal_(conv_layer.weight)
    else:
        raise ValueError(f"Unsupported initialization type: {initialization}. Please choose one of 'kaiming_uniform', 'kaiming_normal' ")