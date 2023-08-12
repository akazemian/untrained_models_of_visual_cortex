import xarray as xr
from sklearn.decomposition import PCA
import functools
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE




def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) 
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                return 
            
            result = func(self, *args, **kwargs)
            with open(cache_path,'wb') as f:
                pickle.dump(result, f,  protocol=4)
            return 
        
        return wrapper
    return decorator




class _PCA:
    
    def __init__(self,
                 n_components:int,
                 device:str = 'cuda'):
        
        self.n_components = n_components
        self.device = device
        
        if not os.path.exists(os.path.join(CACHE,'pca')):
            os.mkdir(os.path.join(CACHE,'pca'))
     
        
    @staticmethod
    def cache_file(iden):
        return os.path.join('pca',iden)

    
    @cache(cache_file)
    def _fit(self, iden):  
   
        X = xr.open_dataset(os.path.join(CACHE,iden)).x.values
        pca = PCA(self.n_components)
        pca.fit(X)
        
        return pca
