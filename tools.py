
import functools
import os
from joblib import dump, load
from config import CACHE


    
    
def cache(file_name_func):

    def decorator(func):
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            file_name = file_name_func(*args, **kwargs) + '.joblib'
            cache_path = os.path.join(CACHE, file_name)
            
            if os.path.exists(cache_path):
                return load(cache_path)
            
            result = func(self, *args, **kwargs)
            dump(result, cache_path)
            return result

        return wrapper
    return decorator

