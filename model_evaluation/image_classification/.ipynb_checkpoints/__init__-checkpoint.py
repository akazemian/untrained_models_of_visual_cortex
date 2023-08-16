import sys
import os
sys.path.append(os.getenv('BONNER_ROOT_PATH'))
from config import CACHE

if not os.path.exists(CACHE):
    os.makedirs(CACHE)
    
if not os.path.exists(os.path.join(CACHE,'classification')):
    os.mkdir(os.path.join(CACHE,'classification'))    
