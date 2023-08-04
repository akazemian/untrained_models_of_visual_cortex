import sys
import os
from config import RESULTS_PATH


if not os.path.exists(os.path.join(RESULTS_PATH)):
    os.mkdir(os.path.join(RESULTS_PATH))
    
    
if not os.path.exists(os.path.join(RESULTS_PATH,'classification')):
    os.mkdir(os.path.join(RESULTS_PATH,'classification'))
    
    
