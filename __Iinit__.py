from config import CACHE, RESULTS
import os

if not os.path.exists(CACHE):
    os.makedir(CACHE)

if not os.path.exists(RESULTS):
    os.makedir(RESULTS)