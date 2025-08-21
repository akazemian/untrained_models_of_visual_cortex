from config import CACHE, RESULTS, PREDS_PATH, FIGURES
import os

os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PREDS_PATH, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)