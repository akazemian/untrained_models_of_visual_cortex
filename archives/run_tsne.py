import os
import sys
import numpy as np
import random

from dotenv import load_dotenv

load_dotenv()
CACHE = os.getenv("CACHE")

from code_.additional_analysis.rsa.metrics import RDM, RDMSimilarity
import pickle 

with open(os.path.join(CACHE, 'category_embeddings'), 'rb') as file:
    category_embeddings = pickle.load(file)

with open(os.path.join(CACHE, 'tsne_results'), 'rb') as file:
    tSNE = pickle.load(file)

idx = random.sample(range(0, len(category_embeddings) + 1), 3000)
category_embeddings = np.array(category_embeddings)
category_embeddings[idx,:]
tSNE = tSNE[idx,:]

rdm = RDM()
rsa = RDMSimilarity(metric='pearsonr')

rdm_tne = rdm(tSNE)
rds_embed = rdm(np.array(category_embeddings))
rsa_embed_tsne = rsa(rdm_tne, rds_embed)

with open(os.path.join(CACHE, 'rsa_embed_tsne'), 'wb') as file:
    pickle.dump(rsa_embed_tsne, file)