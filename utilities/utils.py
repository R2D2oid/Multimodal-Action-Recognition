
from os import listdir
from os.path import isfile, join
import numpy as np

def load_feats_from(path, emb_size):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # load i3d video features 
    feats = []
    for f in files:
        temp = np.load(join(path, f))
        temp = temp.reshape(-1, emb_size)
        feats.append(temp)
    
    return feats
