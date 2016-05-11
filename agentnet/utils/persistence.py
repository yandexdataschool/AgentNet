__doc__="""Saves and loads lasagne model weights"""

import pickle as pickle
import lasagne
from .shared import set_shared


def save(nn,fname):
    params= lasagne.layers.get_all_param_values(nn)
    with open(fname,'w') as fout:
        pickle.dump(params,fout)
        
def load(nn,fname):
    with open(fname,'r') as fin:
        saved_params=pickle.load(fin)
    lasagne.layers.set_all_param_values(nn,saved_params)
    return nn
