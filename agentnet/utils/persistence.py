__doc__="""Saves and loads lasagne model weights"""

import pickle
import lasagne
from shared import set_shared


def save(nn,fname):
    param_list = lasagne.layers.get_all_params(nn)
    
    params = {par.name: par.get_value() for par in param_list}
    
    if len(params) != len(param_list):#assert no duplicate layer names
        raise ValueError, "all params must have unique names (todo: fix)"
    with open(fname,'w') as fout:
        pickle.dump(params,fout)
        
def load(nn,fname):
    param_containters = lasagne.layers.get_all_params(nn)
    
    with open(fname,'r') as fin:
        saved_params=pickle.load(fin)
    
    for param in param_containters:
        set_shared(param,saved_params[param.name])
    return nn



