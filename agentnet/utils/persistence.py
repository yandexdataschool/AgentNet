__doc__="""Saves and loads lasagne model weights"""

import sys
if sys.version_info < (3):
    import cPickle as pickle
else:
    import pickle

import lasagne
from .shared import set_shared


def save(nn,fname):
    params = lasagne.layers.get_all_param_values(nn)
    with open(fname,'wb') as fout:
        pickle.dump(params, fout, protocol=2)
        
def load(nn,fname):
    kwargs = {}
    if sys.version_info >=(3,):
        kwargs = {'encoding':'latin1'}
        
    with open(fname,'rb') as fin:
        saved_params = pickle.load(fin, **kwargs)
        
    lasagne.layers.set_all_param_values(nn, saved_params)
    return nn