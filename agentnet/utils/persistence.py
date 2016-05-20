"""
Utils to save and load lasagne model weights
"""

import sys
from six.moves import cPickle as pickle
import lasagne


def save(nn, filename):
    params = lasagne.layers.get_all_param_values(nn)
    with open(filename, 'wb') as fout:
        pickle.dump(params, fout, protocol=2)


def load(nn, filename):
    kwargs = {}
    if sys.version_info >= (3,):
        kwargs = {'encoding': 'latin1'}

    with open(filename, 'rb') as fin:
        saved_params = pickle.load(fin, **kwargs)

    lasagne.layers.set_all_param_values(nn, saved_params)
    return nn
