"""
Utils to save and load lasagne model weights
"""

import sys
from six.moves import cPickle as pickle
import lasagne


def save(nn, filename):
    """
    Saves lasagne network weights to the target file.
    Does not store the architecture itself.

    Basic usage:
    >> nn = lasagne.layers.InputLayer(...)
    >> nn = lasagne.layers.SomeLayer(...)
    >> nn = lasagne.layers.SomeLayer(...)
    >> train_my_nn()
    >> save(nn,"nn_weights.pcl")

    Loading weights is possible through .load function in the same module.

    :param nn: neural network output layer(s)
    :param filename: weight filename
    """
    params = lasagne.layers.get_all_param_values(nn)
    with open(filename, 'wb') as fout:
        pickle.dump(params, fout, protocol=2)


def load(nn, filename):
    """
    Loads lasagne network weights from the target file into NN you provided.
    Requires that NN architecture is exactly same as NN which weights were saved.
    Minor alterations like changing hard-coded batch size will probably work, but are not guaranteed.

    Basic usage:
    >> nn = lasagne.layers.InputLayer(...)
    >> nn = lasagne.layers.SomeLayer(...)
    >> nn = lasagne.layers.SomeLayer(...)
    >> train_my_nn()
    >> save(nn,"previously_saved_weights.pcl")
    >> crash_and_lose_progress()
    >> nn = the_same_nn_as_before()
    >> load(nn,"previously_saved_weights.pcl")


    :param nn: neural network output layer(s)
    :param filename: weight filename
    :return: the network with weights loaded

     WARNING!
     the load() function is inplace, meaning that weights are loaded in the NN instance you provided and NOT in a copy.
    """
    kwargs = {}
    if sys.version_info >= (3,):
        kwargs = {'encoding': 'latin1'}

    with open(filename, 'rb') as fin:
        saved_params = pickle.load(fin, **kwargs)

    lasagne.layers.set_all_param_values(nn, saved_params)
    return nn
