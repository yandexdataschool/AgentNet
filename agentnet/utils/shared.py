"""
Helpers for theano code
"""

import numpy as np
import theano
import theano.tensor as T


def create_shared(name, initial_value, dtype='floatX', strict=False, allow_downcast=True,device=None):
    if dtype == "floatX":
        dtype = theano.config.floatX
    initial_value = np.array(initial_value, dtype=dtype)
    if device is not None:
        variable = theano.tensor._shared(initial_value, name=name, strict=strict, allow_downcast=allow_downcast,target=device)
    else:
        variable = theano.shared(initial_value, name=name, strict=strict, allow_downcast=allow_downcast)
    return variable


def set_shared(var, value):
    val_array = np.array(value, dtype=var.dtype)
    var.set_value(val_array)
