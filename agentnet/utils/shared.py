"""
Helpers for theano code
"""

import numpy as np
import theano
import theano.tensor as T


def create_shared(name, initial_value, dtype='floatX', strict=False, allow_downcast=True):
    if dtype == "floatX":
        dtype = theano.config.floatX
    initial_value = np.ascontiguousarray(initial_value, dtype=dtype)  # gpuarray fix by @kashif
    variable = theano.shared(initial_value, name=name, strict=strict, allow_downcast=allow_downcast)
    return variable


def set_shared(var, value):
    val_array = np.ascontiguousarray(value, dtype=var.dtype) # gpuarray fix by @kashif
    var.set_value(val_array)
