"""
Helpers for theano code
"""

import numpy as np
import theano


def create_shared(name, initial_value, dtype='floatX', strict=False, allow_downcast=True):
    if dtype == "floatX":
        dtype = theano.config.floatX
    initial_value = np.array(initial_value, dtype=dtype)
    variable = theano.shared(initial_value, name=name, strict=strict, allow_downcast=allow_downcast)
    return variable


def set_shared(var, value):
    val_array = np.array(value, dtype=var.dtype)
    var.set_value(val_array)
