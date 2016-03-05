__doc__= """theano shared variable helpers for less verbose code"""

from theano import tensor as T
import theano
import numpy as np

def create_shared(name,val,dtype='floatX',strict=False,allow_downcast=True):
    if dtype == "floatX":
        dtype = theano.config.floatX
    variable = theano.shared(np.array(val,dtype=dtype),
                         name,strict = strict,allow_downcast=allow_downcast)
    return variable


def set_shared(var,value):
    val_array = np.array(value,dtype=var.dtype)
    
    var.set_value(val_array)