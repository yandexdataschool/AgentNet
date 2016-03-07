

__doc__="""Math-related operations with vectors, matrices and tensors, all SYMBOLIC"""


from theano import tensor as T
import theano
import numpy as np


def norm(x,axis=-1,keepdims = True):
    """Compute l2 norm of x along axis"""
    return T.sqrt((x**2).sum(axis=axis,keepdims=keepdims))

def normalize(x,axis=-1):
    """return X divided by norm(x)
    If an element has zero norm, normalized element will still be zeros"""
    norms = norm(x,axis=axis,keepdims=True)
    return T.switch(T.eq(norms,0),0,x/norms)

def linspace(start,stop,num_units,dtype = "float32"):
    """a minimalistic symbolic equivalent of numpy.linspace"""
    return start + T.arange(num_units,dtype=dtype)*(stop-start) / (num_units-1)

def in1d(arr,in_arr):
    """for each element in arr returns 1 if in_arr contains this element, otherwise 0
    Output shape matches arr shape, in_arr must be 1d"""
    
    return T.eq(arr.reshape([1,-1]),in_arr.reshape([-1,1])).any(axis=0).reshape(arr.shape)

def prefix_ravel(seq,n_raveled_dim = 2): 
    """ravels first n_raveled_dimensions of seq into one
    p.e. if you have dimemsions of [batch_size,time_step,n_units],
    than prefix_ravel with 2 raveled dimensions will have [batch_and_time,n_units] dimension"""
    new_ndim = seq.ndim - n_raveled_dim+1
    new_shape = T.concatenate([[-1],seq.shape[n_raveled_dim:]])
    return seq.reshape(new_shape,ndim =new_ndim )



def append_dim(arg):
    """add 1 fake dimension to the end of arg"""
    return arg.reshape([i for i in arg.shape]+[1])

def insert_dim(arg,pos):
    """insert 1 fake dimension inside the arg before pos'th dimension"""
    shape = [i for i in arg.shape]
    shape.insert(pos,1)
    return arg.reshape(shape)
