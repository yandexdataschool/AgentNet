from theano import tensor as T
import theano

__doc__ = """helper functions for symbolic theano code"""

#shared variables
import numpy as np
def _shared(name,val,dtype='floatX',strict=True,allow_downcast=True):
    if dtype == "floatX":
        dtype = theano.config.floatX
    return theano.shared(np.array(val,dtype=dtype),
                         name,strict = strict,allow_downcast=allow_downcast)
        
def set_shared(var,value):
    var.set_value(np.array(value,dtype=var.dtype))


#vector algebra
_norm = lambda x: T.sqrt((x**2).sum(axis=-1,keepdims=True))
def _normalize(x):
    norms = _norm(x)
    return T.switch(T.eq(norms,0),0,x/norms)

def _append_dim(_arg):
    return _arg.reshape([i for i in _arg.shape]+[1])

def _insert_dim(_arg,pos):
    shape = [i for i in _arg.shape]
    shape.insert(pos,1)
    return _arg.reshape(shape)

def _linspace(start,stop,num_units,dtype = "float32"):
    """a minimalistic symbolic equivalent of numpy.linspace"""
    return start + T.arange(num_units,dtype=dtype)*(stop-start) / (num_units-1)
def _in1d(arr,in_arr):
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


#randomness
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
def mrg_sample(rng,arr,shape,indices_type='uint32'):
    """a random sample function with returns elements of arr sampled WITH repetition
    rng - MRG_RandomStreams instance from theano.sandbox.rng_mrg
    arr - elements to sample from
    shape - tuple of int or theano variables: output shape
    indices_type - what dtype of indices to use when indexing arr (in case of >2^32 elements)
    
    implemented via MRG random stream
    samples [shape] elements of arr(1d array) """
    
    
    n_samples = T.prod(shape) 
    n_options = arr.shape[0]
    random_zero_one = rng.uniform((n_samples,))
    indices = (random_zero_one*n_options).astype(indices_type)%n_options
    return arr[indices].reshape(shape)
def get_mrg(seed=None):
    if seed is None:
        seed = get_rng().randint(1, 2147462579)
    return RandomStreams(seed)


#gradients
from theano.tensor.opt import register_canonicalize
class ConsiderConstant(theano.compile.ViewOp):
    """treats input as constant when computing grads"""
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()
register_canonicalize(theano.gof.OpRemove(consider_constant), name='remove_consider_constant_op')
