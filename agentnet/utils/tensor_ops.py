"""
Symbolic operations with vectors, matrices and tensors.
"""

from theano import tensor as T

def norm(x, axis=-1, keepdims=True):
    """Compute l2 norm of x along axis"""
    return T.sqrt((x ** 2).sum(axis=axis, keepdims=keepdims))

def normalize(x, axis=-1):
    """return x divided by norm(x)
    If an element has zero norm, normalized element will still be zeros"""
    norms = norm(x, axis=axis, keepdims=True)
    return T.switch(T.eq(norms, 0), 0, x / norms)

def linspace(start, stop, num_units, dtype="float32"):
    """a minimalistic symbolic equivalent of numpy.linspace"""
    return start + T.arange(num_units, dtype=dtype) * (stop - start) / (num_units - 1)


def in1d(arr, in_arr):
    """for each element in arr returns 1 if in_arr contains this element, otherwise 0
    Output shape matches arr shape, in_arr must be 1d"""
    return T.eq(arr.reshape([1, -1]), in_arr.reshape([-1, 1])).any(axis=0).reshape(arr.shape)

def insert_dim(arg, pos=-1):
    """insert 1 fake dimension inside the arg before pos'th dimension"""
    shape = [i for i in arg.shape]
    shape.insert(pos, 1)
    return arg.reshape(shape)

def get_type(tensor):
    """creates a theano.tensor.TensorType that matches given tensor"""
    return T.TensorType(tensor.dtype,
                        tensor.broadcastable,
                        sparse_grad=getattr(tensor.type,"sparse_grad",False))

def cast_to_type(tensor,type):
    """converts tensor to given type. Invokes theano.TensorType.convert_variable and asserts it's successful"""
    assert type.ndim == tensor.ndim, "Make sure tensor and type have same number of dimensions"
    converted = type.convert_variable(tensor)
    assert converted is not None, "Failed to cast tensor to given type. Make sure they're compatible"
    return converted
