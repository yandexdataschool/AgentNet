"""
'Layers' introduces a number of auxiliary lasagne layers that are used throughout AgentNet and examples
"""

import theano
import theano.tensor as T

from lasagne.layers import MergeLayer, ExpressionLayer
from lasagne.layers import NonlinearityLayer, ElemwiseMergeLayer


# shortcut functions
def mul(*args, **kwargs):
    """Element-wise multiply layers"""
    inp_names = [layer.name or "layer" + str(i) for i, layer in enumerate(args)]
    kwargs["name"] = kwargs.get("name",
                                "sum(%s)" % (', '.join(inp_names)))

    return ElemwiseMergeLayer(args, T.mul, **kwargs)


def add(*args, **kwargs):
    """Element-wise sum of layers"""
    inp_names = [layer.name or "layer" + str(i) for i, layer in enumerate(args)]
    kwargs["name"] = kwargs.get("name",
                                "sum(%s)" % (', '.join(inp_names)))

    return ElemwiseMergeLayer(args, T.add, **kwargs)


def transform(a, fun=lambda x: x, **kwargs):
    """Alias for NonlinearityLayer"""
    kwargs["name"] = kwargs.get("name",
                                (a.name or "layer") + ".transform", )

    return NonlinearityLayer(a, fun, **kwargs)


def clip_grads(layer, grad_clipping):
    """Clips grads passing through a lasagne.layers.layer"""
    grad_clipping = abs(grad_clipping)
    grad_clip_op = lambda v: theano.gradient.grad_clip(v, -grad_clipping, grad_clipping)
    name = layer.name or "layer"
    return NonlinearityLayer(layer, grad_clip_op, name=name + ".grad_clip")


# dtype checker

def get_layer_dtype(layer, default=None):
    """ takes layer's output_dtype property if it is defined, 
    otherwise defaults to default or (if it's not given) theano.config.floatX"""
    return layer.output_dtype if hasattr(layer, "output_dtype") else default or theano.config.floatX


class TupleLayer(MergeLayer):
    """
    An abstract base class for Lasagne layer that returns several outputs.
    Has to implement get_output_shape so that it contains a list/tuple of output shapes(tuples),
    one for each output.

    In other words, if you return 2 elements of shape (None, 25) and (None,15,5,7),
    self.get_output_shape must return [(None,25),(None,15,5,7)]

    warning: this layer is needed for the purposes of optimal coding,
    it slightly breaks Lasagne conventions, so it is hacky.
    """

    @property
    def output_shapes(self):
        """
        One must implement this method when inheriting from TupleLayer.
        TupleLayer's shape must be a sequence of shapes for each output (depth-2 list or tuple)
        """
        raise NotImplementedError

    def get_output_shape_for(self, input_shape):
        """
        This is a mock output_shape that should only be used for service reasons.
        To get actual shapes, consider self.output_shapes
        """
        return (len(self.output_shapes),)

    @property
    def element_names(self):
        name = (self.name or "tuple layer")
        return [name + "[{}]".format(i)
                for i in range(len(self.output_shapes))]

    @property
    def disable_tuple(self):
        """
        if True, forces tuple to work as a single layer.
        Useful if your layer has a different behavior depending on parameters
        """
        return False

    @property
    def output_dtype(self):
        """ dtypes of tuple outputs"""
        return [theano.config.floatX for i in range(len(self))]

    def __len__(self):
        """an amount of output layers in a tuple"""
        return len(self.output_shapes) if not self.disable_tuple else 1

    def __getitem__(self, ind):
        """ returns a lasagne layer that yields i-th output of the tuple.
        parameters:
            ind - an integer index or a slice.
        returns:
            a particular layer of the tuple or a list of such layers if slice is given 
        """
        assert not self.disable_tuple
        assert type(ind) in (int, slice)

        if type(ind) is slice:
            return list(self)[ind]

        assert type(ind) == int

        item_layer = ExpressionLayer(self, lambda values: values[ind],
                                     output_shape=self.output_shapes[ind],
                                     name=self.element_names[ind])

        item_layer.output_dtype = self.output_dtype[ind]

        return item_layer

    def __iter__(self):
        """ iterate over tuple output layers"""
        if self.disable_tuple:
            return [self]
        else:
            return (self[i] for i in range(len(self)))

# TODO implement rev_grad here?
