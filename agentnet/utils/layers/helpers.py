import theano
import theano.tensor as T

from lasagne.layers import get_all_layers
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


def clip_grads(layer, clipping_bound):
    """Clips grads passing through a lasagne.layers.layer"""
    clipping_bound = abs(clipping_bound)
    grad_clip_op = lambda v: theano.gradient.grad_clip(v, -clipping_bound, clipping_bound)
    name = layer.name or "clipping_layer"
    return NonlinearityLayer(layer, grad_clip_op, name=name + ".grad_clip")


# dtype checker
def get_layer_dtype(layer, default=None):
    """ takes layer's output_dtype property if it is defined,
    otherwise defaults to default or (if it's not given) theano.config.floatX"""
    return layer.output_dtype if hasattr(layer, "output_dtype") else default or theano.config.floatX


def get_automatic_updates(layer_or_layers,treat_as_input=None,**kwargs):
    """
    Returns automatic updates from all the layers given and all layers they depend on.
    :param layer_or_layers: layer(s) to collect updates from
    :param treat_as_input: see same param in lasagne.layers.get_all_layers
    """
    updates = theano.OrderedUpdates()
    for layer in get_all_layers(layer_or_layers,treat_as_input=treat_as_input):
        if hasattr(layer,'get_automatic_updates'):
            updates += layer.get_automatic_updates(**kwargs)
    return updates
