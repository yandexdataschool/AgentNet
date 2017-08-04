"""
An utility layer that applies in itself a custom lasagne network.
Also contains a convenience function reapply.
>>> #build original network
>>> l_in1 = InputLayer([None,10],T.zeros([5,10]))
>>> l_d1 = DenseLayer(l_in1,20)
>>> l_d2 = DenseLayer(l_in1,30)
>>> l_cat = ConcatLayer([l_d1,l_d2])
>>> l_d3 = DenseLayer(l_cat,20)
>>> l_in2 = InputLayer([None,10],T.zeros([5,10]))
>>> new_l_d3 = reapply(l_d3,{l_in1:l_in2})  #reapply the whole network to a new in 
>>> l_in3 = InputLayer([None,30],T.zeros([5,30]))
>>> new_l_d3 = reapply(l_d3,{l_d2:l_in3}) #reapply just one layer
"""
from collections import OrderedDict
from lasagne.layers import Layer, get_output, get_all_params
from .dict import DictLayer
from .helpers import get_layer_dtype
from ..format import check_list,check_ordered_dict

class ReapplyLayer(DictLayer):
    def __init__(self,layers,replacements,output_shapes=None,output_dtypes=None,**kwargs):
        """
        Applies a part of lasagne network to a new place.
        :param layers: layers to be re-applied
        :type layers: dict {key:layer} or list of layers
        :param replacemnts: a dict {old_layer:new_layer} that defines which layers should be substituted by which other layers
        :type replacements: dict of {Layer:Layer}
        :param output_shapes: dict or list of layer shapes. By default, assumes old shapes to be valid.
        :param output_dtypes: a dict or list of dtypes. By default, keeps old dtypes.
        """

        if not isinstance(layers,dict):
            #if layers is a list (or single layer), convert to {0:layers[0],1:layers[1],...}
            layers = dict(enumerate(check_list(layers)))
            
        self.layers = check_ordered_dict(layers)
        output_shapes = output_shapes or {k:l.output_shape for k,l in layers.items()}
        output_dtypes = output_dtypes or {k:get_layer_dtype(l) for k,l in layers.items()}
        
        self.input_keys,input_layers = zip(*replacements.items())

        super(ReapplyLayer,self).__init__(input_layers,output_shapes,output_dtypes,**kwargs)
        
    def get_output_for(self,inputs,**kwargs):
        
        output_keys,output_layers = zip(*self.layers.items())
        outputs = get_output(output_layers, dict(zip(self.input_keys,inputs)),**kwargs)
        
        return OrderedDict(zip(output_keys,outputs))
        
    def get_params(self,**kwargs):
        return get_all_params(list(self.layers.values()),**kwargs)

def reapply(layer_or_layers,replacements,**kwargs):
    """
    A convenience function to for ReapplyLayer.
    If given only a single layer, returns a single layer. If a list - returns a list. Dict -> dict.
    
    :param layer_or_layers: a single layer, dict-like or list-like of layers to be reapplied.
    :param replacements: a dictionary of {old_layer:replacement_layer}.

    Examples
    --------

    >>> #build original network
    >>> l_in1 = InputLayer([None,10],T.zeros([5,10]))
    >>> l_d1 = DenseLayer(l_in1,20)
    >>> l_d2 = DenseLayer(l_in1,30)
    >>> l_cat = ConcatLayer([l_d1,l_d2])
    >>> l_d3 = DenseLayer(l_cat,20)
    >>> l_in2 = InputLayer([None,10],T.zeros([5,10]))
    >>> new_l_d3 = reapply(l_d3,{l_in1:l_in2})  #reapply the whole network to a new in 
    >>> l_in3 = InputLayer([None,30],T.zeros([5,30]))
    >>> new_l_d3 = reapply(l_d3,{l_d2:l_in3}) #reapply just one layer
    
    """
    
    assert isinstance(replacements,dict),"replacements must be a dictionary {old_layer:new_layer}"
    assert isinstance(layer_or_layers,Layer) or hasattr(layer_or_layers,"__getitem__"),"layers must be either "\
        "a single layer, dict-like or iterable of layers. In the latter case, it must support indexing."
    
    reapplied = ReapplyLayer(layer_or_layers,replacements,**kwargs)
    if isinstance(layer_or_layers,Layer): #single layer
        return reapplied[0]
    
    elif isinstance(layer_or_layers,dict): #dict of layers
        return {k: reapplied[k] for k in layer_or_layers.keys()}
    
    else: #list, tuple or similar
        return [reapplied[i] for i in range(len(layer_or_layers))]
