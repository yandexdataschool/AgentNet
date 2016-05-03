import theano
import theano.tensor as T
import lasagne
import numpy as np

from lasagne.layers import DenseLayer, NonlinearityLayer,ElemwiseSumLayer

from ..utils.format import check_list

def clip_grads(v, grad_clipping):
    grad_clipping = abs(grad_clipping)
    return  theano.gradient.grad_clip(v, -grad_clipping, grad_clipping)


def RecurrentMemoryLayer(prev_state,
                         input_or_inputs = [],
                         nonlinearity = lasagne.nonlinearities.sigmoid,
                         num_units = None,
                         name = "YetAnotherRNNLayer",
                         grad_clipping=5.):
    """
        
    Implements a one-step recurrent neural network (RNN) with arbitrary number of units.
    
    parameters:
        prev_state: input that denotes previous state (shape must be (None, n_units) )
        input_or_inputs: a single layer or a list/tuple of layers that go as inputs
        nonlinearity: which nonlinearity to use
        num_units: how many recurrent cells to use. None means "as in prev_state"
        grad_clipping: maximum gradient absolute value. 0 or None means "no clipping"
        
    
    for developers:
        Works by stacking DenseLayers with ElemwiseSumLayer.
        is a function mock, not actual class.
    
    """
    
    assert len(prev_state.output_shape)==2
    #if needed, infer num_units
    if num_units is None:
        num_units = prev_state.output_shape[1]
    #else check it    
    assert num_units == prev_state.output_shape[1]
    
    inputs = check_list(input_or_inputs)

    
    if grad_clipping:
        prev_state = NonlinearityLayer(prev_state, lambda v: clip_grads(v,grad_clipping))
        
        inputs = [ NonlinearityLayer(layer, lambda v: clip_grads(v,grad_clipping))
                  for layer in inputs]
        
        

    #from prev state to current state (with bias)
    hid_to_hid = DenseLayer(prev_state, 
                            num_units = num_units,
                            nonlinearity=None,
                            name = name+".hid_to_hid")
    
    #from inputs to current state (without bias)
    inputs_to_hid = [DenseLayer(input_layer, 
                                num_units = num_units,
                                nonlinearity=None,
                                b = None, #Cruicial! This disables additional bias layers
                                name = name+".input%i_to_hid"%(i))
                     for i, input_layer in enumerate(inputs)]
    
    
    #stack them
    elwise_sum = ElemwiseSumLayer([hid_to_hid]+inputs_to_hid, name = name+".sum")
    
    #finally, apply nonlinearity
    
    new_hid = NonlinearityLayer(elwise_sum,
                                nonlinearity,
                                name=name+".nonlinearity")
    
    return new_hid
    
                     
    
