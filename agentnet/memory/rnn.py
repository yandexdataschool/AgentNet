import theano
import theano.tensor as T
import lasagne
import numpy as np

from lasagne.layers import DenseLayer, NonlinearityLayer,ElemwiseSumLayer

from ..utils.format import check_list
from ..utils.layers import clip_grads

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
        prev_state = clip_grads(prev_state,grad_clipping)
        inputs = map(lambda lyr: clip_grads(lyr,grad_clipping), inputs)
        
        

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
                                name=name+".new_state")
    
    return new_hid
    
                     
import theano
import theano.tensor as T
import lasagne
import numpy as np

from lasagne.layers import DenseLayer, NonlinearityLayer,ElemwiseMergeLayer

from agentnet.utils.format import check_list
from agentnet.utils.layers import clip_grads
from agentnet.memory.gate import GateLayer



def GRUCell(prev_state,
             input_or_inputs = [],
             num_units = None,
             forgetgate_nonlinearity = lasagne.nonlinearities.sigmoid,
             updategate_nonlinearity = lasagne.nonlinearities.sigmoid,
             hidden_update_nonlinearity = lasagne.nonlinearities.tanh,
             name = "YetAnotherGRULayer",
             grad_clipping=5.
            ):
    """
        
    Implements a one-step gated recurrent unit (GRU) with arbitrary number of units.
    
    parameters:
        prev_state: input that denotes previous state (shape must be (None, n_units) )
        input_or_inputs: a single layer or a list/tuple of layers that go as inputs
        *_nonlinearity: which nonlinearity to use for a particular gate
        num_units: how many recurrent cells to use. None means "as in prev_state"
        grad_clipping: maximum gradient absolute value. 0 or None means "no clipping"
        
    
    for developers:
        Works by stacking other lasagne layers;
        is a function mock, not actual class.
    
    """
    
    assert len(prev_state.output_shape)==2
    #if required, infer num_units
    if num_units is None:
        num_units = prev_state.output_shape[1]
    #else check it    
    assert num_units == prev_state.output_shape[1]
    
    inputs = check_list(input_or_inputs)
    
    if grad_clipping:
        prev_state = clip_grads(prev_state,grad_clipping)
        inputs = map(lambda lyr: clip_grads(lyr,grad_clipping), inputs)
    
    
    #apply gates
    inp_to_gates = GateLayer(inputs,[num_units]*3,
                             gate_nonlinearities=None)
    inp_forget, inp_update, inp_hidden_update = inp_to_gates

    hid_to_gates = GateLayer(prev_state,[num_units]*3,
                             gate_nonlinearities=None,
                             bias_init = None)
    hid_forget, hid_update, hid_hidden_update = hid_to_gates

    #showtcut functions
    def gate(inp,hid,nonlinearity):
        return NonlinearityLayer(
            ElemwiseSumLayer([inp,hid]),
            nonlinearity)
    def mul(a,b):
        return ElemwiseMergeLayer([a,b],T.mul)
    def add(a,b):
        return ElemwiseMergeLayer([a,b],T.add)

    #compute both gates
    forgetgate = gate(inp_forget, hid_forget,forgetgate_nonlinearity)
    updategate = gate(inp_update, hid_update,updategate_nonlinearity)
    
    #compute hidden update
    hidden_update = add(
        inp_hidden_update,
        mul(forgetgate,hid_hidden_update)
    )
    hidden_update = NonlinearityLayer(hidden_update, hidden_update_nonlinearity)
    
    
    
    #compute new hidden values
    new_hid = add(
        mul(updategate,prev_state), 
        mul(updategate,hidden_update)
    )

    
    
    return new_hid
    
                     
    

