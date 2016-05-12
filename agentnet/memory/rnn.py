import theano
import theano.tensor as T
import lasagne
import numpy as np

from lasagne.layers import DenseLayer,NonlinearityLayer,ElemwiseMergeLayer,NonlinearityLayer
from ..utils.format import check_list







#Vanilla RNN cell

def RNNCell(prev_state,
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
    
    
    
    #hidden to gates
    hid_to_gates = GateLayer(prev_state,[num_units]*3,
                             gate_nonlinearities=None,
                             bias_init = None,
                             name = name+".hidden_to_gates_stacked")
    hid_forget, hid_update, hidden_update_hid = hid_to_gates

    #clip grads #1
    if grad_clipping:
        inputs = map(lambda lyr: clip_grads(lyr,grad_clipping), inputs)
        hid_forget, hid_update, hidden_update_hid = map(lambda lyr: clip_grads(lyr,grad_clipping),
                                                        [hid_forget, hid_update, hidden_update_hid])
        
    #input to gates
    inp_to_gates = GateLayer(inputs,[num_units]*3,
                             gate_nonlinearities=None,
                             name=name+".input_to_gates_stacked")
    inp_forget, inp_update, hidden_update_in = inp_to_gates



    #compute forget and update gates
    forgetgate = transform(
        add(inp_forget, hid_forget),
        forgetgate_nonlinearity,
        name="forgetgate"
    )
    updategate = transform(
        add(inp_update, hid_update),
        updategate_nonlinearity,
        name="updategate"
    )

    inv_updategate = transform(updategate, 
                               lambda x : 1-x,
                               name="1 - updategate")
    
    #compute hidden update
    hidden_update = add(
        hidden_update_in,
        mul(forgetgate,hidden_update_hid),
        name="hid_update"
    )
    
    #clip grads #2
    if grad_clipping:
        hidden_update = clip_grads(hidden_update,
                                   grad_clipping) 
    
    hidden_update = NonlinearityLayer(hidden_update, 
                                      hidden_update_nonlinearity)
    
    
    
    #compute new hidden values
    new_hid = add(
        mul(inv_updategate,prev_state), 
        mul(updategate,hidden_update),
        name= name
    )

    
    
    return new_hid


    
                     
    

