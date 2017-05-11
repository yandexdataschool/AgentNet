import lasagne
from lasagne import init
from lasagne.layers import DenseLayer, NonlinearityLayer, DropoutLayer

from ..utils.format import check_list
from ..utils.layers import clip_grads, add, mul
from ..memory.gate import GateLayer



# Vanilla RNN cell

def RNNCell(prev_state,
            input_or_inputs=tuple(),
            nonlinearity=lasagne.nonlinearities.sigmoid,
            num_units=None,
            name=None,
            grad_clipping=5.,
            Whid = init.GlorotUniform(),
            Winp = init.GlorotUniform(),
            b = init.Constant(),
            ):
    """
        
    Implements a one-step recurrent neural network (RNN) with arbitrary number of units.
    

    :param prev_state: input that denotes previous state (shape must be (None, n_units) )
    :param input_or_inputs: a single layer or a list/tuple of layers that go as inputs
    :param nonlinearity: which nonlinearity to use
    :param num_units: how many recurrent cells to use. None means "as in prev_state"
    :param grad_clipping: maximum gradient absolute value. 0 or None means "no clipping"

    :returns: updated memory layer
    :rtype: lasagne.layers.Layer



    for developers:
        Works by stacking DenseLayers with ElemwiseSumLayer.
        is a function mock, not actual class.
    
    """

    assert len(prev_state.output_shape) == 2
    # if needed, infer num_units
    if num_units is None:
        num_units = prev_state.output_shape[1]
    # else check it
    assert num_units == prev_state.output_shape[1]

    inputs = check_list(input_or_inputs)
    

    if grad_clipping:
        prev_state = clip_grads(prev_state, grad_clipping)
        inputs = [clip_grads(lyr, grad_clipping) for lyr in inputs]

    # from prev state to current state (with bias)
    hid_to_hid = DenseLayer(prev_state,
                            num_units=num_units,
                            nonlinearity=None,
                            name=(name or "") + ".hid_to_hid",
                            W= Whid,
                            b= b)

    Winp = check_list(Winp)
    if len(Winp) ==1:
        Winp *=len(inputs)

    # from inputs to current state (without bias)
    inputs_to_hid = [DenseLayer(input_layer,
                                num_units=num_units,
                                nonlinearity=None,
                                W=Winp[i],
                                b=None,  #This disables additional bias layers
                                name=(name or "") + ".input%i_to_hid" % (i))
                     for i, input_layer in enumerate(inputs)]

    # stack them
    elwise_sum = add(*([hid_to_hid] + inputs_to_hid), name=(name or "") + ".sum")

    # finally, apply nonlinearity

    new_hid = NonlinearityLayer(elwise_sum,
                                nonlinearity,
                                name=(name or "") + ".new_state")

    return new_hid




def GRUCell(prev_state,
            input_or_inputs=tuple(),
            num_units=None,
            weight_init=init.Normal(),
            bias_init=init.Constant(),
            forgetgate_nonlinearity=lasagne.nonlinearities.sigmoid,
            updategate_nonlinearity=lasagne.nonlinearities.sigmoid,
            hidden_update_nonlinearity=lasagne.nonlinearities.tanh,
            dropout=0,
            name="YetAnotherGRULayer",
            grad_clipping=5.
            ):
    """
        
    Implements a one-step gated recurrent unit (GRU) with arbitrary number of units.
    
    
    :param prev_state: input that denotes previous state (shape must be (None, n_units) )
    :type prev_state: lasagne.layers.Layer
    :param input_or_inputs: a single layer or a list/tuple of layers that go as inputs
    :type input_or_inputs: lasagne.layers.Layer or list of such
    
    :param num_units: how many recurrent cells to use. None means "as in prev_state"
    :type num_units: int
    
    :param weight_init: either a lasagne initializer to use for every gate weights
        or a list of two initializers:
        - first used for all weights from hidden -> <any>_gate and hidden update
        - second used for all weights from input(s) -> <any>_gate weights and hidden update
        or a list of two objects elements:
        - second list is hidden -> forget gate, update gate, hidden update
        - second list of lists where
        list[i][0,1,2] = input[i] -> [forget gate, update gate, hidden update]
    :param <any>_nonlinearity: which nonlinearity to use for a particular gate

    :param dropout: dropout rate as per https://arxiv.org/abs/1603.05118

    :param grad_clipping: maximum gradient absolute value. 0 or None means "no clipping"

    :returns: updated memory layer
    :rtype: lasagne.layers.Layer

    
    for developers:
        Works by stacking other lasagne layers;
        is a function mock, not actual class.
    
    """

    assert len(prev_state.output_shape) == 2
    # if required, infer num_units
    if num_units is None:
        num_units = prev_state.output_shape[1]
    # else check it
    assert num_units == prev_state.output_shape[1]

    inputs = check_list(input_or_inputs)
    
    #handle weight init
    weight_init = check_list(weight_init)
    if len(weight_init) == 1:
        weight_init *= 2
    hidden_W_init, input_W_init = weight_init


    # hidden to gates
    hid_to_gates = GateLayer(prev_state, [num_units] * 3,
                             gate_nonlinearities=None,
                             channel_names=["to_resetgate","to_updategate", "to_hidden_update"],
                             bias_init=None,
                             weight_init=hidden_W_init,
                             name=name or "")
    
    hid_forget, hid_update, hidden_update_hid = hid_to_gates.values()

    # clip grads #1
    if grad_clipping:
        inputs = [clip_grads(lyr, grad_clipping) for lyr in inputs]
        hid_forget, hid_update, hidden_update_hid = [clip_grads(lyr, grad_clipping) for lyr in
                                                     [hid_forget, hid_update, hidden_update_hid]]

    # input to gates
    inp_to_gates = GateLayer(inputs, [num_units] * 3,
                             gate_nonlinearities=None,
                             channel_names=["to_resetgate", "to_updategate", "to_hidden_update"],
                             bias_init = bias_init,
                             weight_init = input_W_init,
                             name=name or "")
    inp_forget, inp_update, hidden_update_in = inp_to_gates.values()

    # compute forget and update gates
    forgetgate = NonlinearityLayer(
        add(inp_forget, hid_forget),
        forgetgate_nonlinearity,
        name=(name or "")+".forgetgate"
    )
    updategate = NonlinearityLayer(
        add(inp_update, hid_update),
        updategate_nonlinearity,
        name=(name or "")+".updategate"
    )

    inv_updategate = NonlinearityLayer(updategate,
                               lambda x: 1 - x,
                               name=(name or "")+".[1 - updategate]")

    # compute hidden update
    hidden_update = add(
        hidden_update_in,
        mul(forgetgate, hidden_update_hid),
        name=(name or "")+".hid_update"
    )

    # clip grads #2
    if grad_clipping:
        hidden_update = clip_grads(hidden_update,
                                   grad_clipping)

    hidden_update = NonlinearityLayer(hidden_update,
                                      hidden_update_nonlinearity)

    if dropout != 0:
        hidden_update = DropoutLayer(hidden_update,p=dropout)

    # compute new hidden values
    new_hid = add(
        mul(inv_updategate, prev_state),
        mul(updategate, hidden_update),
        name=name
    )

    return new_hid


def LSTMCell(prev_cell,
             prev_out,
             input_or_inputs=tuple(),
             num_units=None,
             peepholes=True,
             weight_init=init.Normal(),
             bias_init=init.Constant(),
             peepholes_W_init=init.Normal(),
             forgetgate_nonlinearity=lasagne.nonlinearities.sigmoid,
             inputgate_nonlinearity=lasagne.nonlinearities.sigmoid,
             outputgate_nonlinearity=lasagne.nonlinearities.sigmoid,
             cell_nonlinearity=lasagne.nonlinearities.tanh,
             output_nonlinearity=lasagne.nonlinearities.tanh,
             dropout=0.,
             name=None,
             grad_clipping=5.,
             ):


    """

    Implements a one-step LSTM update. Note that LSTM requires both c_t (private memory) and h_t aka output.


    :param prev_cell: input that denotes previous "private" state (shape must be (None, n_units) )
    :type prev_cell: lasagne.layers.Layer
    :param prev_out: input that denotes previous "public" state (shape must be (None,n_units))
    :type prev_out: lasagne.layers.Layer
    :param input_or_inputs: a single layer or a list/tuple of layers that go as inputs
    :type input_or_inputs: lasagne.layers.Layer or list of such

    :param num_units: how many recurrent cells to use. None means "as in prev_state"
    :type num_units: int

    :param peepholes: If True, the LSTM uses peephole connections.
        When False, peepholes_W_init are ignored.
    :type peepholes: bool

    :param bias_init: either a lasagne initializer to use for every gate weights
                        or a list of 4 initializers for  [input gate, forget gate, cell, output gate]

    :param weight_init: either a lasagne initializer to use for every gate weights:
        or a list of two initializers,
        - first used for all weights from hidden -> <all>_gate and cell
        - second used for all weights from input(s) -> <all>_gate weights and cell
        or a list of two objects elements,
        - second list is hidden -> input gate, forget gate, cell, output gate,
        - second list of lists where list[i][0,1,2] = input[i] -> [input_gate, forget gate, cell,output gate ]

    :param peepholes_W_init: either a lasagne initializer or a list of 3 initializers for
                        [input_gate, forget gate,output gate ] weights. If peepholes=False, this is ignored.
                        
    :param <any>_nonlinearity: which nonlinearity to use for a particular gate
    :param dropout: dropout rate as per https://arxiv.org/pdf/1603.05118.pdf
    :param grad_clipping: maximum gradient absolute value. 0 or None means "no clipping"


    :returns: a tuple of (new_cell,new_output) layers
    :rtype: (lasagne.layers.Layer,lasagne.layers.Layer)


    for developers:
        Works by stacking other lasagne layers;
        is a function mock, not actual class.

    """

    assert len(prev_cell.output_shape) == 2
    # if required, infer num_units
    if num_units is None:
        num_units = prev_cell.output_shape[1]
    # else check it
    assert num_units == prev_cell.output_shape[1]


    # gates and cell (before nonlinearities)

    gates = GateLayer([prev_out] + check_list(input_or_inputs),
                      [num_units] * 4,
                      channel_names=["to_ingate", "to_forgetgate", "to_cell", "to_outgate"],
                      gate_nonlinearities=None,
                      bias_init=bias_init,
                      weight_init=weight_init,
                      name=name or "")

    ingate, forgetgate, cell_input, outputgate = gates.values()




    # clip grads #1
    if grad_clipping:
        ingate, forgetgate, cell_input, outputgate = [clip_grads(lyr, grad_clipping) for lyr in
                                                     [ingate, forgetgate, cell_input, outputgate]]

    if peepholes:
        # cast bias init to a list
        peepholes_W_init = check_list(peepholes_W_init)
        assert len(peepholes_W_init) in (1, 3)
        if len(peepholes_W_init) == 1:
            peepholes_W_init *= 3
        W_cell_to_ingate_init,W_cell_to_forgetgate_init= peepholes_W_init[:2]

        peep_ingate = lasagne.layers.ScaleLayer(prev_cell,W_cell_to_ingate_init,shared_axes=[0,],
                                  name= (name or "") + ".W_cell_to_ingate_peephole")

        peep_forgetgate = lasagne.layers.ScaleLayer(prev_cell,W_cell_to_forgetgate_init,shared_axes=[0,],
                                  name= (name or "") + ".W_cell_to_forgetgate_peephole")


        ingate = add(ingate,peep_ingate)
        forgetgate = add(forgetgate,peep_forgetgate)



    # nonlinearities
    ingate = NonlinearityLayer(
        ingate,
        inputgate_nonlinearity,
        name=(name or "")+".inputgate"
    )
    forgetgate = NonlinearityLayer(
        forgetgate,
        forgetgate_nonlinearity,
        name=(name or "")+".forgetgate"
    )

    cell_input = NonlinearityLayer(cell_input,
                          nonlinearity=cell_nonlinearity,
                          name=(name or "")+'.cell_nonlinearity')

    if dropout != 0:
        cell_input = DropoutLayer(cell_input,p=dropout)


    # cell = input * ingate + prev_cell * forgetgate
    new_cell= add(mul(cell_input,ingate),
                  mul(prev_cell, forgetgate))

    # output gate
    if peepholes:
        W_cell_to_outgate_init = peepholes_W_init[2]

        peep_outgate = lasagne.layers.ScaleLayer(new_cell,W_cell_to_outgate_init,shared_axes=[0,],
                                  name= (name or "") + ".W_cell_to_outgate_peephole")

        outputgate = add(outputgate, peep_outgate)

    outputgate = NonlinearityLayer(
        outputgate,
        outputgate_nonlinearity,
        name=(name or "")+".outgate"
    )

    #cell output

    new_output= NonlinearityLayer(new_cell,
                                   output_nonlinearity,
                                   name=(name or "")+'.outgate_nonlinearity')


    new_output = mul(
        outputgate,
        new_output,
        name=(name or "")+'.outgate'
    )



    return new_cell, new_output
