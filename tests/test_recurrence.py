"""
tests for recurrence class
"""
import numpy as np
import theano
import agentnet
from agentnet.memory import RNNCell,GRUCell, LSTMCell
import lasagne
from lasagne.layers import *


def test_recurrence():
    """minimalstic test"""
    sequence = InputLayer((None, None, 3), name='input sequence')
    initial = InputLayer((None, 10), name='gru zero tick')

    # step
    inp = InputLayer((None, 3))
    prev_gru = InputLayer((None, 10))
    gru = GRUCell(prev_gru, inp, name='rnn')


    rec = agentnet.Recurrence(input_sequences={inp: sequence},
                              state_variables={gru: prev_gru},
                              state_init={gru: initial},  # defaults to zeros
                              unroll_scan=False)

    weights = get_all_params(rec)

    gru_states = rec[gru]

    run = theano.function([sequence.input_var, initial.input_var], get_output(gru_states), )

    assert tuple(run(np.random.randn(5, 25, 3), np.random.randn(5, 10)).shape) == (5, 25, 10)


def test_recurrence_larger():
    """larger recurrence"""
    sequence = InputLayer((None, None, 3), name='input sequence')
    initial_cell = InputLayer((None, 20), name='lstm cell zero tick')

    # step
    inp = InputLayer((None, 3))
    prev_rnn = InputLayer((None, 10))
    rnn = RNNCell(prev_rnn, inp, name='rnn')

    prev_lstm_cell = InputLayer((None,20)) #lstm cell
    prev_lstm_hid = InputLayer((None, 20)) #lstm output
    lstm_cell,lstm_hid = LSTMCell(prev_lstm_cell,prev_lstm_hid,input_or_inputs=rnn)

    lstm_hid = DropoutLayer(lstm_hid,p=0.5) #dropout hid, but not cell. Just to check it works

    from collections import OrderedDict #one can use regular dict but that causes a warning

    rec = agentnet.Recurrence(input_sequences={inp: sequence},
                              state_variables=OrderedDict({rnn: prev_rnn,
                                               lstm_hid:prev_lstm_hid,
                                                lstm_cell:prev_lstm_cell
                                               }),
                              state_init={lstm_cell: initial_cell},  # defaults to zeros
                              unroll_scan=False)

    weights = get_all_params(rec)

    rnn_states = rec[rnn]
    lstm_cell_states = rec[lstm_cell]
    lstm_hid_states = rec[lstm_hid]

    run = theano.function([sequence.input_var, initial_cell.input_var],
                          get_output([rnn_states,lstm_cell_states,lstm_hid_states]),
                          updates = rec.get_automatic_updates() #if any randomness is used AND unroll_scan,
                                                                # one has to pass automatic updates
                          )

    out = run(np.random.randn(5, 25, 3), np.random.randn(5, 20))

    assert tuple(out[0].shape) == (5, 25, 10) #rnn
    assert tuple(out[1].shape) == (5, 25, 20) #lstm cell
    assert tuple(out[2].shape) == (5, 25, 20) #lstm hid (aka output)


def test_recurrence_substituted():
    """test whether it is possible to use intermediate layers as recurrence inputs"""
    sequence = InputLayer((None, None, 3), name='input sequence')
    sequence_intermediate = InputLayer((None, None, 5), name='intermediate values sequence')
    initial = InputLayer((None, 10), name='gru zero tick')

    # step
    inp = InputLayer((None, 3),name='input')
    intermediate = DenseLayer(inp,5,name='intermediate')
    prev_gru = InputLayer((None, 10),name='prev rnn')
    gru = GRUCell(prev_gru, intermediate, name='rnn')

    #regular recurrence, provide inputs, intermediate is computed regularly
    rec = agentnet.Recurrence(input_sequences={inp: sequence},
                              state_variables={gru: prev_gru},
                              state_init={gru: initial},  # defaults to zeros
                              unroll_scan=False)

    weights = get_all_params(rec)
    assert intermediate.b in weights

    gru_states = rec[gru]

    run = theano.function([sequence.input_var, initial.input_var], get_output(gru_states), )

    assert tuple(run(np.random.randn(5, 25, 3), np.random.randn(5, 10)).shape) == (5, 25, 10)

    #recurrence with substituted intermediate values
    rec2= agentnet.Recurrence(input_sequences={intermediate: sequence_intermediate},
                              state_variables={gru: prev_gru},
                              state_init={gru: initial},  # defaults to zeros
                              unroll_scan=False)

    weights2 = get_all_params(rec2)
    assert intermediate.b not in weights2

    gru_states2 = rec2[gru]

    run = theano.function([sequence_intermediate.input_var, initial.input_var], get_output(gru_states2), )

    assert tuple(run(np.random.randn(5, 25, 5), np.random.randn(5, 10)).shape) == (5, 25, 10)



def test_recurrence_mask():
    """test mask_input"""
    np.random.seed(1337)

    sequence = InputLayer((None, None, 2), name='input sequence')
    mask = InputLayer((None, None), name="rnn mask [batch,tick]")

    # step
    inp = InputLayer((None, 2))
    prev_rnn = InputLayer((None, 3))
    rnn = RNNCell(prev_rnn, inp, name='rnn',
                  nonlinearity=lasagne.nonlinearities.linear,
                  b=lasagne.init.Constant(100.0))  # init with positive constant to make sure hiddens change

    out = DenseLayer(rnn,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)

    rec = agentnet.Recurrence(input_sequences={inp: sequence},
                              state_variables={rnn: prev_rnn},
                              tracked_outputs=[out],
                              unroll_scan=False,
                              mask_input=mask)

    rnn_states = rec[rnn]
    outs = rec[out]

    run = theano.function([sequence.input_var, mask.input_var], get_output([rnn_states,outs]))

    seq = np.random.randn(4, 5, 2)
    mask = np.zeros([4, 5])
    mask[:2, :3] = 1
    mask[2:, 2:] = 1

    h_seq, out_seq = run(seq, mask)

    assert tuple(h_seq.shape) == (4, 5, 3)
    assert tuple(out_seq.shape) == (4,5,10)

    diff_out = np.diff(h_seq, axis=1)
    assert np.all(np.diff(h_seq, axis=1)[:2, 2:] == 0)
    assert np.all(np.diff(h_seq, axis=1)[:2, :2] != 0)
    assert np.all(np.diff(h_seq, axis=1)[2:, 1:] != 0)
    assert np.all(np.diff(h_seq, axis=1)[2:, :1] == 0)

