"""
This test tries to make sure that LSTM and GRU implementations can indeed be easily transferred between lasagne and agentnet.
"""
from __future__ import print_function
from lasagne.layers import InputLayer, GRULayer, LSTMLayer, get_output, get_all_params
from agentnet.memory import LSTMCell, GRUCell
from agentnet.agent import Recurrence
from collections import OrderedDict

import theano.tensor as T
import theano
import numpy as np


def test_memory_cells(batch_size=3, seq_len=50, input_dim=8, n_hidden=16):
    # lasagne way
    l_in = InputLayer((None, seq_len, input_dim),
                      input_var=theano.shared(np.random.normal(size=[batch_size, seq_len, input_dim])),
                      name='input seq')

    l_lstm0 = LSTMLayer(l_in, n_hidden, name='lstm')
    l_gru0 = GRULayer(l_in, n_hidden, name='gru')

    f_predict0 = theano.function([], get_output([l_lstm0, l_gru0]))

    # agentnet way
    s_in = InputLayer((None, input_dim), name='in')

    s_prev_cell = InputLayer((None, n_hidden), name='cell')
    s_prev_hid = InputLayer((None, n_hidden), name='hid')
    s_lstm_cell, s_lstm_hid = LSTMCell(s_prev_cell, s_prev_hid, s_in, name='lstm')

    s_prev_gru = InputLayer((None, n_hidden), name='hid')
    s_gru = GRUCell(s_prev_gru, s_in, name='gru')

    rec = Recurrence(state_variables=OrderedDict({
        s_lstm_cell: s_prev_cell,
        s_lstm_hid: s_prev_hid,
        s_gru: s_prev_gru}),
        input_sequences={s_in: l_in},
        unroll_scan=False)

    state_seqs, _ = rec.get_sequence_layers()

    l_lstm1 = state_seqs[s_lstm_hid]
    l_gru1 = state_seqs[s_gru]

    f_predict1 = theano.function([], get_output([l_lstm1, l_gru1]))

    # lstm param transfer
    old_params = sorted(get_all_params(l_lstm0, trainable=True), key=lambda p: p.name)
    new_params = sorted(get_all_params(s_lstm_hid, trainable=True), key=lambda p: p.name)

    for old, new in zip(old_params, new_params):
        print (old.name, '<-', new.name)
        assert tuple(old.shape.eval()) == tuple(new.shape.eval())
        old.set_value(new.get_value())

    # gru param transfer
    old_params = sorted(get_all_params(l_gru0, trainable=True), key=lambda p: p.name)
    new_params = sorted(get_all_params(s_gru, trainable=True), key=lambda p: p.name)

    for old, new in zip(old_params, new_params):
        print (old.name, '<-', new.name)
        assert tuple(old.shape.eval()) == tuple(new.shape.eval())
        old.set_value(new.get_value())

    lstm0_out, gru0_out = f_predict0()
    lstm1_out, gru1_out = f_predict1()

    assert np.allclose(lstm0_out, lstm1_out)
    assert np.allclose(gru0_out, gru1_out)
