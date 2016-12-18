import agentnet,lasagne
import theano
from theano import tensor as T
import numpy as np
from agentnet import Recurrence
from lasagne.layers import *
from agentnet.memory import *
from agentnet.resolver import EpsilonGreedyResolver

def test_out_batch1():
    """minimalstic test for batch_size=1,
    https://github.com/yandexdataschool/AgentNet/issues/79"""
    # step
    prev_out = InputLayer((None,))
    prev_gru = InputLayer((None, 10))
    gru = GRUCell(prev_gru, EmbeddingLayer(prev_out, 3, 3), name='rnn')
    probs = DenseLayer(gru, 3, nonlinearity=lasagne.nonlinearities.softmax)
    out = EpsilonGreedyResolver(probs)

    batch_size = 1

    rec = agentnet.Recurrence(state_variables={gru: prev_gru,
                                               out: prev_out,},
                              unroll_scan=False,
                              n_steps=5,
                              batch_size=batch_size)

    run = theano.function([], get_output(rec[out]), updates=rec.get_automatic_updates())

    assert tuple(run().shape) == (1, 5)

def test_grad_1unit():
    """make sure graph inputs and outputs are of same shape with 1-unit output
    https://github.com/yandexdataschool/AgentNet/issues/83"""
    l_in = InputLayer((None,None,5))
    step_inp = InputLayer((None,5))
    step_out = DenseLayer(step_inp,1,name='dense')

    rec = Recurrence(input_sequences={step_inp:l_in},
                     tracked_outputs=[step_out],
                     unroll_scan=False)
    out_seq = rec[step_out]


    params = get_all_params(step_out,trainable=True)
    grad = T.grad(get_output(out_seq).mean(),params)
    #check if grad does not fail on single-unit layer due to broadcastability.
    f = theano.function([l_in.input_var],grad)
    f(np.zeros([2,3,5]))