"""
This module contains a number of Lasagne layers useful when designing agent memory.

Memory layers can be vaguely divided into classical recurrent layers (e.g. RNN, LSTM, GRU)
and augmentations (Stack augmentation, window augmentation, etc.).

Technically memory layers are lasagne layers that take previous memory state and some optional inputs
to return new memory state.

For example, to build RNN with 36 units one has to define

#RNN input
rnn_input = some_lasagne_layer

#rnn state from previous tick
prev_rnn = InputLayer( (None,36) ) #None for batch size

#new RNN state (i.e. sigma(Wi * rnn_input + Wh * prev_rnn + b) )
new_rnn = RNNCell(prev_rnn, rnn_input)


When using inside Agent (MDPAgent) or Recurrence, one must register them as
agent_states (for agent) or state_variables (for recurrence), e.g.


from agentnet.agent import Agent
agent = Agent(observations,{new_rnn : prev_rnn},...)
"""

from __future__ import division, print_function, absolute_import
from .stack import StackAugmentation
from .window import WindowAugmentation
from .rnn import RNNCell,GRUCell,LSTMCell
from .gate import GateLayer

from .gru import GRUMemoryLayer
from .logical import  CounterLayer,SwitchLayer
from .attention import AttentionLayer,DotAttentionLayer

