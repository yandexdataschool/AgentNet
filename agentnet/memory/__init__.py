"""
This module contains a number of Lasagne layers useful when designing agent memory.
"""

from __future__ import division, print_function, absolute_import

from .gru import GRUMemoryLayer
from .stack import StackAugmentation
from .window import WindowAugmentation
from .rnn import RNNCell,GRUCell,LSTMCell
from .gate import GateLayer

