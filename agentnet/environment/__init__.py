"""

Environment is an MDP abstraction that defines which observations does agent get and how does environment external state
 change given agent actions and previous state.

There's a base class for environment definition, as well as special environments for Experience Replay.

When designing your own experiment,
 - if it is done entirely in theano, implement BaseEnvironment. See ./experiments/wikicat or boolean_reasoning for example.
 - if it isn't (which is probably the case), use SessionPoolEnvironment to train from recorded interactions as in Atari examples

"""
from __future__ import division, print_function, absolute_import

from .base import BaseEnvironment, EnvironmentStepLayer
from .session_pool import SessionPoolEnvironment
from .session_batch import SessionBatchEnvironment

