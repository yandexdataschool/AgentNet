"""
Layers that convert policy or Q-value vectors into action ids
"""

from .base import *
from .epsilon_greedy import *
from .probabilistic import *

#alias
GreedyResolver = BaseResolver

