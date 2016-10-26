"""
Layers that convert policy or Q-value vectors into action ids
"""

from .base import *
from .epsilon_greedy import *
from .probabilistic import *


def ProbabilisticResolver(*args,**kwargs):
    raise ValueError("Use Probabilistic resolver (with i)")

