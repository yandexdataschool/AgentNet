"""
Layers that convert Q-value vectors into action ids
"""

from .base import *
from .epsilon_greedy import *
from .probabilistic import *

from ..deprecated import deprecated
ProbablisticResolver = deprecated("ProbabilisticResolver (with 'i')","0.1.0")(ProbabilisticResolver)
