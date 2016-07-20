"""
The Agent abstraction and core AgentNet functionality lies here.
"""
from __future__ import division, print_function, absolute_import

from .mdp_agent import MDPAgent
from .recurrence import Recurrence

# alias for MDP agent
Agent = MDPAgent
