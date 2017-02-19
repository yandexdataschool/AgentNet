"""
AgentNet - a library for deep reinforcement learning agent design and research
"""

from __future__ import division, print_function, absolute_import

from . import agent
from . import environment
from . import objective
from . import memory
from . import resolver
from . import utils
from .agent import Agent,Recurrence

__version__ = '0.10.3'
__author__ = 'YandexDataSchool and contributors.'


###Warnings verbosity:
class config:

    verbose=2
    # 0 = shut up
    # 1 = essential only
    # 2 = tell me everything

    #verbosity functions for fun
    @staticmethod
    def shut_up():
        """sets agentnet.verbose to 0"""
        config.verbose = 0

from warnings import warn as default_warn
def warn(message="Hello, world!",verbosity_level=1,**kwargs):
    if config.verbose >= verbosity_level:
        default_warn("[Verbose>=%s] %s"%(verbosity_level,message))

    #TODO make unsuppressable somehow

