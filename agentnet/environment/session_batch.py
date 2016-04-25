from theano import tensor as T
from ..objective import BaseObjective
from ..environment import BaseEnvironment
import numpy as np
import theano

from collections import OrderedDict

from ..utils import create_shared,set_shared
from ..utils.format import check_list

class SessionBatchEnvironment(BaseEnvironment,BaseObjective):
    def __init__(self,observations,actions=None,rewards=None,is_alive=None,preceding_agent_memory=None):
        """
        A generic pseudo-environment that replays sessions loaded on creation,
        ignoring agent actions completely.
        
        The environment takes symbolic expression for sessions represented as (.observations, .actions, .rewards)
        Unlike SessionPoolEnvironment, this one does not store it's own pool of sessions.
        
        To create experience-replay sessions, call Agent.get_sessions with this as an environment.
        During experience replay sessions,
         - states are replaced with a fake one-unit state
         - observations, actions and rewards match original ones
         - agent memory states, Qvalues and all in-agent expressions (but for actions) will correspond to what
           agent thinks NOW about the replay.
         - is_alive [optional] - whether or not session has still not finished by a particular tick
         - preceding_agent_memory [optional] - what was agent's memory state prior to the first tick of the replay session.
         
        
        
        Allthough it is possible to get rewards via the regular functions, it is usually faster to take self.rewards as rewards
        with no additional computation.
        
        """
        
        #setting environmental variables. Their shape is [batch_i,time_i,something]
        self.observations = check_list(observations)
        if actions is not None:
            self.actions = check_list(actions)
        self.rewards = rewards
        self.is_alive = is_alive
        
        if preceding_agent_memory is not None:
            self.preceding_agent_memory = check_list(preceding_agent_memory)

        self.padded_observations = [
            T.concatenate([obs,T.zeros_like(obs[:,0,None])],axis=1)
            for obs in self.observations
        ]

        self.batch_size = self.observations[0].shape[0]
        self.sequence_length =self.observations[0].shape[1]
    @property 
    def state_size(self):
        """Environment state size"""
        return []
    @property 
    def observation_size(self):
        """Single observation size"""
        return [obs.shape[-1] for obs in self.padded_observations]
    
    def get_action_results(self,last_state,action,time_i):
        """
        computes environment state after processing agent's action
        arguments:
            last_state float[batch_id, memory_id0,[memory_id1],...]: environment state on previous tick
            action int[batch_id]: agent action after observing last state
        returns:
            new_state float[batch_id, memory_id0,[memory_id1],...]: environment state after processing agent's action
            observation float[batch_id,n_agent_inputs]: what agent observes after commiting the last action
        """
        return [],[obs[:,time_i+1] for obs in self.padded_observations]
        
    def get_reward(self,session_states,session_actions,batch_i):
        """
        WARNING! this runs on a single session, not on a batch
        reward given for taking the action in current environment state
        arguments:
            session_states float[batch_id, memory_id]: environment state before taking action
            session_actions int[batch_id]: agent action at this tick
        returns:
            reward float[batch_id]: reward for taking action from the given state
        """
        return self.rewards[batch_i,:]
