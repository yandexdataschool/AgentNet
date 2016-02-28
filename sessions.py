from theano import tensor as T
from objective import BaseObjective
from environment import BaseEnvironment
import numpy as np
import theano
from auxilary import _shared,set_shared

class SessionEnvironment(BaseEnvironment,BaseObjective):
    def __init__(self):
        """
        A generic pseudo-environment that replays sessions loaded via .load_sessions(...),
        ignoring agent actions completely.
        """
        #setting environmental variables. Their shape is [batch_i,time_i,something]
        self.observations = _shared("sessions.observations_history",np.zeros([10,5,1],dtype=theano.config.floatX))
        self.padded_observations = T.concatenate([self.observations,T.zeros_like(self.observations[:,0,None,:])],axis=1)

        self.actions = _shared("session.actions_history",np.zeros([10,5],dtype='int32'))
        self.rewards = _shared("session.rewards_history",np.zeros([10,5],dtype=theano.config.floatX))
        self._batch_size = self.actions.shape[0]
        self._sequence_length =self.actions.shape[1]
    @property 
    def state_size(self):
        """must return integer(not symbolic)"""
        return 1
    @property 
    def observation_size(self):
        """must return integer(not symbolic)"""
        return self.padded_observations.shape[2]
    
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
        return last_state,self.padded_observations[:,time_i+1]
    def load_sessions(self,observation_seq,action_seq,reward_seq):
        """
        loads a batch of sessions into env. The loaded sessions are that used during agent interactions
        """
        set_shared(self.observations,observation_seq)
        set_shared(self.actions,action_seq)
        set_shared(self.rewards,reward_seq)
        
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
