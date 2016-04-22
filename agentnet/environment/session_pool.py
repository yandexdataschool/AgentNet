from theano import tensor as T


import numpy as np
import theano

from collections import OrderedDict


from base import BaseEnvironment
from session_batch import SessionBatchEnvironment

from ..objective import BaseObjective

from ..utils import create_shared,set_shared

class SessionPoolEnvironment(BaseEnvironment,BaseObjective):
    def __init__(self,rng_seed=1337):
        """
        A generic pseudo-environment that replays sessions loaded via .load_sessions(...),
        ignoring agent actions completely.
        The environment maintains it's own pool of sessions represented as (.observations, .actions, .rewards)
        
        To create experience-replay sessions, call Agent.get_sessions with this as an environment.
        During experience replay sessions,
         - states are replaced with a fake one-unit state
         - observations, actions and rewards match original ones
         - agent memory states, Qvalues and all in-agent expressions (but for actions) will correspond to what
           agent thinks NOW about the replay.
        
        
        Allthough it is possible to get rewards via the regular functions, it is usually faster to take self.rewards as rewards
        with no additional computation.
        
        """
        #setting environmental variables. Their shape is [batch_i,time_i,something]
        self.observations = create_shared("sessions.observations_history",np.zeros([10,5,1],dtype=theano.config.floatX))
        self.padded_observations = T.concatenate([self.observations,T.zeros_like(self.observations[:,0,None,:])],axis=1)

        self.actions = create_shared("session.actions_history",np.zeros([10,5]),dtype='int32')
        self.rewards = create_shared("session.rewards_history",np.zeros([10,5]),dtype=theano.config.floatX)
        
        
        self.is_alive = create_shared("session.is_alive",np.zeros([10,5]),dtype='uint8')
        
        #agent memory at state 0: floatX[batch_i,unit]
        self.preceding_agent_memory = create_shared("session.prev_memory",np.zeros([10,5]),dtype=theano.config.floatX)
        
        
        
        
        self.batch_size = self.pool_size = self.actions.shape[0]
        self.sequence_length =self.actions.shape[1]
        
        #rng used to .sample_session_batch
        self.rng = T.shared_randomstreams.RandomStreams(rng_seed)

        
    @property 
    def state_size(self):
        """Environment state size"""
        return 1
    @property 
    def observation_size(self):
        """Single observation size"""
        return self.padded_observations.shape[-1]
    
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
    
    
    
    
    def load_sessions(self,observation_seq,action_seq,reward_seq,is_alive=None,prev_memory=None):
        """
        loads a batch of sessions into env. The loaded sessions are that used during agent interactions
        """
        set_shared(self.observations,observation_seq)
        set_shared(self.actions,action_seq)
        set_shared(self.rewards,reward_seq)
        if is_alive is not None:
            set_shared(self.is_alive,is_alive)
        if prev_memory is not None:
            set_shared(self.preceding_agent_memory,prev_memory)
    
    def get_session_updates(self,observation_seq,action_seq,reward_seq,is_alive=None,prev_memory=None,cast_dtypes=True):
        """
        returns a dictionary of updates that will set shared variables to argument state
        is cast_dtypes is True, casts all updates to the dtypes of their respective variables
        """
        updates = OrderedDict({
            self.observations:observation_seq,
            self.actions:action_seq.astype(self.actions.dtype),
            self.rewards:reward_seq
        })
        if is_alive is not None:
            updates[self.is_alive]=is_alive
        if prev_memory is not None:
            updates[self.preceding_agent_memory] = prev_memory

        if cast_dtypes:
            casted_updates = OrderedDict({})
            for var,upd in updates.items():
                casted_updates[var] = upd.astype(var.dtype)
            updates = casted_updates
            
        return updates
    def select_session_batch(self,selector):
        """
        returns SessionBatchEnvironment with sessions (observations,actions,rewards)
        from pool at given indices
        
        Note that if this environment did not load is_alive or preceding_memory, 
        you won't be able to use them at the SessionBatchEnvironment
        
        
        """
        
        return SessionBatchEnvironment(self.observations[selector],self.actions[selector],self.rewards[selector],
                                       self.is_alive[selector],self.preceding_agent_memory[selector])

    def sample_session_batch(self,max_n_samples,replace=False):
        """
        returns SessionBatchEnvironment with sessions(observations,actions,rewards)
        that will be sampled uniformly from this session pool.
        if replace=False, the amount of samples is min(max_n_sample, current pool)
        Otherwise it equals max_n_samples
        
        The chosen session ids will be sampled at random using self.rng on each iteration
        p.s. no need to propagate rng updates! It does so by itself. 
        Unless you are calling it inside theano.scan, ofc, but i'd recomment that you didn't.
        unroll_scan works ~probably~ perfectly fine btw
        """
        if replace:
            n_samples = max_n_samples
        else:
            n_samples = T.minimum(max_n_samples,self.pool_size)
            
        sample_ids = self.rng.choice(size = (n_samples,), a = self.pool_size,dtype='int32',replace=replace)
        return self.select_session_batch(sample_ids)
        
        
        