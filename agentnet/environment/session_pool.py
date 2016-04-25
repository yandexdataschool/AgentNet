from theano import tensor as T


import numpy as np
import theano

from collections import OrderedDict


from base import BaseEnvironment
from session_batch import SessionBatchEnvironment

from ..objective import BaseObjective

from ..utils import create_shared,set_shared

class SessionPoolEnvironment(BaseEnvironment,BaseObjective):
    def __init__(self,n_observations =1,
                 n_actions=1,action_dtypes=['int32'],
                 n_agent_memories = 1,
                 rng_seed=1337):
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
        self.observations = [
            create_shared("sessions.observations_history",np.zeros([10,5,1],dtype=theano.config.floatX))
            for i in range(n_observations)
            ]
        self.padded_observations = [
            T.concatenate([obs,T.zeros_like(self.observations[:,0,None,:])],axis=1)
            for obs in self.observations
            ]

        
        if len(action_dtypes) > n_actions:
            action_dtypes = action_dtypes[:n_actions]
        elif len(action_dtypes) < n_actions:
            action_dtypes += action_dtypes[-1:]*(n_actions - len(action_dtypes))
            
        self.actions = [
            create_shared("session.actions_history",np.zeros([10,5]),dtype=dtype)
            for i,dtype in zip(range(n_actions),action_dtypes)
            ]
        
        
        self.rewards = create_shared("session.rewards_history",np.zeros([10,5]),dtype=theano.config.floatX)
        
        
        self.is_alive = create_shared("session.is_alive",np.zeros([10,5]),dtype='uint8')
        
        #agent memory at state 0: floatX[batch_i,unit]
        self.preceding_agent_memories = [
            create_shared("session.prev_memory",np.zeros([10,5]),dtype=theano.config.floatX)
            for i in range(n_agent_memories)
        ]
        
        
        
        self.batch_size = self.pool_size = self.rewards.shape[0]
        self.sequence_length =self.rewards.shape[1]
        
        #rng used to .sample_session_batch
        self.rng = T.shared_randomstreams.RandomStreams(rng_seed)

        
    @property 
    def state_size(self):
        """Environment state size"""
        return []
    @property 
    def observation_size(self):
        """Single observation size"""
        return [obs.shape[-1] for obs in self.padded_observations]
    
    def get_action_results(self,last_states,actions,time_i):
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
    
    
    
    
    def load_sessions(self,observation_sequences,action_sequences,reward_seq,is_alive=None,prev_memories=None):
        """
        loads a batch of sessions into env. The loaded sessions are that used during agent interactions
        """
        
        assert len(observation_sequences) == len(self.observations)
        assert len(action_sequences) == len(self.actions)
        if prev_memories is not None:
            assert len(prev_memories) == len(self.preceding_agent_memories)
        
        for observation_var,observation_seq in zip(self.observations,observation_sequences):
            set_shared(observation_var,observation_seq)
        for action_var, action_seq in zip(self.actions,action_sequences):
            set_shared(action_var,action_seq)
            
        set_shared(self.rewards,reward_seq)
        
        if is_alive is not None:
            set_shared(self.is_alive,is_alive)
        
        if prev_memories is not None:
            for prev_memory_var,prev_memory_value in zip(self.preceding_agent_memories,prev_memories):
                set_shared(prev_memory_var,prev_memory)
    
    def get_session_updates(self,observation_seq,action_seq,reward_seq,is_alive=None,prev_memory=None,cast_dtypes=True):
        """
        returns a dictionary of updates that will set shared variables to argument state
        is cast_dtypes is True, casts all updates to the dtypes of their respective variables
        """
        assert len(observation_sequences) == len(self.observations)
        assert len(action_sequences) == len(self.actions)
        if prev_memories is not None:
            assert len(prev_memories) == len(self.preceding_agent_memories)
        
        
        updates = OrderedDict()
        
        for observation_var,observation_seq in zip(self.observations,observation_sequences):
            updates[observation_var] = observation_seq
        for action_var, action_seq in zip(self.actions,action_sequences):
            updates[action_var] = action_seq
            
        updates[self.rewards] = reward_seq
        
        if is_alive is not None:
            updates[self.is_alive] = is_alive
        
        if prev_memories is not None:
            for prev_memory_var,prev_memory_value in zip(self.preceding_agent_memories,prev_memories):
                updates[prev_memory_var] = prev_memory
                
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
        selected_observations = [ observation_seq[selector] for observation_seq in self.observations]
        selected_actions = [ action_seq[selector] for action_seq in self.actions]
        selected_prev_memories = [ prev_memory[selector] for prev_memory in self.preceding_agent_memory]
        
        return SessionBatchEnvironment(selected_observations,selected_actions,self.rewards[selector],
                                       self.is_alive[selector],selected_prev_memories)

    def sample_session_batch(self,max_n_samples,replace=False,selector_dtype='int32'):
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
            
        sample_ids = self.rng.choice(size = (n_samples,), a = self.pool_size,dtype=selector_dtype,replace=replace)
        return self.select_session_batch(sample_ids)
        
        
        