from theano import tensor as T


import numpy as np
import theano

from collections import OrderedDict


from .base import BaseEnvironment
from .session_batch import SessionBatchEnvironment

from ..objective import BaseObjective

from ..utils import create_shared,set_shared,insert_dim
from ..utils.format import check_list
from ..utils.layers import get_layer_dtype

from warnings import warn

class SessionPoolEnvironment(BaseEnvironment,BaseObjective):
    def __init__(self,observations =1,
                 actions=1,
                 agent_memories = 1,
                 default_action_dtype="int32",
                 rng_seed=1337):
        """
        A generic pseudo-environment that replays sessions loaded via .load_sessions(...),
        ignoring agent actions completely.
        
        It has a single scalar integer env_state, corresponding to time tick.
        
        The environment maintains it's own pool of sessions represented as (.observations, .actions, .rewards)
        
        
        parameters:
         - observations - number of floatX flat observations or a list of observation inputs to mimic
         - actions - number of int32 scalar actions or a list of resolvers to mimic
         - agent memories
            
        To setup custom dtype, set the .output_dtype property of layers you send as actions, observations of memories.
        
        
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
        
        
        #observations
        if type(observations) is int:
            observation_init = np.zeros([10,5,2])
            self.observations = [
                create_shared("sessions.observations_history."+str(i),
                              observation_init,
                              dtype=theano.config.floatX)
                for i in range(observations)
            ]
        else:
            observations = check_list(observations)
            self.observations = [
                create_shared(
                    "sessions.observations_history."+str(i),
                    np.zeros(    
                        (10,5)+tuple(obs.output_shape[1:]),
                        dtype=  get_layer_dtype(obs)
                    ) 
                )
                for i,obs in enumerate(observations)
            ]
 
        #padded observations (to avoid index error when interacting with agent)
        self.padded_observations = [
            T.concatenate([obs,T.zeros_like(insert_dim(obs[:,0],1))],axis=1)
            for obs in self.observations
            ]
        
        
        
        #actions log
        if type(actions) is int:
            self.actions = [
                create_shared("session.actions_history."+str(i),np.zeros([10,5]),dtype=default_action_dtype)
                for i in range(actions)
            ]
            
        else:
            actions = check_list(actions)
            self.actions = [
                create_shared(
                    "session.actions_history."+str(i),
                    np.zeros((10,5)+tuple(action.output_shape[1:])),
                    dtype= get_layer_dtype(action,default_action_dtype)
                )
                for i,action in enumerate(actions)
            ]

        
            
        
        #agent memory at state 0: floatX[batch_i,unit]
        if type(agent_memories) is int:
            memory_init = np.zeros([10,5])
            self.preceding_agent_memories = [
                create_shared("session.prev_memory."+str(i),
                              memory_init,
                              dtype=theano.config.floatX)
                for i in range(agent_memories)
            ]
            
        else:
            agent_memories = check_list(agent_memories)
            
            self.preceding_agent_memories = [
                create_shared(
                    "session.prev_memory."+str(i),
                    np.zeros((10,5)+tuple(mem.output_shape[1:]),
                             dtype= get_type(mem)
                    ) 
                )
                for i,mem in enumerate(agent_memories)
            ]
            

        #rewards
        self.rewards = create_shared("session.rewards_history",np.zeros([10,5]),dtype=theano.config.floatX)
        
        #is_alive
        self.is_alive = create_shared("session.is_alive",np.ones([10,5]),dtype='uint8')
        
        
        
        #shapes
        self.batch_size = self.pool_size = self.rewards.shape[0]
        self.sequence_length =self.rewards.shape[1]
        
        #rng used to .sample_session_batch
        self.rng = T.shared_randomstreams.RandomStreams(rng_seed)

        
    @property 
    def state_shapes(self):
        """Environment state sizes. In this case, it's a timer"""
        return [tuple()]
    @property
    def state_dtypes(self):
        """environment state dtypes. In this case, it's a timer"""
        return ["int32"]
    
    
    @property 
    def observation_shapes(self):
        """observation shapes"""
        return [obs.get_value().shape[2:] for obs in self.observations]
    
    @property 
    def observation_dtypes(self):
        """observation dtypes"""
        return [obs.dtype for obs in self.observations]
    @property 
    def action_shapes(self):
        """action shapes"""
        return [act.get_value().shape[2:] for act in self.actions]
    @property 
    def action_dtypes(self):
        """action dtypes"""
        return [act.dtype for act in self.actions]
    
    
    
    
    
    def get_action_results(self,last_states,actions):
        """
        computes environment state after processing agent's action
        arguments:
            last_state float[batch_id, memory_id0,[memory_id1],...]: environment state on previous tick
            action int[batch_id]: agent action after observing last state
        returns:
            new_state float[batch_id, memory_id0,[memory_id1],...]: environment state after processing agent's action
            observation float[batch_id,n_agent_inputs]: what agent observes after commiting the last action
        """
        time_i = check_list(last_states)[0]
        
        batch_range = T.arange(time_i.shape[0])
        
        new_observations = [obs[batch_range,time_i+1] 
                               for obs in self.padded_observations]
        return [time_i+1],new_observations
        
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
        warn("Warning - a sesssion pool has all the rewards already stored as .rewards property."\
             "Recomputing them this way is probably just a slower way of calling your_session_pool.rewards")
        return self.rewards[batch_i,:]
    
    
    
    
    def load_sessions(self,observation_sequences,action_sequences,reward_seq,is_alive=None,prev_memories=None):
        """
        loads a batch of sessions into env. The loaded sessions are that used during agent interactions
        """
        observation_sequences = check_list(observation_sequences)
        action_sequences = check_list(action_sequences)
        
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
                set_shared(prev_memory_var,prev_memory_value)
    
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
            for var,upd in list(updates.items()):
                casted_updates[var] = upd.astype(var.dtype)
            updates = casted_updates
            
        return updates
    def select_session_batch(self,selector):
        """
        returns SessionBatchEnvironment with sessions (observations,actions,rewards)
        from pool at given indices
        parameters:
            selector - an array of integers that contains all indices of sessions to take.
        
        Note that if this environment did not load is_alive or preceding_memory, 
        you won't be able to use them at the SessionBatchEnvironment
        
        
        """
        selected_observations = [ observation_seq[selector] for observation_seq in self.observations]
        selected_actions = [ action_seq[selector] for action_seq in self.actions]
        selected_prev_memories = [ prev_memory[selector] for prev_memory in self.preceding_agent_memories]
        
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
        
        
        