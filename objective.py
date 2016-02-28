import theano.tensor as T
import theano
from auxilary import _shared
import numpy as np


from collections import namedtuple
ReferenceTuple = namedtuple("ReferenceTuple",["chosen_Qvalues_predicted", "reference_Qvalues","is_end_ravel"])

class BaseObjective:
    """
    instance, that:
        - determines Q-values for all actions given current agent state and current observation,
    """
    def __init__(self):
        raise NotImplemented
    def reset(self,batch_size):
        """
        performs this action each time a new session [batch] is loaded
            batch size: size of the new batch
        """
        pass
    def get_reward(self,last_environment_state,agent_action,batch_i):
        """
        WARNING! this function is computed on a session, not on a batch!
        reward given for taking the action in current environment state
        arguments:
            last_environment_state float[time_i, memory_id]: environment state before taking action
            action int[time_i]: agent action at this tick
        returns:
            reward float[time_i]: reward for taking action
        """
        
        return T.zeros_like(agent_action).astype(theano.config.floatX)

    def get_reward_sequences(self,env_state_sessions,agent_action_sessions):
        """
        computes the rewards given to agent at each time step for each batch
        parameters:
            env_state_seq - environment state [batch_i,seq_i,state_units] history for all sessions
            agent_action_seq - int[batch_i,seq_i]
        returns:
            rewards float[batch_i,seq_i] - what reward was given to an agent for corresponding action from state in that batch

        """
        
        
        
        def compute_reward(batch_i,session_states,session_actions,last_reward):
            return self.get_reward(session_states,session_actions,batch_i)



        sequences = [
            T.arange(env_state_sessions.shape[0],),
            env_state_sessions,
            agent_action_sessions,
        ]

        rewards,updates = theano.scan(compute_reward,
                              sequences=sequences,
                              outputs_info = [T.zeros( (env_state_sessions.shape[1],) )],
                              non_sequences = [])
        assert len(updates)==0
        return rewards.reshape(agent_action_sessions.shape) #reshape bach to original
    

    
    
    #reference tuples
    
    
    def get_action_Qvalues(self,Qvalues,actions):
        """auxilary function to select Qvalues corresponding to actions taken
        Returns Qvalues predicted that resulted in actions: float[batch,tick]"""
        
        batch_i = T.arange(Qvalues.shape[0])[:,None]
        time_i = T.arange(Qvalues.shape[1])[None,:]
        action_Qvalues_predicted= Qvalues[batch_i,time_i, actions]
        return action_Qvalues_predicted
    def get_end_indicator(self, is_alive):
        """ auxilary function to transform session alive indicator into end action indicator"""
        #session-ending action indicator: uint8[batch,tick]
        is_end = T.eq(is_alive[:,:-1] - is_alive[:,1:],1)
        is_end = T.concatenate(
            [is_end,
             T.ones((is_end.shape[0],1),dtype=is_end.dtype)],
            axis=1
        )
        return is_end
    
    
    def ravel_alive(self,is_alive,*args):
        """takes all is_alive ticks from all sessions and merges them into 1 dimension"""
        alive_selector = is_alive.nonzero()
        return [arg[alive_selector] for arg in args]
    
    default_gamma = _shared('gamma_default',np.float32(0.99), theano.config.floatX)
    
    
    def get_reference(self, Qvalues,actions,rewards,is_alive='always',
                                   gamma_or_gammas = default_gamma,
                                   aggregation_function = lambda qv:T.max(qv,axis=1)
                            ):
        """
        parameters:
        Qvalues [batch,tick,action_id] - predicted qvalues
        actions [batch,tick] - commited actions
        is_alive[batch,tick] - 1 if session is still active by this tick, 0 otherwise
        gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 

        Returns:
        computes three vectors:
        action IDs (1d,single integer) vector of all actions that were commited during all sequences
        in the batch, concatenated in one vector... that is, excluding the ones that happened
        after the end_code action was sent.
        Qpredicted - qvalues for action_IDs ONLY at each time 
        before sequence end action was committed for each sentence in the batch (concatenated) 
        but for the last time-slot.
        Qreference - sum over immediate rewards and gamma*predicted qvalues for
        next round after first vector predictions
        if naive == True, all actions are considered optimal in terms of Qvalue
        """
        
        if is_alive == 'always':
            is_alive = T.ones_like(actions,dtype='uint8')

        action_Qvalues_predicted = self.get_action_Qvalues(Qvalues,actions)

        is_end = self.get_end_indicator(is_alive)
        

        # Qvalues for "next" states (padded with zeros at the end): float[batch,tick,action]
        next_Qvalues_predicted = T.concatenate(
            [Qvalues[:,1:],
             T.zeros_like(Qvalues[:,0,None,:])
            ],
            axis=1
        )

        #"optimal next reward" after commiting action : float[batch,tick]
        ravel_Qnext = next_Qvalues_predicted.reshape([-1,next_Qvalues_predicted.shape[-1]])
        optimal_next_Qvalue = aggregation_function(ravel_Qnext).reshape(next_Qvalues_predicted.shape[:-1])



        # zero out future rewards after session end
        optimal_next_Qvalue = T.switch( is_end,
                                        0.,
                                        optimal_next_Qvalue
                                       )

        #full Qvalue formula (taking chosen_action and behaving optimally later on)
        reference_Qvalues = rewards + gamma_or_gammas*optimal_next_Qvalue
          

        return ReferenceTuple( action_Qvalues_predicted, reference_Qvalues,is_end)
    
    def get_reference_naive(self, Qvalues,actions,rewards,is_alive='always',
                                   gamma_or_gammas = default_gamma,
                                   dependencies=[],strict = True):
        """ computes Qvalues assuming all actions are optimal
        params:
            rewards: immediate rewards floatx[batch_size,time]
            is_alive: whether the session is still active int/bool[batch_size,time]
            gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]
            dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
            strict: whether to evaluate Qvalues using strict theano scan or non-strict one
        returns:
            Qvalues: floatx[batch_size,time]
                the regular Qvalue is computed via:
                    Qs(t,action_at(t) ) = _rewards[t] + _gamma_or_gammas* Qs(t+1, best_action_at(t+1))
                using out assumption on optimality:
                    best_action_at(t) = action(t)
                this essentially becomes:
                    Qs(t,picked_action) = _rewards[t] + _gamma_or_gammas* Qs(t+1,next_picked_action)
                and does not require actions themselves to compute (given rewards)

        """
        if is_alive == 'always':
            is_alive = T.ones_like(actions,dtype='uint8')

        action_Qvalues_predicted = self.get_action_Qvalues(Qvalues,actions)

        is_end = self.get_end_indicator(is_alive)
                            
                            
                            
        #recurrent computation of Qvalues reference (backwards through time)
                            
        outputs_info = [T.zeros_like(rewards[:,0]),]
        non_seqs = [gamma_or_gammas]+dependencies

        sequences = [rewards.T,is_alive.T] #transpose to iterate over time, not over batch

        def backward_qvalue_step(rewards,is_alive, next_Qs,*args):
            this_Qs = T.switch(is_alive,
                                   rewards + gamma_or_gammas * next_Qs, #assumes optimal next action
                                   0.
                              )
            return this_Qs

        reference_Qvalues = theano.scan(backward_qvalue_step,
                    sequences=sequences,
                    non_sequences=non_seqs,
                    outputs_info=outputs_info,

                    go_backwards=True,
                    strict = strict
                   )[0] #shape: [time_seq_inverted,batch]
        
        reference_Qvalues = reference_Qvalues.T[:,::-1] #[batch,time_seq]
        

        return ReferenceTuple(action_Qvalues_predicted, reference_Qvalues,is_end)

    
    
