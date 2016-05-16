__doc__="""
This module contains several helper functions used in various reinforcement learning algorithms.
"""


import theano
import theano.tensor as T
import numpy as np

from ..utils import insert_dim

from warnings import warn





#########################################
#N-step reference state/action values

def get_n_step_value_reference(state_values,
                               rewards,
                               is_alive="always",
                               n_steps = None,
                               gamma_or_gammas =0.99,
                               optimal_state_values = "same_as_state_values",
                               optimal_state_values_after_end = "zeros",     
                               dependencies=[],
                               strict = True):
    """
    Computes the reference for state value function via n-step algorithm:
    
    Vref = r(t) + gamma*r(t+1) + gamma^2*r(t+2) + ... + gamma^n*t(t+n) where n == n_steps
    
    Used by all n_step methods, including Q-learning, a2c and dpg
    
    Works with both Q-values and state values, depending on aggregation_function
    
    parameters:
        state_values - float[batch,tick] predicted state values V(s) at given batch session and time tick
        
        rewards - float[batch,tick] rewards achieved by commiting actions at [batch,tick]
    
        is_alive: whether the session is still active int/bool[batch_size,time]
        
        
        n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
        
        gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]
        
        optimal_state_values: state values given optimal actions.
            - for Q-learning, it's max over Q-values
            - for state-value based methods (a2c, dpg), it's same as state_values (defaults to that)
        
        
        optimal_state_values_after_end - symbolic expression for "next state values" for last tick used for reference only. 
                        Defaults at  T.zeros_like(values[:,0,None,:])
                        If you wish to simply ignore the last tick, 
                        use defaults and crop output's last tick ( qref[:,:-1] )
                        
        dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        
        strict: whether to evaluate values using strict theano scan or non-strict one
    returns:
        V reference [batch,action_at_tick] according n-step algorithms
        
            e.g. mentioned here http://arxiv.org/pdf/1602.01783.pdf as A3c and k-step Q-learning

    """
    
    
    #if we use state values and not action Q-values
        
    #handle aggregation function
    if optimal_state_values == "same_as_state_values":
        optimal_state_values = state_values

        
    #check dimensions
    if state_values.ndim != 2:
        if state_values.ndim ==3:
            warn("state_values must have shape [batch,tick] (ndim = 2).\n"\
                 "Currently assuming state_values you provided to have shape [batch, tick,1].\n"\
                 "Working with state_values[:,:,0].\n"\
                 "If that isn't what you intended, fix state_values shape to [batch,tick]\n")
            state_values = state_values[:,:,0]
        else:
            raise ValueError("state_values must have shape [batch,tick] (ndim = 2),"\
                             "while you have"+str(state_values.ndim))
        
        
        
            

    #fill default values
    if is_alive == "always":
        is_alive = T.ones_like(rewards)
        
    
        

    
    #get "Next state_values": floatx[batch,time] at each tick
    #do so by shifting state_values backwards in time, pad with state_values_after_end
    if optimal_state_values_after_end == "zeros":
        optimal_state_values_after_end = T.zeros_like(insert_dim(optimal_state_values[:,0],1))

    next_state_values = T.concatenate(
        [
            optimal_state_values[:,1:] * is_alive[:,1:],
            optimal_state_values_after_end,
        ],
        axis=1
    )
    
    
    
    #recurrent computation of reference state values (backwards through time)

    #initialize each reference with ZEROS after the end (won't be in output tensor)
    outputs_info = [T.zeros_like(rewards[:,0]),]   
    
    
    non_seqs = [gamma_or_gammas]+dependencies
    
    time_ticks = T.arange(rewards.shape[1])

    sequences = [rewards.T,
                 is_alive.T,
                 next_state_values.T,#transpose to iterate over time, not over batch
                 time_ticks] 

    def backward_V_step(rewards,
                        is_alive,
                        next_Vpred,
                        time_i, 
                        next_Vref,
                        *args):
        """scan inner computation step, going backwards in time
        params:
        rewards, is_alive, next_Vpred, time_i - sequences
        next_Vref - recurrent state value for next turn
        
        returns:
            current_Vref - recurrent state value at this turn
            
            current_Vref is computed thus:
                Once every n_steps or at session end:
                    current_Vref = r + gamma*next_Vpred   #computation through next predicted state value
                Otherwise:
                    current_Vref = r + gamma*next_Vref    #recurrent computation through next Qvalue
            
        """
        
        
        propagated_Vref = rewards + gamma_or_gammas * next_Vref #propagates value from actual next action 
        optimal_Vref = rewards + gamma_or_gammas *next_Vpred #uses agent's prediction for next state
        
        
        

                          
        if n_steps is None:
            chosen_Vref = propagated_Vref
        else:
            is_Tmax = T.eq(time_i % n_steps,0) #indicator for Tmax 
            
            #pick new_Vref if is_Tmax, else propagate existing one 
            chosen_Vref = T.switch(is_Tmax,
                                   optimal_Vref,
                                   propagated_Vref,)
        
        
        #zero out references if session has ended already
        this_Vref = T.switch(is_alive,
                             chosen_Vref,
                             0.)

                                 
        
        return this_Vref

    reference_state_values = theano.scan(backward_V_step,
                                    sequences=sequences,
                                    non_sequences=non_seqs,
                                    outputs_info=outputs_info,
                                    go_backwards=True,
                                    strict = strict
                                   )[0] #shape: [time_seq_inverted,batch]
        
    reference_state_values = reference_state_values.T[:,::-1] #[batch,time_seq]
    
    
    
    return reference_state_values





#####################################################
#minor helpers

def get_action_Qvalues(Qvalues,actions):
    """auxilary function to select Qvalues corresponding to actions taken
        Returns Qvalues predicted that resulted in actions: float[batch,tick]"""

    batch_i = T.arange(Qvalues.shape[0])[:,None]
    time_i = T.arange(Qvalues.shape[1])[None,:]
    action_Qvalues_predicted= Qvalues[batch_i,time_i, actions]
    return action_Qvalues_predicted

def get_end_indicator( is_alive,force_end_at_t_max = False):
    """ auxilary function to transform session alive indicator into end action indicator
    If force_end_at_t_max is True, all sessions that didn't end by the end of recorded sessions
    are ended at the last recorded tick."""
    #session-ending action indicator: uint8[batch,tick]
    is_end = T.eq(is_alive[:,:-1] - is_alive[:,1:],1)
    
    if force_end_at_t_max:
        session_ended_before = T.sum(is_end,axis=1,keepdims=True)
        is_end_at_tmax = 1 - T.gt(session_ended_before, 0 )
    else:
        is_end_at_tmax = T.zeros((is_end.shape[0],1),dtype=is_end.dtype) 
    
    is_end = T.concatenate(
        [is_end,
         is_end_at_tmax],
        axis=1
    )
    return is_end


def ravel_alive(is_alive,*args):
    """takes all is_alive ticks from all sessions and merges them into 1 dimension"""
    alive_selector = is_alive.nonzero()
    return [arg[alive_selector] for arg in args]

