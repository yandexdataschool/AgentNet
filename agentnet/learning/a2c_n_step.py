
__doc__="""N-step Advantage Actor-Critic (A2c) implementation.\nWorks with action probabilities and state values instead of Q-values"""

from warnings import warn

import theano
import theano.tensor as T
import numpy as np


from lasagne.objectives import squared_error


from ..utils.mdp import get_end_indicator, get_action_Qvalues
from ..utils.grad import consider_constant
from ..utils import create_shared


default_gamma = create_shared('a3c_gamma_default',np.float32(0.99), theano.config.floatX)
    
def get_state_value_reference(state_values,rewards,
             is_alive="always",
             n_steps = None,
             gamma_or_gammas = default_gamma,
             state_values_after_end = "zeros",     
             dependencies=[],strict = True):
    """
    Computes the reference for state value function according to Advantage Actor-Critic (a2c) learning algorithm
    
    parameters:
        state_values - float[batch,tick] predicted state values V(s) at given batch session and time tick
        rewards - float[batch,tick] rewards achieved by commiting actions at [batch,tick]
    
        is_alive: whether the session is still active int/bool[batch_size,time]
        
        qvalues_after_end - symbolic expression for "next state q-values" for last tick used for reference only. 
                        Defaults at  T.zeros_like(values[:,0,None,:])
                        If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
        
        n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
            
        
        gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]
        
        aggregation_function - a function that takes all values for "next state qvalues" term and 
                    returns what is the "best next Qvalue" at the END of n-step cycle. 
                    Normally you should not touch it.

        dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        strict: whether to evaluate values using strict theano scan or non-strict one
    returns:
        Q reference [batch,action_at_tick] according to N-step Q_learning
        mentioned here http://arxiv.org/pdf/1602.01783.pdf as Algorithm 3

    """
    
    
    if state_values.ndim != 2:
        if state_values.ndim ==3:
            warn("state_values must have shape [batch,tick] (ndim = 2).\n"\
                 "Assuming state_values you provided to have shape [batch, tick,1].\n"\
                 "Working with state_values[:,:,0].\n"\
                 "If that isn't what you intended, fix state_values shape to [batch,tick]\n")
            state_values = state_values[:,:,0]
        else:
            raise ValueError("state_values must have shape [batch,tick] (ndim = 2), while you have"+str(state_values.ndim))
    if is_alive == "always":
        is_alive = T.ones_like(rewards)
        
    
    if state_values_after_end == "zeros":
        state_values_after_end = T.zeros([state_values.shape[0],1])
        

    #get "Next state_values": floatx[batch,time] at each tick, pad with state_values_after_end
    next_state_values = T.concatenate(
        [
            state_values[:,1:] * is_alive[:,1:],
            state_values_after_end,
        ],
        axis=1
    )
    
    
    
    #recurrent computation of reference state values (backwards through time)

    #initialize each reference with ZEROS after the end (won't be in output tensor)
    outputs_info = [T.zeros_like(rewards[:,0]),]   
    
    
    non_seqs = [gamma_or_gammas]+dependencies
    
    time_ticks = T.arange(rewards.shape[1])

    sequences = [rewards.T,is_alive.T,next_state_values.T,#transpose to iterate over time, not over batch
                 time_ticks] 

    def backward_V_step(rewards,is_alive,next_Vpred,time_i, 
                        next_Vref,
                        *args):
        
        
        
        propagated_Vref = T.switch(is_alive,
                           rewards + gamma_or_gammas * next_Vref, #assumes optimal next action
                           0.
                          )
        
        if n_steps is None:
            this_Vref = propagated_Vref
        else:
            
            Vref_at_tmax = T.switch(is_alive,
                           rewards + gamma_or_gammas *next_Vpred,
                           0.
                          )
            
            this_Vref = T.switch(T.eq(time_i % n_steps,0), #if Tmax
                                        Vref_at_tmax,  #use special case values
                                        propagated_Vref #else use generic ones
                                   )
                                         
                                 
        
        
        
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






    
def _get_objective(policy,state_values,actions,reference_state_values,
                         is_alive = "always",min_log_proba = -1e50):
    """returns a2v loss sum"""
    if is_alive == "always":
        is_alive = T.ones_like(actions,dtype=theano.config.floatX)
        
    
    action_probas =  get_action_Qvalues(policy,actions)
    
    reference_state_values = consider_constant(reference_state_values)
    
    log_probas = T.maximum(T.log(action_probas),min_log_proba)
    
    policy_loss_elwise = - log_probas * (reference_state_values - consider_constant(state_values))
    
    
    V_err_elwise = squared_error(reference_state_values,state_values)
    
    return (policy_loss_elwise + V_err_elwise)*is_alive
    

    
    

    
def get_elementwise_objective(policy,state_values,actions,rewards,
                              is_alive = "always",
                              n_steps = None,
                              gamma_or_gammas = 0.95,
                              force_values_after_end = True,
                              state_values_after_end = "zeros",
                              consider_value_reference_constant = True,
                              consider_predicted_value_constant=True,
                              scan_dependencies = [], scan_strict = True,
                              min_log_proba = -1e50):
    """
    returns crossentropy-like objective function for Actor-Critic method

        L_policy = - log(policy)*(Vreference - const(V))
        L_V = (V - Vreference)^2
            
    parameters:
    
        policy [batch,tick,action_id] - predicted action probabilities
        state_values [batch,tick] - predicted state values
        actions [batch,tick] - commited actions
        rewards [batch,tick] - immediate rewards for taking actions at given time ticks
        
        is_alive [batch,tick] - whether given session is still active at given tick. Defaults to always active.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss
        
        n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
        
        gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
        
        force_values_after_end - if true, sets reference policy at session end to rewards[end] + qvalues_after_end
        
        state_values_after_end[batch,1,n_actions] - "next state values" for last tick used for reference only. 
                            Defaults at  T.zeros_like(state_values[:,0,None,:])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )

        
        
        scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        scan_strict: whether to evaluate values using strict theano scan or non-strict one
        
    Returns:
                
        elementwise sum of policy_loss + state_value_loss

    """
    
    
    #get reference values via Q-learning algorithm
    reference_state_values = get_state_value_reference(state_values,rewards,is_alive,
                                              n_steps = n_steps,
                                              state_values_after_end = state_values_after_end,
                                              gamma_or_gammas = gamma_or_gammas,
                                              dependencies = scan_dependencies,
                                              strict = scan_strict
                                             )    
    
    
    #if agent is always alive, return the simplified loss
    if is_alive == "always":
        
        return _get_objective(policy,state_values,actions,reference_state_values,
                         is_alive = is_alive,min_log_proba = min_log_proba)
            

        
        
    else: #we are given an is_alive matrix : uint8[batch,tick] 

        #if asked to force reference_Q[end_tick+1,a] = 0, do it
        #note: if agent is always alive, this is meaningless
        
        if force_values_after_end:
            #set future rewards at session end to rewards+qvalues_after_end
            end_ids = get_end_indicator(is_alive,force_end_at_t_max = True).nonzero()

            if state_values_after_end == "zeros":
                # "set reference state values at end action ids to just the immediate rewards"
                reference_state_values = T.set_subtensor(reference_state_values[end_ids],
                                                    rewards[end_ids]
                                                    )
            else:
            
                # "set reference state values at end action ids to the immediate rewards + qvalues after end"
                reference_state_values = T.set_subtensor(reference_state_values[end_ids],
                                                    rewards[end_ids] + gamma_or_gammas*state_values_after_end[end_ids[0],0]
                                                    )
        
    
        return _get_objective(policy,
                              state_values,actions,reference_state_values,
                              is_alive = is_alive,
                              min_log_proba = min_log_proba)
    