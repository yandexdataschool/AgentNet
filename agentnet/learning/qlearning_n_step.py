
__doc__="""N-step Q-learning implementation. Works with discrete action space"""

import theano.tensor as T
import theano
import numpy as np

from lasagne.objectives import squared_error

from ..utils.grad import consider_constant

from .helpers import get_n_step_value_reference,get_end_indicator, get_action_Qvalues


    
def get_elementwise_objective(Qvalues,actions,rewards,
                              is_alive = "always",
                              n_steps = None,
                              gamma_or_gammas = 0.95,
                              force_qvalues_after_end = True,
                              optimal_qvalues_after_end = "zeros",
                              consider_reference_constant = True,
                              aggregation_function = lambda qv:T.max(qv,axis=-1),
                              return_reference = False,
                              scan_dependencies = [],
                              scan_strict = True):
    """
    Returns squared error between predicted and reference Qvalues according to Q-learning algorithm
    
        Qreference(state,action) = reward(state,action) + gamma* max[next_action]( Q(next_state,next_action)  
        loss = mean over (Qvalues - Qreference)**2
        
    parameters:
    
        Qvalues [batch,tick,action_id] - predicted qvalues
        actions [batch,tick] - commited actions
        rewards [batch,tick] - immediate rewards for taking actions at given time ticks
        
        is_alive [batch,tick] - whether given session is still active at given tick. Defaults to always active.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss
        
        n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
        
        gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
        
        force_qvalues_after_end - if true, sets reference Qvalues at session end to rewards[end] + qvalues_after_end
        
        optimal_qvalues_after_end [batch,1] - symbolic expression for "best next state q-values" for last tick 
                            used when computing reference Q-values only. 
                            Defaults at  T.zeros_like(Qvalues[:,0,None,0])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )

        consider_reference_constant - whether or not zero-out gradient flow through reference_Qvalues (True highly recommended)

        aggregation_function - a function that takes all Qvalues for "next state qvalues" term and returns what 
                                is the "best next Qvalue". Normally you should not touch it. Defaults to max over actions.
                                Normaly you shouldn't touch this
                                Takes input of [batch,n_actions] Qvalues
                                
        return_reference - if True, returns reference Qvalues. If False, returns squared_error(action_Qvalues, reference_Qvalues)


        scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        scan_strict: whether to evaluate Qvalues using strict theano scan or non-strict one
    Returns:
                
        mean squared error over Qvalues (using formua above for loss)

    """

    
    
    #get Qvalues of best actions (used every K steps for reference Qvalue computation
    optimal_Qvalues = aggregation_function(Qvalues)
    
    #get predicted qvalues for commited actions 
    #(to compare with reference Qvalues and use for recurrent reference computation)
    action_Qvalues = get_action_Qvalues(Qvalues,actions)

    
    
    #get reference Qvalues via Q-learning algorithm
    
    
    
    reference_Qvalues = get_n_step_value_reference(
        state_values = action_Qvalues,
        rewards = rewards,
        is_alive = is_alive,
        n_steps = n_steps,
        gamma_or_gammas = gamma_or_gammas,
        optimal_state_values = optimal_Qvalues,
        optimal_state_values_after_end = optimal_qvalues_after_end,
        dependencies = scan_dependencies,
        strict = scan_strict
    )
    
    if consider_reference_constant:
        #do not pass gradient through reference Qvalues (since they DO depend on Qvalues by default)
        reference_Qvalues = consider_constant(reference_Qvalues)
    

    
    if force_qvalues_after_end and is_alive != "always":
        #if asked to force reference_Q[end_tick+1,a] = 0, do it
        #note: if agent is always alive, this is meaningless
        #set future rewards at session end to rewards+qvalues_after_end
        end_ids = get_end_indicator(is_alive,force_end_at_t_max = True).nonzero()

        if optimal_qvalues_after_end == "zeros":
            # "set reference Qvalues at end action ids to just the immediate rewards"
            reference_Qvalues = T.set_subtensor(reference_Qvalues[end_ids],
                                                rewards[end_ids]
                                               )
        else:
            # "set reference Qvalues at end action ids to the immediate rewards + qvalues after end"
            reference_Qvalues = T.set_subtensor(reference_Qvalues[end_ids],
                                                rewards[end_ids] + gamma_or_gammas*optimal_qvalues_after_end[end_ids[0],0]
                                               )

    
    if return_reference:
        return reference_Qvalues
    else:
        #tensor of elementwise squared errors
        elwise_squared_error = squared_error(reference_Qvalues,action_Qvalues)
        return elwise_squared_error * is_alive
    