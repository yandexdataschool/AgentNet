"""
N-step Advantage Actor-Critic (A2c) implementation.
Works with action probabilities and state values instead of Q-values.

Works with discrete action space.

Follows the article http://arxiv.org/pdf/1602.01783v1.pdf 
"""
from __future__ import division, print_function, absolute_import

import theano
import theano.tensor as T
from lasagne.objectives import squared_error

from .helpers import get_n_step_value_reference, get_end_indicator, get_action_Qvalues
from ..utils.grad import consider_constant


def get_elementwise_objective(policy,
                              state_values,
                              actions,
                              rewards,
                              is_alive="always",
                              n_steps=None,
                              gamma_or_gammas=0.99,
                              crop_last=True,
                              force_values_after_end=True,
                              state_values_after_end="zeros",
                              consider_value_reference_constant=True,
                              consider_predicted_value_constant=True,
                              scan_dependencies=[],
                              scan_strict=True,
                              min_log_proba=-1e50):
    """
    returns cross-entropy-like objective function for Actor-Critic method

        L_policy = - log(policy) * (V_reference - const(V))
        L_V = (V - Vreference)^2
            
    parameters:
    
        policy [batch,tick,action_id] - predicted action probabilities
        state_values [batch,tick] - predicted state values
        actions [batch,tick] - committed actions
        rewards [batch,tick] - immediate rewards for taking actions at given time ticks
        
        is_alive [batch,tick] - whether given session is still active at given tick. Defaults to always active.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss
        
        n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
        
        gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
        
        crop_last - if True, zeros-out loss at final tick, if False - computes loss VS Qvalues_after_end
        
        force_values_after_end - if true, sets reference policy at session end to rewards[end] + qvalues_after_end
        
        state_values_after_end[batch,1,n_actions] - "next state values" for last tick used for reference only. 
                            Defaults at  T.zeros_like(state_values[:,0,None,:])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )

        
        
        scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        scan_strict: whether to evaluate values using strict theano scan or non-strict one
        
    Returns:
                
        elementwise sum of policy_loss + state_value_loss

    """

    # get reference values via Q-learning algorithm
    reference_state_values = get_n_step_value_reference(state_values, rewards, is_alive,
                                                        n_steps=n_steps,
                                                        optimal_state_values_after_end=state_values_after_end,
                                                        gamma_or_gammas=gamma_or_gammas,
                                                        dependencies=scan_dependencies,
                                                        strict=scan_strict
                                                        )

    # if we have to set after_end values
    if is_alive != "always" and force_values_after_end:
        # if asked to force reference_Q[end_tick+1,a] = 0, do it
        # note: if agent is always alive, this is meaningless

        # set future rewards at session end to rewards+qvalues_after_end
        end_ids = get_end_indicator(is_alive, force_end_at_t_max=True).nonzero()

        if state_values_after_end == "zeros":
            # "set reference state values at end action ids to just the immediate rewards"
            reference_state_values = T.set_subtensor(reference_state_values[end_ids], rewards[end_ids])
        else:

            # "set reference state values at end action ids to the immediate rewards + qvalues after end"
            new_state_values = rewards[end_ids] + gamma_or_gammas * state_values_after_end[end_ids[0], 0]
            reference_state_values = T.set_subtensor(reference_state_values[end_ids], new_state_values)

    # now compute the loss
    if is_alive == "always":
        is_alive = T.ones_like(actions, dtype=theano.config.floatX)

    # actor loss
    action_probas = get_action_Qvalues(policy, actions)

    reference_state_values = consider_constant(reference_state_values)
    if crop_last:
        reference_state_values = T.set_subtensor(reference_state_values[:,-1],
                                                 consider_constant(state_values[:,-1]))

    

    log_probas = T.maximum(T.log(action_probas), min_log_proba)

    policy_loss_elwise = - log_probas * (reference_state_values - consider_constant(state_values))

    # critic loss
    V_err_elwise = squared_error(reference_state_values, state_values)

    return (policy_loss_elwise + V_err_elwise) * is_alive
