"""
N-step deterministic policy gradient
"""
from __future__ import division, print_function, absolute_import

import theano
import theano.tensor as T
from lasagne.objectives import squared_error

from .helpers import get_n_step_value_reference, get_end_indicator
from ..utils.grad import consider_constant


def get_elementwise_objective_components(policy,
                                         rewards,
                                         policy_values,
                                         action_values='same',
                                         is_alive="always",
                                         n_steps=None,
                                         gamma_or_gammas=0.99,
                                         crop_last = True,
                                         force_values_after_end=True,
                                         state_values_after_end="zeros",
                                         consider_value_reference_constant=True,
                                         consider_predicted_value_constant=True,
                                         scan_dependencies=tuple(),
                                         scan_strict=True,
                                         ):
    """

    N-step Deterministic Policy Gradient (A2c) implementation.

    Works with continuous action space (real value or vector of such)

    Requires action policy(mu) and state values.

    Based on
    http://arxiv.org/abs/1509.02971
    http://jmlr.org/proceedings/papers/v32/silver14.pdf

    This particular implementation also allows N-step reinforcement learning

    The code mostly relies on the same architecture as advantage actor-critic a2c_n_step


    returns deterministic policy gradient components for actor and critic

        L_policy = -critic(state,policy) = -action_values 
        L_V = (V - Vreference)^2
        
        You will have to independently compute updates for actor and critic and then add them up.
            
    parameters:
    
        policy [batch,tick,action_id] - predicted "optimal policy" (mu)
        rewards [batch,tick] - immediate rewards for taking actions at given time ticks
        policy_values [batch,tick] - predicted state values given OPTIMAL policy
        action_values [batch,tick] - predicted Q_values for commited actions INCLUDING EXPLORATION if any
                            Default value implies action_values = state_values if we have no exploration
        
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
                
        Element-wise sum of policy_loss + state_value_loss

    """

    if action_values == 'same':
        action_values = policy_values


    # get reference values via DPG algorithm
    reference_action_values = get_n_step_value_reference(action_values,
                                                         rewards,
                                                         is_alive,
                                                         n_steps=n_steps,
                                                         optimal_state_values_after_end=state_values_after_end,
                                                         gamma_or_gammas=gamma_or_gammas,
                                                         dependencies=scan_dependencies,
                                                         strict=scan_strict
                                                         )

    if is_alive != "always" and force_values_after_end:
        # if asked to force reference_Q[end_tick+1,a] = 0, do it
        # note: if agent is always alive, this is meaningless

        # set future rewards at session end to rewards+qvalues_after_end
        end_ids = get_end_indicator(is_alive, force_end_at_t_max=True).nonzero()

        if state_values_after_end == "zeros":
            # "set reference state values at end action ids to just the immediate rewards"
            reference_action_values = T.set_subtensor(reference_action_values[end_ids], rewards[end_ids])
        else:
            # "set reference state values at end action ids to the immediate rewards + qvalues after end"
            new_subtensor_values = rewards[end_ids] + gamma_or_gammas * state_values_after_end[end_ids[0], 0]
            reference_action_values = T.set_subtensor(reference_action_values[end_ids], new_subtensor_values)

    # now compute the loss components
    if is_alive == "always":
        is_alive = T.ones_like(action_values, dtype=theano.config.floatX)

    # actor loss
    # here we rely on fact that state_values = critic(state,optimal_policy)
    # using chain rule,
    # grad(state_values,actor_weights) = grad(state_values, optimal_policy)*grad(optimal_policy,actor_weights)
    policy_loss_elwise = -policy_values

    # critic loss
    reference_action_values = consider_constant(reference_action_values)
    v_err_elementwise = squared_error(reference_action_values, action_values)
    
    if crop_last:
        v_err_elementwise = T.set_subtensor(v_err_elementwise[:,-1],0)


    return policy_loss_elwise * is_alive, v_err_elementwise * is_alive
