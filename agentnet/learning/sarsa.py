"""
State-Action-Reward-State-Action (sars'a') learning algorithm implementation
"""
from __future__ import division, print_function, absolute_import

import theano.tensor as T
import theano
import numpy as np

from lasagne.objectives import squared_error

from .helpers import get_end_indicator, get_action_Qvalues
from ..utils.grad import consider_constant
from ..utils import create_shared

default_gamma = create_shared('sarsa_gamma_default', np.float32(0.99), theano.config.floatX)


def get_reference_Qvalues(Qvalues,
                          actions,
                          rewards,
                          gamma_or_gammas=default_gamma,
                          qvalues_after_end="zeros"
                          ):
    """
    Returns reference Qvalues according to State-Action-Reward-State-Action (SARSA) algorithm
    
    parameters:
    Qvalues [batch,tick,action_id] - predicted Q-values
    actions [batch,tick] - committed actions
    gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
    qvalues_after_end - symbolic expression for "future rewards" term for last tick used for reference only. 
                        Defaults at  T.zeros_like(rewards[:,0,None])
                        If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
                        
        
    Returns:
    Qreference - reference Q-values at [batch,tick] using formula
    
        Q reference [batch,action_at_tick] = rewards[t] + gamma_or_gammas * Qs(t+1,action[t+1])
        Where action[t+1] is simply action that agent took at next time tick [padded with qvalues_after_end]
    
    
    """
    if qvalues_after_end == "zeros":
        qvalues_after_end = T.zeros_like(rewards[:, 0, None])

    # Q-values for "next" states (missing last tick): float[batch,tick-1,action]
    next_Qvalues_predicted = Qvalues[:, 1:]

    # actions committed at next ticks (missing last tick): int[batch,tick-1]
    next_actions = actions[:, 1:]

    future_rewards_estimate = get_action_Qvalues(next_Qvalues_predicted, next_actions)

    # adding the last tick
    future_rewards_estimate = T.concatenate(
        [
            future_rewards_estimate,
            qvalues_after_end,
        ],
        axis=1
    )

    # full Q-value formula (SARSA algorithm)
    reference_Qvalues = rewards + gamma_or_gammas * future_rewards_estimate

    return reference_Qvalues


def get_elementwise_objective(Qvalues,
                              actions,
                              rewards,
                              is_alive="always",
                              gamma_or_gammas=0.95,
                              crop_last=True,
                              force_qvalues_after_end=True,
                              qvalues_after_end="zeros",
                              consider_reference_constant=True, ):
    """
    Returns squared error between predicted and reference Qvalues according to Q-learning algorithm
    
        Qreference(state,action) = reward(state,action) + gamma* Q(next_state,next_action)  
        loss = mean over (Qvalues - Qreference)**2
        
    parameters:
    
        Qvalues [batch,tick,action_id] - predicted qvalues
        actions [batch,tick] - commited actions
        rewards [batch,tick] - immediate rewards for taking actions at given time ticks
        
        is_alive [batch,tick] - whether given session is still active at given tick. Defaults to always active.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss
        
        gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
        
        crop_last - if True, zeros-out loss at final tick, if False - computes loss VS Qvalues_after_end
        
        force_qvalues_after_end - if true, sets reference Qvalues at session end to rewards[end] + qvalues_after_end
        
        qvalues_after_end [batch,1,n_actions] - symbolic expression for "next state q-values" for last tick used for reference only. 
                            Defaults at  T.zeros_like(Qvalues[:,0,None,:])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )

        consider_reference_constant - whether or not zero-out gradient flow through reference_Qvalues
            (True is highly recommended)
    Returns:
                
        tensor [batch, tick] of squared errors over Qvalues (using formula above for loss)

    """
    # get reference Qvalues via Q-learning algorithm
    reference_Qvalues = get_reference_Qvalues(Qvalues, actions, rewards,
                                              gamma_or_gammas=gamma_or_gammas,
                                              qvalues_after_end=qvalues_after_end,
                                              )

    if consider_reference_constant:
        # do not pass gradient through reference Q-values (since they DO depend on Q-values by default)
        reference_Qvalues = consider_constant(reference_Qvalues)

    # get predicted qvalues for committed actions (to compare with reference Q-values)
    action_Qvalues = get_action_Qvalues(Qvalues, actions)

    # if agent is always alive, return the simplified loss
    if is_alive == "always":

        # tensor of element-wise squared errors
        elwise_squared_error = squared_error(reference_Qvalues, action_Qvalues)

    else:
        # we are given an is_alive matrix : uint8[batch,tick]

        # if asked to force reference_Q[end_tick+1,a] = 0, do it
        # note: if agent is always alive, this is meaningless

        if force_qvalues_after_end:
            # set future rewards at session end to rewards + qvalues_after_end
            end_ids = get_end_indicator(is_alive, force_end_at_t_max=True).nonzero()

            if qvalues_after_end == "zeros":
                # "set reference Q-values at end action ids to just the immediate rewards"
                reference_Qvalues = T.set_subtensor(reference_Qvalues[end_ids], rewards[end_ids])
            else:
                last_optimal_rewards = T.zeros_like(rewards[:, 0])

                # "set reference Q-values at end action ids to the immediate rewards + qvalues after end"
                reference_Qvalues = T.set_subtensor(reference_Qvalues[end_ids],
                                                    rewards[end_ids] + gamma_or_gammas * last_optimal_rewards[
                                                        end_ids[0], 0]
                                                    )

        # tensor of element-wise squared errors
        elwise_squared_error = squared_error(reference_Qvalues, action_Qvalues)

        # zero-out loss after session ended
        elwise_squared_error = elwise_squared_error * is_alive
     
    if crop_last:
        elwise_squared_error = T.set_subtensor(elwise_squared_error[:,-1],0)


    return elwise_squared_error
