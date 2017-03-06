"""
Q-learning implementation.
Works with discrete action space.
Supports n-step updates and custom state value function (max(Q(s,a)), double q-learning, boltzmann, mellowmax, expected value sarsa,...)
"""

from __future__ import division, print_function, absolute_import

import theano.tensor as T
from lasagne.objectives import squared_error

from .generic import get_n_step_value_reference, get_values_for_actions
from ..utils.grad import consider_constant

__author__ = "Konstantin Sidorov(partobs-mdp), justheuristic"

def get_elementwise_objective(qvalues, actions, rewards,
                              is_alive="always",
                              qvalues_target=None,
                              state_values_target=None,
                              n_steps=1,
                              gamma_or_gammas=0.99,
                              crop_last=True,
                              state_values_target_after_end="zeros",
                              consider_reference_constant=True,
                              aggregation_function="deprecated",
                              force_end_at_last_tick=False,
                              return_reference=False,
                              loss_function=squared_error):
    """
    Returns squared error between predicted and reference Q-values according to n-step Q-learning algorithm

        Qreference(state,action) = reward(state,action) + gamma*reward(state_1,action_1) + ... + gamma^n * max[action_n]( Q(state_n,action_n)
        loss = mean over (Qvalues - Qreference)**2

    :param qvalues: [batch,tick,actions] - predicted qvalues
    :param actions: [batch,tick] - commited actions
    :param rewards: [batch,tick] - immediate rewards for taking actions at given time ticks
    :param is_alive: [batch,tick] - whether given session is still active at given tick. Defaults to always active.

    :param qvalues_target: Q-values used when computing reference (e.g. r+gamma*Q(s',a_max). shape [batch,tick,actions]
        examples:
        (default) If None, uses current Qvalues.
        Older snapshot Qvalues (e.g. from a target network)

    :param state_values_target: state values V(s), used when computing reference (e.g. r+gamma*V(s'), shape [batch_size,seq_length,1]
        double q-learning V(s) = Q_old(s,argmax Q_new(s,a))
        expected_value_sarsa V(s) = E_a~pi(a|s) Q(s,a)
        state values from teacher network (knowledge transfer)

    Must provide either nothing or qvalues_target or state_values_target, not both at once

    :param n_steps: if an integer is given, uses n-step q-learning algorithm
            If 1 (default), this works exactly as normal q-learning
            If None: propagating rewards throughout the whole sequence of state-action pairs.

    :param gamma_or_gammas: delayed reward discounts: a single value or array[batch,tick](can broadcast dimensions).

    :param crop_last: if True, zeros-out loss at final tick, if False - computes loss VS Qvalues_after_end

    :param state_values_target_after_end: [batch,1] - symbolic expression for "next best q-values" for last tick
                            used when computing reference Q-values only.
                            Defaults at  T.zeros_like(Q-values[:,0,None,0]). if crop_last=True, simply does not penalize at last tick.
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
    :param consider_reference_constant: whether or not zero-out gradient flow through reference_qvalues
            (True is highly recommended)

    :param force_end_at_last_tick: if True, forces session end at last tick unless ended otehrwise

    :param return_reference: if True, returns reference Qvalues.
            If False, returns squared_error(action_qvalues, reference_qvalues)
    :param loss_function: loss_function(V_reference,V_predicted). Defaults to (V_reference-V_predicted)**2.
                            Use to override squared error with different loss (e.g. Huber or MAE)

    :return: mean squared error over Q-values (using formula above for loss)

    """
    if aggregation_function != "deprecated":
        raise NotImplementedError("aggregation function has beed deprecated and removed. You can now manually compute "
                                  "any V(s) and pass it as state_state_values_target. By default it's qvalues.max(axis=-1)")
    #set defaults and assert shapes
    if is_alive == 'always':
        is_alive = T.ones_like(rewards)
    assert qvalues_target is None or state_values_target is None, "Please provide only one of (qvalues_target," \
                                                                  "state_values_target) or none of them, not both"
    assert actions.ndim == rewards.ndim == is_alive.ndim == 2, "actions, rewards and is_alive must have shape [batch,time]"

    assert qvalues.ndim == 3, "q-values must have shape [batch,time,n_actions]"
    assert qvalues_target is None or qvalues_target.ndim == 3,"qvalues_target must have shape[batch,time,n_actions]]"
    assert state_values_target is None or state_values_target.ndim ==2, "state values must have shape [batch,time]"


    #unless already given V(s), compute V(s) as Qvalues of best actions
    state_values_target = state_values_target or T.max(qvalues_target or qvalues, axis=-1)

    # get predicted Q-values for committed actions by both current and target networks
    action_qvalues = get_values_for_actions(qvalues, actions)

    # get reference Q-values via Q-learning algorithm
    reference_qvalues = get_n_step_value_reference(
        state_values=state_values_target,
        rewards=rewards,
        is_alive=is_alive,
        n_steps=n_steps,
        gamma_or_gammas=gamma_or_gammas,
        state_values_after_end=state_values_target_after_end,
        end_at_tmax=force_end_at_last_tick,
        crop_last=crop_last,
    )

    if consider_reference_constant:
        # do not pass gradient through reference Qvalues (since they DO depend on Qvalues by default)
        reference_qvalues = consider_constant(reference_qvalues)

    #If asked, make sure loss equals 0 for the last time-tick.
    if crop_last:
        reference_qvalues = T.set_subtensor(reference_qvalues[:,-1],action_qvalues[:,-1])

    if return_reference:
        return reference_qvalues
    else:
        # tensor of elementwise squared errors
        elwise_squared_error = loss_function(reference_qvalues, action_qvalues)
        return elwise_squared_error * is_alive
