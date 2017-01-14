"""
Q-learning implementation.
Works with discrete action space.
Supports eligibility traces and custom state value function (max(Q(s,a)), boltzmann, mellowmax, expected value sarsa)
"""
from __future__ import division, print_function, absolute_import

import theano.tensor as T
from lasagne.objectives import squared_error

from .generic import get_n_step_value_reference, get_action_Qvalues
from ..utils.grad import consider_constant


def get_elementwise_objective(qvalues, actions, rewards,
                              is_alive="always",
                              qvalues_target=None,
                              n_steps=1,
                              gamma_or_gammas=0.99,
                              crop_last=True,
                              state_values_target_after_end="zeros",
                              consider_reference_constant=True,
                              aggregation_function=lambda qv: T.max(qv, axis=-1),
                              force_end_at_last_tick=False,
                              return_reference=False,
                              loss_function=squared_error,
                              scan_dependencies=(),
                              scan_strict=True):
    """
    Returns squared error between predicted and reference Q-values according to n-step Q-learning algorithm

        Qreference(state,action) = reward(state,action) + gamma*reward(state_1,action_1) + ... + gamma^n * max[action_n]( Q(state_n,action_n)
        loss = mean over (Qvalues - Qreference)**2

    :param qvalues: [batch,tick,action_id] - predicted qvalues
    :param actions: [batch,tick] - commited actions
    :param rewards: [batch,tick] - immediate rewards for taking actions at given time ticks
    :param is_alive: [batch,tick] - whether given session is still active at given tick. Defaults to always active.

    :param qvalues_target: Older snapshot Qvalues (e.g. from a target network). If None, uses current Qvalues

    :param n_steps: if an integer is given, the references are computed in loops of 3 states.
            If 1 (default), this works exactly as Q-learning (though less efficient one)
            If None: propagating rewards throughout the whole session.
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.

    :param gamma_or_gammas: delayed reward discounts: a single value or array[batch,tick](can broadcast dimensions).

    :param crop_last: if True, zeros-out loss at final tick, if False - computes loss VS Qvalues_after_end

    :param qvalues_after_end: [batch,1] - symbolic expression for "best next state q-values" for last tick
                            used when computing reference Q-values only.
                            Defaults at  T.zeros_like(Q-values[:,0,None,0])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
    :param consider_reference_constant: whether or not zero-out gradient flow through reference_qvalues
            (True is highly recommended)
    :param aggregation_function: a function that takes all Qvalues for "next state Q-values" term and returns what
                                is the "best next Q-value". Normally you should not touch it. Defaults to max over actions.
                                Normally you shouldn't touch this
                                Takes input of [batch,n_actions] Q-values

    :param force_end_at_last_tick: if True, forces session end at last tick unless ended otehrwise

    :param return_reference: if True, returns reference Qvalues.
            If False, returns squared_error(action_qvalues, reference_qvalues)
    :param loss_function: loss_function(V_reference,V_predicted). Defaults to (V_reference-V_predicted)**2.
                            Use to override squared error with different loss (e.g. Huber or MAE)

    :param scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
    :param scan_strict: whether to evaluate Qvalues using strict theano scan or non-strict one
    :return: mean squared error over Q-values (using formula above for loss)

    """
    #set defaults
    if qvalues_target is None:
        qvalues_target = qvalues
    if is_alive != 'always':
        is_alive = T.ones_like(rewards)
    assert qvalues.ndim == qvalues_target.ndim == 3
    assert actions.ndim == rewards.ndim ==2
    assert is_alive.ndim == 2


    # get Qvalues of best actions (used every K steps for reference Q-value computation
    state_values_target = aggregation_function(qvalues_target)

    # get predicted Q-values for committed actions by both current and target networks
    # (to compare with reference Q-values and use for recurrent reference computation)
    action_qvalues = get_action_Qvalues(qvalues, actions)

    # get reference Q-values via Q-learning algorithm
    reference_qvalues = get_n_step_value_reference(
        state_values=state_values_target,
        rewards=rewards,
        is_alive=is_alive,
        n_steps=n_steps,
        gamma_or_gammas=gamma_or_gammas,
        state_values_after_end=state_values_target_after_end,
        end_at_tmax=force_end_at_last_tick,
        dependencies=scan_dependencies,
        strict=scan_strict,
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
