"""
Advantage Actor-Critic (A2c or A3c) implementation.
Follows the article http://arxiv.org/pdf/1602.01783v1.pdf
Supports K-step advantage estimation as in https://arxiv.org/pdf/1506.02438v5.pdf

Agent should output action probabilities and state values instead of Q-values.
Works with discrete action space only.

"""
from __future__ import division, print_function, absolute_import

import theano
import theano.tensor as T
from lasagne.objectives import squared_error

from .generic import get_n_step_value_reference, get_action_Qvalues
from ..utils.grad import consider_constant
from warnings import warn

def get_elementwise_objective(policy,state_values,actions,rewards,
                              is_alive="always",
                              state_values_target=None,
                              n_steps=1,
                              n_steps_advantage='same',
                              gamma_or_gammas=0.99,
                              crop_last=True,
                              state_values_target_after_end="zeros",
                              consider_value_reference_constant=True,
                              force_end_at_last_tick=False,
                              return_separate=False,
                              loss_function=squared_error,
                              scan_dependencies=(),
                              scan_strict=True,
                              ):
    """
    returns cross-entropy-like objective function for Actor-Critic method

        L_policy = - log(policy) * (V_reference - const(V))
        L_V = (V - Vreference)^2


    :param policy: [batch,tick,action_id] - predicted action probabilities
    :param state_values: [batch,tick] - predicted state values
    :param actions: [batch,tick] - committed actions
    :param rewards: [batch,tick] - immediate rewards for taking actions at given time ticks
    :param is_alive: [batch,tick] - binary matrix whether given session is active at given tick. Defaults to all ones.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss
    :param state_values_target: there should be state values used to compute reference (e.g. older network snapshot)
                If None (defualt), uses current Qvalues to compute reference

    :param n_steps: if an integer is given, the STATE VALUE references are computed in loops of 3 states.
            If 1 (default), this works exactly as Q-learning (though less efficient one)
            If None: propagating rewards throughout the whole session.
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.

    :param n_steps_advantage: same as n_steps, but for advantage term (for actor loss). Defaults to same as n_steps

    :param gamma_or_gammas: a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts

    :param crop_last: if True, zeros-out loss at final tick, if False - computes loss VS Qvalues_after_end

    :param force_values_after_end: if true, sets reference policy at session end to rewards[end] + qvalues_after_end

    :param state_values_target_after_end: [batch,1,n_actions] - "next state values" for last tick used for reference only.
                            Defaults at  T.zeros_like(state_values[:,0,None,:])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )

    :param consider_value_reference_constant: whether or not to zero-out critic gradients through the reference state values term

    :param return_separate: if True, returns a tuple of (actor loss , critic loss ) instead of their sum.
    :param loss_function: loss_function(V_reference,V_predicted) used for CRITIC. Defaults to (V_reference-V_predicted)**2
                                Use to override squared error with different loss (e.g. Huber or MAE)


    :param scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)

    :param force_end_at_last_tick: if True, forces session end at last tick unless ended otherwise

    :param scan_strict: whether to evaluate values using strict theano scan or non-strict one


    :return: elementwise sum of policy_loss + state_value_loss [batch,tick]

    """

    if state_values_target is None:
        state_values_target = state_values
    if is_alive == "always":
        is_alive = T.ones_like(actions, dtype=theano.config.floatX)

    # check dimensions
    assert policy.ndim==3
    assert state_values.ndim in (2,3)
    assert state_values_target.ndim in (2,3)
    assert actions.ndim == rewards.ndim ==2
    if is_alive != 'always': assert is_alive.ndim==2

    #fix state_values dimensions
    #note: state_values_target is validated inside get_n_step_value_reference
    if state_values.ndim == 3:
        warn("""state_values must have shape [batch,tick] (ndim = 2).
            Currently assuming state_values you provided to have shape [batch, tick,1].""")
        state_values = state_values[:, :, 0]

    #####################
    #####Critic loss#####
    #####################

    reference_state_values = get_n_step_value_reference(
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

    if consider_value_reference_constant:
        reference_state_values = consider_constant(reference_state_values)

    #loss on each tick.  [squared error by default]
    critic_loss_elwise = loss_function(reference_state_values, state_values)*is_alive

    ####################
    #####Actor loss#####
    ####################

    #logprobas for actions taken
    action_logprobas = get_action_Qvalues(T.log(policy), actions)


    #if n_steps_advantage is different than n_steps, compute actor advantage separately. Otherwise reuse old
    if n_steps_advantage == 'same' or n_steps_advantage == n_steps:
        observed_state_values = reference_state_values
    else:
        observed_state_values = get_n_step_value_reference(
        state_values=state_values_target,
        rewards=rewards,
        is_alive=is_alive,
        n_steps=n_steps_advantage,
        gamma_or_gammas=gamma_or_gammas,
        state_values_after_end=state_values_target_after_end,
        end_at_tmax=force_end_at_last_tick,
        dependencies=scan_dependencies,
        strict=scan_strict,
        crop_last=crop_last,
    )

    advantage = consider_constant(observed_state_values - state_values)

    actor_loss_elwise = - action_logprobas * advantage * is_alive

    if return_separate:
        return actor_loss_elwise,critic_loss_elwise
    else:
        return actor_loss_elwise+critic_loss_elwise
