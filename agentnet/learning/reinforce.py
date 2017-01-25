"""
Basic REINFORCE algorithm.
Simple policy gradient method for discounted reward MDPs.
http://kvfrans.com/simple-algoritms-for-solving-cartpole/
"""

from __future__ import division, print_function, absolute_import

import theano
import theano.tensor as T
from lasagne.objectives import squared_error

from .generic import get_n_step_value_reference, get_values_for_actions
from ..utils.grad import consider_constant
from warnings import warn

def get_elementwise_objective(policy,actions,rewards,
                              is_alive="always",
                              baseline="zeros",
                              gamma_or_gammas=0.99,
                              crop_last=True,
                              treat_policy_as_logpolicy=False,
                              scan_dependencies=(),
                              scan_strict=True,
                              ):
    """
    Compute and return policy gradient as evaluates

        L_policy = - log(policy) * (V_reference - baseline)
        L_V = (V - Vreference)^2


    :param policy: [batch,tick,action_id] - predicted action probabilities
    :param actions: [batch,tick] - committed actions
    :param rewards: [batch,tick] - immediate rewards for taking actions at given time ticks
    :param is_alive: [batch,tick] - binary matrix whether given session is active at given tick. Defaults to all ones.
    :param baseline: [batch,tick] - REINFORCE  baselines tensor for each batch/tick. Uses no baseline by default.
    :param gamma_or_gammas: a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts
    :param crop_last: if True, zeros-out loss at final tick
    :param treat_policy_as_logpolicy: if True, policy is used as log(pi(a|s)). You may want to do this for numerical stability reasons.
    :param scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
    :param scan_strict: whether to evaluate values using strict theano scan or non-strict one
    :return: elementwise sum of policy_loss + state_value_loss [batch,tick]

    """

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
    if state_values_target.ndim == 3:
        warn("""state_values_target must have shape [batch,tick] (ndim = 2).
            Currently assuming state_values_target you provided to have shape [batch, tick,1].""")
        state_values_target = state_values_target[:, :, 0]

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

    #logprobas for all actions
    logpolicy = T.log(policy) if not treat_policy_as_logpolicy else policy
    #logprobas for actions taken
    action_logprobas = get_values_for_actions(logpolicy, actions)


    #if n_steps_advantage is different than n_steps, compute actor advantage separately. Otherwise reuse old
    if n_steps_advantage == 'same':
        n_steps_advantage = n_steps

    #estimate n-step advantage. Note that we use current state values here (and not e.g. state_values_target)
    observed_state_values = get_n_step_value_reference(
        state_values=state_values,
        rewards=rewards,
        is_alive=is_alive,
        n_steps=n_steps_advantage,
        gamma_or_gammas=gamma_or_gammas,
        state_values_after_end=state_values_after_end,
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
