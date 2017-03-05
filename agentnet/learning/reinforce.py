"""
Basic REINFORCE algorithm.
Simple policy gradient method for discounted reward MDPs.
http://kvfrans.com/simple-algoritms-for-solving-cartpole/
"""

from __future__ import division, print_function, absolute_import

import theano
import theano.tensor as T

from .generic import get_n_step_value_reference, get_values_for_actions
from ..utils.grad import consider_constant

def get_elementwise_objective(policy,actions,rewards,
                              is_alive="always",
                              baseline="zeros",
                              gamma_or_gammas=0.99,
                              crop_last=True,
                              treat_policy_as_logpolicy=False,
                              ):
    """
    Compute and return policy gradient as evaluates

        L_policy = - log(policy) * (V_reference - baseline)
        L_V = (V - Vreference)^2


    :param policy: [batch,tick,action_id] - predicted action probabilities
        either for all actions, shape [batch,tick,action]
        or for chosen actions, shape [batch,tick]

    :param actions: [batch,tick] - committed actions
    :param rewards: [batch,tick] - immediate rewards for taking actions at given time ticks
    :param is_alive: [batch,tick] - binary matrix whether given session is active at given tick. Defaults to all ones.
    :param baseline: [batch,tick] - REINFORCE  baselines tensor for each batch/tick. Uses no baseline by default.
    :param gamma_or_gammas: a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts
    :param crop_last: if True, zeros-out loss at final tick
    :param treat_policy_as_logpolicy: if True, policy is used as log(pi(a|s)). You may want to do this for numerical stability reasons.
    :return: elementwise sum of policy_loss + state_value_loss [batch,tick]

    """

    if is_alive == "always":
        is_alive = T.ones_like(actions, dtype=theano.config.floatX)
    if baseline == "zeros":
        baseline = T.zeros_like(rewards,dtype=theano.config.floatX)
    # check dimensions
    assert policy.ndim in (2,3),"policy must have shape either [batch,tick,action], for all actions," \
                                " or [batch,tick], for chosen actions"

    assert actions.ndim == rewards.ndim == is_alive.ndim==2, "actions, rewards and is_alive must have shape [batch,time]"

    #logprobas for all actions
    logpolicy = T.log(policy) if not treat_policy_as_logpolicy else policy

    #logprobas for actions taken
    given_action_probas = (logpolicy.ndim==2)
    action_logprobas = logpolicy if given_action_probas else get_values_for_actions(logpolicy, actions)

    #estimate n-step advantage. Note that we use current state values here (and not e.g. state_values_target)
    observed_state_values = get_n_step_value_reference(
        state_values=T.zeros_like(rewards,dtype=theano.config.floatX),
        rewards=rewards,
        is_alive=is_alive,
        n_steps=None,
        gamma_or_gammas=gamma_or_gammas,
        end_at_tmax=True,
        crop_last=crop_last,
    )

    advantage = consider_constant(observed_state_values - baseline)

    loss_elwise = - action_logprobas * advantage * is_alive

    return loss_elwise
