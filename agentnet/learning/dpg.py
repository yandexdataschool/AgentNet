"""
Deterministic policy gradient loss, Also used for model-based acceleration algorithms.
Supports regular and k-step implementation.
Based on:
- http://arxiv.org/abs/1509.02971
- http://arxiv.org/abs/1603.00748
- http://jmlr.org/proceedings/papers/v32/silver14.pdf
"""
from __future__ import division, print_function, absolute_import
import theano.tensor as T
from .generic import get_n_step_value_reference
from ..utils.grad import consider_constant
from lasagne.objectives import squared_error



def get_elementwise_objective_critic(action_qvalues,
                                     state_values,
                                     rewards,
                                     is_alive="always",
                                     n_steps=1,
                                     gamma_or_gammas=0.99,
                                     crop_last=True,
                                     state_values_after_end="zeros",
                                     force_end_at_last_tick=False,
                                     consider_reference_constant=True,
                                     return_reference=False,
                                     loss_function = squared_error,
                                     scan_dependencies=(),
                                     scan_strict=True):
    """
    Returns squared error between action values and reference (r+gamma*V(s')) according to deterministic policy gradient.

    This function can also be used for any model-based acceleration like Qlearning with normalized advantage functions.
        - Original article: http://arxiv.org/abs/1603.00748
        - Since you can provide any state_values, you can technically use any other advantage function shape
            as long as you can compute V(s).

    If n_steps > 1, the algorithm will use n-step Temporal Difference updates
        V_reference(state,action) = reward(state,action) + gamma*reward(state_1,action_1) + ... + gamma^n * V(state_n)

    :param action_qvalues: [batch,tick,action_id] - predicted qvalues
    :param state_values: [batch,tick] - predicted state values (aka qvalues for best actions)
    :param rewards: [batch,tick] - immediate rewards for taking actions at given time ticks
    :param is_alive: [batch,tick] - whether given session is still active at given tick. Defaults to always active.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss

    :param n_steps: if an integer is given, uses n-step TD algorithm
            If 1 (default), this works exactly as normal TD
            If None: propagating rewards throughout the whole sequence of state-action pairs.

    :param gamma_or_gammas: delayed reward discounts: a single value or array[batch,tick](can broadcast dimensions).
    :param crop_last: if True, zeros-out loss at final tick, if False - computes loss VS Qvalues_after_end
    :param state_values_after_end: [batch,1] - symbolic expression for "best next state q-values" for last tick
                            used when computing reference Q-values only.
                            Defaults at  T.zeros_like(Q-values[:,0,None,0])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
    :param force_end_at_last_tick: if True, forces session end at last tick unless ended otehrwise

    :param consider_reference_constant: whether or not zero-out gradient flow through reference_qvalues
            (True is highly recommended)

    :param return_reference: if True, returns reference Qvalues.
            If False, returns loss_function(action_Qvalues, reference_qvalues)

    :param loss_function: loss_function(V_reference,V_predicted). Defaults to (V_reference-V_predicted)**2.
                            Use to override squared error with different loss (e.g. Huber or MAE)

    :return: mean squared error over Q-values (using formula above for loss)

    """

    assert action_qvalues.ndim  == state_values.ndim == rewards.ndim ==2
    if is_alive == 'always':
        is_alive = T.ones_like(rewards)
    assert is_alive.ndim==2


    # get reference Q-values via Q-learning algorithm
    reference_qvalues= get_n_step_value_reference(
        state_values=state_values,
        rewards=rewards,
        is_alive=is_alive,
        n_steps=n_steps,
        gamma_or_gammas=gamma_or_gammas,
        state_values_after_end=state_values_after_end,
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
        elwise_squared_error = loss_function(reference_qvalues, action_qvalues) * is_alive
        return elwise_squared_error
