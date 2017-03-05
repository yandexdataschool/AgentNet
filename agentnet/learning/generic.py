"""
Several helper functions used in various reinforcement learning algorithms.
"""
from __future__ import division, print_function, absolute_import

import theano
import theano.tensor as T
from ..utils.logging import warn


def get_n_step_value_reference(state_values,rewards,
                               is_alive="always",
                               n_steps=None,
                               gamma_or_gammas=0.99,
                               crop_last=True,
                               state_values_after_end="zeros",
                               end_at_tmax=False,
                               force_n_step=False):
    """
    Computes the reference for state value function via n-step TD algorithm:
    
    Vref = r(t) + gamma*r(t+1) + gamma^2*r(t+2) + ... + gamma^n*V(s[t+n]) where n == n_steps
    
    Used by all n_step methods, including Q-learning, a2c and dpg
    
    Works with both Q-values and state values, depending on aggregation_function
    
    :param state_values: float[batch,tick] predicted state values V(s) at given batch session and time tick
            - for Q-learning, it's max over Q-values
            - for state-value based methods (a2c, dpg), it's same as state_values
        
    :param rewards: - float[batch,tick] rewards achieved by commiting actions at [batch,tick]
    
    :param is_alive: whether the session is still active at given tick, int[batch_size,time] of ones and zeros
        
    :param n_steps: if an integer is given, the references are computed in loops of n_steps
            Every n_steps'th step reference is set to  V = r + gamma * next V_predicted
            On other steps, reference is propagated V = r + gamma * next V reference
            Defaults to None: propagating rewards throughout the whole session.
            Widely known as "lambda" in RL community (TD-lambda, Q-lambda) plus or minus one :)
            If n_steps equals 1, this works exactly as regular TD (though a less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
        
    :param gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]

    :param crop_last: if True, ignores loss for last tick(default)
    :param state_values_after_end: - symbolic expression for "next state values" for last tick used for reference only.
                        Defaults at  T.zeros_like(values[:,0,None,:])
                        If you wish to simply ignore the last tick, 
                        use defaults and crop output's last tick ( qref[:,:-1] )
    :param end_at_tmax: if True, forces session end at last tick if there was no other session end.
    :param force_n_step: if True, does NOT fall back to 1-step algorithm if n_steps = 1

    :returns: V reference [batch,action_at_tick] according n-step algorithms ~ eligibility traces
            e.g. mentioned here http://arxiv.org/pdf/1602.01783.pdf as A3c and k-step Q-learning
            also described here https://arxiv.org/pdf/1506.02438v5.pdf for k-step advantage


    """
    if n_steps ==1 and not force_n_step:
        #fall back to a faster 1-step algorithm [without scan].
        return get_1_step_value_reference(
            state_values=state_values,
            rewards=rewards,
            is_alive=is_alive,
            gamma_or_gammas=gamma_or_gammas,
            crop_last=crop_last,
            state_values_after_end=state_values_after_end,
            end_at_tmax=end_at_tmax
        )

    # check dimensions
    if state_values.ndim != 2:
        if state_values.ndim == 3:
            warn("""state_values must have shape [batch,tick] (ndim = 2).
            Currently assuming state_values you provided to have shape [batch, tick,1].
            Working with state_values[:,:,0].
            If that isn't what you intended, fix state_values shape to [batch,tick]""")
            state_values = state_values[:, :, 0]
        else:
            raise ValueError("state_values must have shape [batch,tick] (ndim = 2),"
                             "while you have" + str(state_values.ndim))

    # fill default values
    if is_alive == "always":
        is_alive = T.ones_like(rewards)
    if state_values_after_end == "zeros":
        state_values_after_end = T.zeros_like(state_values[:, :1])

    #cast everything to floatX to avoid unwanted upcasts
    floatX = theano.config.floatX
    tensors = state_values,state_values_after_end,rewards,is_alive
    tensors = [tensor.astype(floatX) for tensor in tensors]
    state_values, state_values_after_end, rewards, is_alive = tensors

    #1 step forward
    next_state_values = T.concatenate([state_values[:, 1:],state_values_after_end], axis=1)

    #create an indicator that is
    # - 1 when Vref(s) should be r+gamma*V(s')     -- only at the end and every n_steps-th tick
    # - 0 when Vref(s) should be r + gamma*r' + gamma^2 * ... = r+gamma*Vref(s')

    tmax_indicator = T.zeros((rewards.shape[1],), dtype='uint8')
    tmax_indicator = T.set_subtensor(tmax_indicator[-1-int(crop_last)], 1) #at the end

    if n_steps is not None:
        #set tmax_indicator at every n_steps'th tick, starting from last (or pre-last if crop_last)
        time_ticks = T.arange(rewards.shape[1])[::-1] + int(crop_last)
        tmax_indicator = T.eq(time_ticks%n_steps,0) #every n_steps-th tick

    # initialize each reference with ZEROS after the end (won't be in output tensor)
    outputs_info = [T.zeros_like(rewards[:, 0]), ]
    non_seqs = (gamma_or_gammas,)

    # end indicator[batch,tick] that is 1 if this is the last state
    is_end = get_end_indicator(is_alive, force_end_at_t_max=end_at_tmax)


    sequences = [rewards.T,
                 is_end.T,
                 next_state_values.T,  # transpose to iterate over time, not over batch
                 tmax_indicator.T]

    # recurrent computation of reference state values (backwards through time)
    def backward_V_step(rewards,
                        is_end,
                        next_Vpred,
                        is_tmax,
                        next_Vref,
                        *args #you won't dare delete me
                       ):
        """scan inner computation step, going backwards in time
        params:
        rewards, is_alive, next_Vpred, time_i - sequences
        next_Vref - recurrent state value for next turn
        
        returns:
            current_Vref - recurrent state value at this turn
            
            current_Vref is computed thus:
                Once every n_steps or at session end:
                    current_Vref = r + gamma*next_Vpred   #computation through next predicted state value
                Otherwise:
                    current_Vref = r + gamma*next_Vref    #recurrent computation through next Qvalue
            
        """

        propagated_Vref = rewards + gamma_or_gammas * next_Vref  # propagates value from actual next action
        optimal_Vref = rewards + gamma_or_gammas * next_Vpred  # uses agent's prediction for next state

        # pick new_Vref if is_Tmax, else propagate existing one
        chosen_Vref = T.switch(is_tmax, optimal_Vref, propagated_Vref)

        # zero out references if session has ended already
        this_Vref = T.switch(is_end, rewards,chosen_Vref)

        return this_Vref

    reference_state_values = theano.scan(backward_V_step,
                                         sequences=sequences,
                                         non_sequences=non_seqs,
                                         outputs_info=outputs_info,
                                         go_backwards=True,
                                         strict=True,
                                         )[0]  # shape: [time_seq_inverted, batch]

    reference_state_values = reference_state_values.T[:, ::-1]  # [batch, time_seq]

    #zero-out last reward if asked
    if crop_last:
        reference_state_values = T.set_subtensor(reference_state_values[:,-1],state_values[:,-1])

    return reference_state_values


def get_1_step_value_reference(state_values,rewards,
                               is_alive="always",
                               gamma_or_gammas=0.99,
                               crop_last=True,
                               state_values_after_end="zeros",
                               end_at_tmax=False):
    """
    Computes the reference for state value function via 1-step TD algorithm:

    Vref = r(t) + gamma*V(s')

    Used as a fall-back by n-step algorithm when n_steps=1 (performance reasons)

    :param state_values: float[batch,tick] predicted state values V(s) at given batch session and time tick
            - for Q-learning, it's max over Q-values
            - for state-value based methods (a2c, dpg), it's same as state_values

    :param rewards: - float[batch,tick] rewards achieved by committing actions at [batch,tick]

    :param is_alive: whether the session is still active int/bool[batch_size,time]


    :param gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]

    :param crop_last: if True, ignores loss for last tick(default)
    :param state_values_after_end: - symbolic expression for "next state values" for last tick used for reference only.
                        Defaults at  T.zeros_like(values[:,0,None,:])
                        If you wish to simply ignore the last tick,
                        use defaults and crop output's last tick ( qref[:,:-1] )

    :param end_at_tmax: if True, forces session end at last tick if there was no other session end.


    :returns: V reference [batch,action_at_tick] = r + gamma*V(s')

    """

    # check dimensions
    if state_values.ndim != 2:
        if state_values.ndim == 3:
            warn("""state_values must have shape [batch,tick] (ndim = 2).
            Currently assuming state_values you provided to have shape [batch, tick,1].
            Working with state_values[:,:,0].
            If that isn't what you intended, fix state_values shape to [batch,tick]""")
            state_values = state_values[:, :, 0]
        else:
            raise ValueError("state_values must have shape [batch,tick] (ndim = 2),"
                             "while you have" + str(state_values.ndim))

    # fill default values
    if is_alive == "always":
        is_alive = T.ones_like(rewards)
    if state_values_after_end == "zeros":
        state_values_after_end = T.zeros_like(state_values[:, :1])

    # cast everything to floatX to avoid unwanted upcasts
    floatX = theano.config.floatX
    tensors = state_values, state_values_after_end, rewards, is_alive
    tensors = [tensor.astype(floatX) for tensor in tensors]
    state_values, state_values_after_end, rewards, is_alive = tensors

    # 1 step forward
    next_state_values = T.concatenate([state_values[:, 1:], state_values_after_end], axis=1)

    # set future state values at session end to 0
    end_ids = get_end_indicator(is_alive, force_end_at_t_max=end_at_tmax).nonzero()
    next_state_values = T.set_subtensor(next_state_values[end_ids],0)

    #reference
    reference_state_values = rewards + gamma_or_gammas * next_state_values

    #zero-out last reward if asked
    if crop_last:
        reference_state_values = T.set_subtensor(reference_state_values[:,-1],state_values[:,-1])

    return reference_state_values


# minor helpers

def get_values_for_actions(values_for_all_actions, actions):
    """
    Auxiliary function to select policy/Q-values corresponding to chosen actions.
    :param values_for_all_actions: qvalues or similar for all actions: floatX[batch,tick,action]
    :param actions: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    batch_i = T.arange(values_for_all_actions.shape[0])[:, None]
    time_i = T.arange(values_for_all_actions.shape[1])[None, :]
    action_values_predicted = values_for_all_actions[batch_i, time_i, actions]
    return action_values_predicted

def get_action_Qvalues(*args,**kwargs):
    "get_action_Qvalues has been renamed to get_values_for_actions in the same module. The alias will be removed in 0.11"
    raise NameError("get_action_Qvalues has been renamed to get_values_for_actions in the same module. The alias will be removed in 0.11")


def get_end_indicator(is_alive, force_end_at_t_max=False):
    """
    Auxiliary function to transform session alive indicator into end action indicator
    :param force_end_at_t_max: if True, all sessions that didn't end by the end of recorded sessions
    are ended at the last recorded tick.

    """

    # session-ending action indicator: uint8[batch,tick]
    # end = is_alive[now] and not is_alive[next tick]
    is_end = T.eq(is_alive[:, :-1] - is_alive[:, 1:], 1)

    if force_end_at_t_max:
        is_end_at_tmax = T.eq(T.sum(is_end, axis=1, keepdims=True),0)
    else:
        is_end_at_tmax = T.zeros((is_end.shape[0], 1), dtype=is_end.dtype)

    is_end = T.concatenate([is_end, is_end_at_tmax], axis=1)

    return is_end

