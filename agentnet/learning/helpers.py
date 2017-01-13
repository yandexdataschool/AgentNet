"""
Several helper functions used in various reinforcement learning algorithms.
"""
from __future__ import division, print_function, absolute_import

from warnings import warn

import theano
import theano.tensor as T


def get_n_step_value_reference(state_values,
                               rewards,
                               is_alive="always",
                               n_steps=None,
                               gamma_or_gammas=0.99,
                               optimal_state_values="same_as_state_values",
                               optimal_state_values_after_end="zeros",
                               dependencies=tuple(),
                               crop_last=True,
                               strict=True):
    """
    Computes the reference for state value function via n-step algorithm:
    
    Vref = r(t) + gamma*r(t+1) + gamma^2*r(t+2) + ... + gamma^n*t(t+n) where n == n_steps
    
    Used by all n_step methods, including Q-learning, a2c and dpg
    
    Works with both Q-values and state values, depending on aggregation_function
    
    :param state_values: float[batch,tick] predicted state values V(s) at given batch session and time tick
        
    :param rewards: - float[batch,tick] rewards achieved by commiting actions at [batch,tick]
    
    :param is_alive: whether the session is still active int/bool[batch_size,time]
        
        
    :param n_steps: if an integer is given, the references are computed in loops of n_steps
            Every n_steps'th step reference is set to  V = r + gamma * next V_predicted_
            On other steps, reference is propagated V = r + gamma * next V reference
            Defaults to None: propagating rewards throughout the whole session.
            Widely known as "lambda" in RL community (TD-lambda, Q-lambda) plus or minus one :)
            If n_steps equals 1, this works exactly as regular TD (though a less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
        
    :param gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]

    :param optimal_state_values: state values given optimal actions.
            - for Q-learning, it's max over Q-values
            - for state-value based methods (a2c, dpg), it's same as state_values (defaults to that)
        
        
    :param optimal_state_values_after_end: - symbolic expression for "next state values" for last tick used for reference only.
                        Defaults at  T.zeros_like(values[:,0,None,:])
                        If you wish to simply ignore the last tick, 
                        use defaults and crop output's last tick ( qref[:,:-1] )
                        
    :param dependencies: everything else you need to evaluate first 3 parameters (only if strict==True)
        
    :param strict: whether to evaluate values using strict theano scan or non-strict one

    :returns: V reference [batch,action_at_tick] according n-step algorithms ~ eligibility traces
            e.g. mentioned here http://arxiv.org/pdf/1602.01783.pdf as A3c and k-step Q-learning

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

    # handle aggregation function
    if optimal_state_values == "same_as_state_values":
        optimal_state_values = state_values

    # fill default values
    if is_alive == "always":
        is_alive = T.ones_like(rewards)

    #cast everything to floatX
    floatX = theano.config.floatX
    tensors = state_values,rewards,is_alive,optimal_state_values,optimal_state_values_after_end
    tensors = [tensor.astype(floatX) for tensor in tensors]
    state_values, rewards, is_alive, optimal_state_values, optimal_state_values_after_end = tensors



    if crop_last:
        #TODO rewrite by precomputing correct td-0 qvalues here to clarify notation
        #alter tensors so that last reference = last prediction
        is_alive = T.set_subtensor(is_alive[:,-1],1)
        rewards = T.set_subtensor(rewards[:,-1],0)
        next_state_values = T.concatenate([optimal_state_values[:, 1:] * is_alive[:, 1:],
                                           state_values[:,-1:]/gamma_or_gammas], axis=1)

    else:
        #crop_last == False
        if optimal_state_values_after_end == "zeros":
            optimal_state_values_after_end = T.zeros_like(optimal_state_values[:, :1])
        # get "Next state_values": floatX[batch,time] at each tick
        # do so by shifting state_values backwards in time, pad with state_values_after_end
        next_state_values = T.concatenate(
            [optimal_state_values[:, 1:] * is_alive[:, 1:], optimal_state_values_after_end], axis=1)

    # initialize each reference with ZEROS after the end (won't be in output tensor)
    outputs_info = [T.zeros_like(rewards[:, 0]), ]

    non_seqs = (gamma_or_gammas,) + tuple(dependencies)

    if n_steps is None:
        tmax_indicator = T.zeros((rewards.shape[1],),dtype='uint8')
        tmax_indicator = T.set_subtensor(tmax_indicator[-1],1)
    else:
        time_ticks = T.arange(rewards.shape[1])
        tmax_indicator = T.eq(time_ticks%n_steps,0)
        tmax_indicator = T.set_subtensor(tmax_indicator[-1], 1).astype('uint8')

    sequences = [rewards.T,
                 is_alive.T,
                 next_state_values.T,  # transpose to iterate over time, not over batch
                 tmax_indicator.T]

    # recurrent computation of reference state values (backwards through time)
    def backward_V_step(rewards,
                        is_alive,
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
        this_Vref = T.switch(is_alive, chosen_Vref, 0)

        return this_Vref

    reference_state_values = theano.scan(backward_V_step,
                                         sequences=sequences,
                                         non_sequences=non_seqs,
                                         outputs_info=outputs_info,
                                         go_backwards=True,
                                         strict=strict
                                         )[0]  # shape: [time_seq_inverted, batch]

    reference_state_values = reference_state_values.T[:, ::-1]  # [batch, time_seq]

    return reference_state_values


# minor helpers

def get_action_Qvalues(Qvalues, actions):
    """
    Auxiliary function to select Q-values corresponding to actions taken.
    Returns Q-values predicted that resulted in actions: float[batch,tick]
    """
    batch_i = T.arange(Qvalues.shape[0])[:, None]
    time_i = T.arange(Qvalues.shape[1])[None, :]
    action_Qvalues_predicted = Qvalues[batch_i, time_i, actions]
    return action_Qvalues_predicted

def get_end_indicator(is_alive, force_end_at_t_max=False):
    """
    Auxiliary function to transform session alive indicator into end action indicator
    If force_end_at_t_max is True, all sessions that didn't end by the end of recorded sessions
    are ended at the last recorded tick."""

    # session-ending action indicator: uint8[batch,tick]
    is_end = T.eq(is_alive[:, :-1] - is_alive[:, 1:], 1)

    if force_end_at_t_max:
        session_ended_before = T.neq(T.sum(is_end, axis=1, keepdims=True),0)
        is_end_at_tmax = 1 - T.gt(session_ended_before, 0)
    else:
        is_end_at_tmax = T.zeros((is_end.shape[0], 1), dtype=is_end.dtype)

    is_end = T.concatenate([is_end, is_end_at_tmax], axis=1)

    return is_end


def ravel_alive(is_alive, *args):
    """
    Takes all is_alive ticks from all sessions and merges them into 1 dimension
    """
    alive_selector = is_alive.nonzero()
    return [arg[alive_selector] for arg in args]
