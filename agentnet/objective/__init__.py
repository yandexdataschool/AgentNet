"""
Base agent objective class that defines when does agent get reward
"""
from __future__ import division, print_function, absolute_import
import theano.tensor as T
import theano

from ..utils.format import check_list, unpack_list


class BaseObjective(object):
    """
    instance, that:
        - determines rewards for all actions agent takes given environment state and agent action,
    """

    def __init__(self):
        raise NotImplemented

    def reset(self, batch_size):
        """
        performs this action each time a new session [batch] is loaded
            batch size: size of the new batch
        """
        pass

    def get_reward(self, last_environment_states, agent_actions, batch_id):
        """
        WARNING! this function is computed on a single session, not on a batch!
        reward given for taking the action in current environment state
        arguments:
            last_environment_state float[time_i, memory_id]: environment state before taking action
            action int[time_i]: agent action at this tick
        returns:
            reward float[time_i]: reward for taking action
        """

        return T.zeros_like(agent_actions[0]).astype(theano.config.floatX)

    def get_reward_sequences(self, env_state_sessions, agent_action_sessions):
        """
        computes the rewards given to agent at each time step for each batch
        parameters:
            env_state_seq - environment state [batch_i,seq_i,state_units] history for all sessions
            agent_action_seq - int[batch_i,seq_i]
        returns:
            rewards float[batch_i,seq_i] - what reward was given to an agent for corresponding action from state in that batch

        """
        env_state_sessions = check_list(env_state_sessions)
        n_states = len(env_state_sessions)
        agent_action_sessions = check_list(agent_action_sessions)
        n_actions = len(agent_action_sessions)

        def compute_reward(batch_i, *args):
            session_states, session_actions = unpack_list(args, [n_states, n_actions])
            return self.get_reward(session_states, session_actions, batch_i)

        sequences = [T.arange(agent_action_sessions[0].shape[0], ), ] + env_state_sessions + agent_action_sessions

        rewards, updates = theano.map(compute_reward, sequences=sequences)

        assert len(updates) == 0
        return rewards.reshape(agent_action_sessions[0].shape)  # reshape bach to original
