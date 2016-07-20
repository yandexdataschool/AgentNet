"""
'sessions' module contains several tools to print or plot agent's actions and state throughout the session
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt


def _select_action_policy(policy_seq, action_seq):
    """
    :param policy_seq: policy for every [batch,tick,action]
    :param action_seq: actions taken on every [batch,tick]
    :return: returns policy for selected actions of shape [batch, tick]
    """
    batch_i = np.arange(policy_seq.shape[0])[:, None]
    time_i = np.arange(policy_seq.shape[1])[None, :]

    return policy_seq[batch_i, time_i, action_seq]


def print_sessions(policy_seq, action_seq, reward_seq, action_names=None,
                   is_alive_seq=None, reference_policy_seq=None,
                   pattern=" {action}(qv = {q_values_predicted}) -> {reward}{q_values_reference} |",
                   plot_policy=True, hidden_seq=None, legend=True,
                   qv_line_width=lambda action_i: 1
                   ):
    """
    Prints the sequence of agent actions along with specified predicted policy

    :param policy_seq: policy for every [batch,tick,action]
    :param action_seq: actions taken on every [batch,tick]
    :param reward_seq: rewards given for action_seq on [batch,tick]
    :param action_names: names of all [actions]. Defaults to "action #i"
    :param is_alive_seq: whether or not session is terminated by [batch, tick]. Defaults to always alive
    :param reference_policy_seq: policy reference  for CHOSEN actions for each [batch,tick]
    :param pattern: how to print a single action cycle. Can use action, q_values_predicted, reward and q_values_reference variables.
    """

    if action_names is None:
        action_names = ["action #{}".format(x) for x in range(np.max(action_seq))]
    if is_alive_seq is None:
        is_alive_seq = np.ones_like(action_seq)
    if reference_policy_seq is None:
        # dummy values
        reference_policy_seq = policy_seq
        print_reference = False
    else:
        print_reference = True

    # if we are on;y given one session, reshape everything as a 1-session batch
    if len(action_seq.shape) == 1:
        policy_seq, action_seq, reward_seq, is_alive_seq, reference_policy_seq = \
            [v[None, :] for v in [policy_seq, action_seq, reward_seq, is_alive_seq, reference_policy_seq]]

    # if all policy values are given for [batch,tick,action], select policy values for taken actions
    assert len(policy_seq.shape) == 3
    if len(reference_policy_seq.shape) == 3:
        reference_policy_seq = _select_action_policy(reference_policy_seq, action_seq)

    # loop over sessions
    for session_id in range(policy_seq.shape[0]):

        time_range = np.arange(policy_seq.shape[1])
        session_tuples = zip(policy_seq[session_id, time_range, action_seq[session_id]],
                             action_seq[session_id],
                             reward_seq[session_id],
                             reference_policy_seq[session_id],
                             is_alive_seq[session_id])

        # print session log
        print("session #", session_id)

        for time_id, (q_values_predicted, action, reward, q_values_reference, is_alive) in enumerate(session_tuples):
            if not is_alive:
                print('\n')
                break

            if print_reference:
                q_values_reference = "(ref = {})".format(q_values_reference)
            else:
                q_values_reference = ""

            action_name = action_names[action]

            print(pattern.format(action=action_name, q_values_predicted=q_values_predicted, reward=reward,
                                 q_values_reference=q_values_reference), end=' ')

        else:
            print("reached max session length")

        # plot policy, actions, etc
        if plot_policy:
            plt.figure(figsize=[16, 8])

            session_len = time_id

            # plot limits
            plt.xlim(0, max(session_len * 1.1, 2))
            plt.xticks(np.arange(session_len))
            plt.grid()

            q_values = policy_seq[session_id].T
            for action in range(q_values.shape[0]):
                plt.plot(q_values[action], label=action_names[action], linewidth=qv_line_width(action))

            if hidden_seq is not None:
                hidden_activity = hidden_seq[session_id].T

                for i, hh in enumerate(hidden_activity):
                    plt.plot(hh, '--', label='hidden #' + str(i))

            session_actions = action_seq[session_id, :session_len]
            action_range = np.arange(len(session_actions))

            plt.scatter(action_range, q_values[session_actions, action_range])

            if legend:
                plt.legend()

            # session end line
            plt.plot(np.repeat(session_len - 1, 2), plt.ylim(), c='blue')

            plt.show()
