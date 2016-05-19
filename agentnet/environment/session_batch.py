from warnings import warn

from theano import tensor as T

from ..environment import BaseEnvironment
from ..objective import BaseObjective
from ..utils import insert_dim
from ..utils.format import check_list


class SessionBatchEnvironment(BaseEnvironment, BaseObjective):
    def __init__(self, observations, single_observation_shapes,
                 actions=None, single_action_shapes="all_scalar",
                 rewards=None, is_alive=None, preceding_agent_memory=None):
        """
        A generic pseudo-environment that replays sessions loaded on creation,
        ignoring agent actions completely.
        
        The environment takes symbolic expression for sessions represented as (.observations, .actions, .rewards)
        Unlike SessionPoolEnvironment, this one does not store it's own pool of sessions.
        
        To create experience-replay sessions, call Agent.get_sessions with this as an environment.
        During experience replay sessions,
         - states are replaced with a fake one-unit state
         - observations, actions and rewards match original ones
         - agent memory states, Q-values and all in-agent expressions (but for actions) will correspond to what
           agent thinks NOW about the replay.
         - is_alive [optional] - whether or not session has still not finished by a particular tick
         - preceding_agent_memory [optional] - what was agent's memory state prior to the first tick of the replay session.
         
        
        Although it is possible to get rewards via the regular functions, it is usually faster to take self.rewards as rewards
        with no additional computation.
        
        """

        # setting environmental variables. Their shape is [batch,time,something]
        self.observations = check_list(observations)
        self.single_observation_shapes = check_list(single_observation_shapes)

        for obs, obs_shape in zip(self.observations, self.single_observation_shapes):
            assert obs.ndim == len(obs_shape) + 2

        if actions is not None:
            self.actions = check_list(actions)
            self.single_action_shapes = single_action_shapes
            if self.single_action_shapes == "all_scalar":
                self.single_action_shapes = [tuple()] * len(self.actions)

        self.rewards = rewards
        self.is_alive = is_alive

        if preceding_agent_memory is not None:
            self.preceding_agent_memory = check_list(preceding_agent_memory)

        self.padded_observations = [
            T.concatenate([obs, insert_dim(T.zeros_like(obs[:, 0]), 1)], axis=1)
            for obs in self.observations
            ]

        self.batch_size = self.observations[0].shape[0]
        self.sequence_length = self.observations[0].shape[1]

    @property
    def state_shapes(self):
        """Environment state sizes. In this case, it's a timer"""
        return [tuple()]

    @property
    def state_dtypes(self):
        """environment state dtypes. In this case, it's a timer"""
        return ["int32"]

    @property
    def observation_shapes(self):
        """observation shapes"""
        return self.single_observation_shapes

    @property
    def observation_dtypes(self):
        """observation dtypes"""
        return [obs.dtype for obs in self.observations]

    @property
    def action_shapes(self):
        """action shapes"""
        assert self.actions is not None
        return self.single_action_shapes

    @property
    def action_dtypes(self):
        """action dtypes"""
        return [act.dtype for act in self.actions]

    # TODO kwargs
    def get_action_results(self, last_states, actions):
        """
        computes environment state after processing agent's action
        arguments:
            last_state float[batch_id, memory_id0,[memory_id1],...]: environment state on previous tick
            action int[batch_id]: agent action after observing last state
        returns:
            new_state float[batch_id, memory_id0,[memory_id1],...]: environment state after processing agent's action
            observation float[batch_id,n_agent_inputs]: what agent observes after commiting the last action
        """
        time_i = check_list(last_states)[0]

        batch_range = T.arange(time_i.shape[0])

        new_observations = [obs[batch_range, time_i + 1] for obs in self.padded_observations]
        return [time_i + 1], new_observations

    def get_reward(self, session_states, session_actions, batch_id):
        """
        WARNING! this runs on a single session, not on a batch
        reward given for taking the action in current environment state
        arguments:
            session_states float[batch_id, memory_id]: environment state before taking action
            session_actions int[batch_id]: agent action at this tick
        returns:
            reward float[batch_id]: reward for taking action from the given state
        """
        warn("Warning - a session pool has all the rewards already stored as .rewards property."
             "Recomputing them this way is probably just a slower way of calling your_session_pool.rewards")
        return self.rewards[batch_id, :]
