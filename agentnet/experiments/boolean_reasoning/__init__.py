"""
Boolean reasoning is a dummy experiment setup that requires agent to make advantage of
a simple logical formula in order to maximize expected reward.

The world agent exists in has a number of boolean hidden factors:
X1, X2, X3, Y1, Y2.

The factors are not independent. Namely,
 - Y1 is true if X1
 - Y2 is true if not X1

In the initial moment of time, agent knows nothing about any of them.
At each turn, agent may decide to
 - "open" one of the hidden factors.
   - if the factor turns out to be 1, agent receives +1 reward for X*, +3 for Y*
   - Otherwise, the reward equals -1 for X*, -3 for Y*
   - checking a single factor more than once a session will result in -0.5 reward for every attempt but for first one
 - decide to quit session
   - yields reward of 0 and ends the interaction.
   - all farther actions will have no effect until next session

It is expected, that in order to maximize it's expected reward, the agent
will learn the policy
1. Check X1
2. If X1, check Y1, else Y2

The experiment setup contains a single class BooleanReasoningEnvironment that
implements both BaseEnvironment and BaseObjective.
"""
from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np

import theano
import theano.tensor as T

from agentnet.objective import BaseObjective
from agentnet.environment import BaseEnvironment

from agentnet.utils.tensor_ops import in1d
from agentnet.utils import create_shared, set_shared
from agentnet.utils.format import check_list

n_attrs = 3
n_categories = 2

x_names = list(map("X{}".format, list(range(1, n_attrs + 1))))
y_names = list(map("Y{}".format, list(range(1, n_categories + 1))))


class BooleanReasoningEnvironment(BaseObjective, BaseEnvironment):
    """generating data"""

    feature_names = x_names + y_names + ["End_session_now"]
    n_actions = n_attrs + n_categories + 1

    def __init__(self):
        # fill shared variables with dummy values
        self.attributes = create_shared("X_attrs_data", np.zeros([10, n_attrs]), 'uint8')
        self.categories = create_shared("categories_data", np.zeros([10, n_categories]), 'uint8')
        self.batch_size = self.attributes.shape[0]

        # "end_session_now" action
        end_action = T.zeros([self.batch_size, 1], dtype='uint8')

        # concatenate data and cast it to float to avoid gradient problems
        self.joint_data = T.concatenate([self.attributes,
                                         self.categories,
                                         end_action,
                                         ], axis=1).astype(theano.config.floatX)

        # indices
        self.category_action_ids = T.arange(
            self.attributes.shape[1],
            self.attributes.shape[1] + self.categories.shape[1]
        )

        # last action id corresponds to the "end session" action
        self.end_action_id = self.joint_data.shape[1] - 1

        # generate dummy data sample
        self.generate_new_data_batch(10)
        
        
        #fill in the shapes
        #single-element lists for states and observations
        observation_shapes=[int((self.joint_data.shape[1] + 2).eval())]
        env_state_shapes=[int(self.joint_data.shape[1].eval())]
        action_shapes=[(),]
        
        #use default dtypes: int32 for actions, floatX for states and observations
        
        BaseEnvironment.__init__(
            self,
            env_state_shapes,
            observation_shapes,
            action_shapes,
        )

    def _generate_data(self, batch_size):
        """generates data batch"""
        df = pd.DataFrame(np.random.randint(0, 2, [batch_size, n_attrs]),
                          columns=x_names)

        df[y_names[0]] = df.X1.astype('int32')
        # You can use any logical expressions here,
        # e.g. np.logical_or(
        #   np.logical_and(df.X1, np.logical_not(df.X2)),
        #   np.logical_and(df.X2, np.logical_not(df.X1))
        #    ).astype(int)
        df[y_names[1]] = 1 - df[y_names[0]]

        return df[x_names].values, df[y_names].values

    def set_data_batch(self, attrs_batch, categories_batch):
        """load data into model"""
        set_shared(self.attributes, attrs_batch)
        set_shared(self.categories, categories_batch)

    def generate_new_data_batch(self, batch_size=10):
        """this method generates new data batch and loads it into the environment. Returns None"""

        attrs_batch, categories_batch = self._generate_data(batch_size)

        self.set_data_batch(attrs_batch, categories_batch)

    
    def get_whether_alive(self, observation_tensors):
        """Given observations, returns whether session has or has not ended.
        Returns uint8 [batch,time_tick] where 1 means session is alive and 0 means session ended already.
        Note that session is considered still alive while agent is commiting end_action
        """
        observation_tensors = check_list(observation_tensors)
        return T.eq(observation_tensors[0][:, :, 1], 0)

    # agent interaction

    def get_action_results(self, last_states, actions):
        # state is a boolean vector: whether or not i-th action
        # was tried already during this session
        # last output[:,end_code] always remains 1 after first being triggered

        last_state = check_list(last_states)[0]
        action = check_list(actions)[0]

        batch_range = T.arange(action.shape[0])

        session_active = T.eq(last_state[:, self.end_action_id], 0)

        state_after_action = T.set_subtensor(last_state[batch_range, action], 1)

        new_state = T.switch(
            session_active.reshape([-1, 1]),
            state_after_action,
            last_state
        )

        session_terminated = T.eq(new_state[:, self.end_action_id], 1)

        observation = T.concatenate([
            self.joint_data[batch_range, action, None],  # uint8[batch,1]
            session_terminated.reshape([-1, 1]),  # whether session has been terminated by now
            T.extra_ops.to_one_hot(action, self.joint_data.shape[1]),
        ], axis=1)

        return new_state, observation

    def get_reward(self, state_sequences, action_sequences, batch_id):
        """
        WARNING! this runs on a single session, not on a batch
        reward given for taking the action in current environment state
        arguments:
            state_sequence float[batch_id, memory_id]: environment state before taking action
            action_sequence int[batch_id]: agent action at this tick
        returns:
            reward float[batch_id]: reward for taking action from the given state
        """

        state_sequence = check_list(state_sequences)[0]
        action_sequence = check_list(action_sequences)[0]

        time_range = T.arange(action_sequence.shape[0])

        has_tried_already = state_sequence[time_range, action_sequence]
        session_is_active = T.eq(state_sequence[:, self.end_action_id], 0)
        has_finished_now = T.eq(action_sequence, self.end_action_id)
        action_is_categorical = in1d(action_sequence, self.category_action_ids)

        response = self.joint_data[batch_id, action_sequence].ravel()

        # categorical and attributes
        reward_for_intermediate_action = T.switch(
            action_is_categorical,
            response * 6 - 3,
            response * 2 - 1
        )
        # include end action
        reward_for_action = T.switch(
            has_finished_now,
            0,
            reward_for_intermediate_action,
        )

        reward_if_first_time = T.switch(
            has_tried_already,
            -0.5,
            reward_for_action,
        )

        final_reward = T.switch(
            session_is_active,
            reward_if_first_time,
            0,
        )

        return final_reward.astype(theano.config.floatX)
