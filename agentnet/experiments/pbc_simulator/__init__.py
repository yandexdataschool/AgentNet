
#TODO(jheuristic) write __doc__
__doc__="""
This is a sketch of PBCSEQ-based automated treatment experiment
"""

import os
import sys
experiment_path = '/'.join(__file__.split('/')[:-1])
dataset_path = os.path.join(experiment_path,"pbcseq.csv")

import pandas as pd
import numpy as np

import theano
import theano.tensor as T


from agentnet.objective import BaseObjective
from agentnet.environment import BaseEnvironment

from agentnet.utils.tensor_ops import in1d
from agentnet.utils import create_shared,set_shared
from agentnet.utils.format import check_list


from collections import defaultdict

default_rewards = defaultdict(lambda:0)
default_rewards["tick_reward"] = 0
default_rewards["end_action_reward"] = 1
default_rewards["stage_exaggerated_penalty_per_point"] = -1
default_rewards["stage_underestimated_penalty_per_point"] = -1
default_rewards["repeated_poll"]=-0.5
default_rewards["unfinished"] = -5



class PBCEnvironment(BaseObjective,BaseEnvironment):


    def __init__(self,rewards = default_rewards,):

        #fill shared variables with dummy values
        self.attributes = create_shared("patient_attributes",np.zeros([1,1]),'float32')
        self.disease_stages =  create_shared("disease_stage_indicator", np.zeros([1, 1]), 'uint8')

        self.batch_size = self.attributes.shape[0]


        #concatenate data and cast it to float to avoid gradient problems
        self.joint_data = T.concatenate([self.attributes,
                                          self.disease_stages,
                                         ],axis=1).astype(theano.config.floatX)
    
        #indices
        self.terminal_action_ids = T.arange(
            self.attributes.shape[1],
            self.attributes.shape[1]+self.disease_stages.shape[1]
        )
        
        self.rw = rewards
        
        
        
        # dimensions
        data_attrs, data_cats, _ = self.get_dataset()
        n_actions = data_cats.shape[1] + data_attrs.shape[1]
        env_state_shapes = (n_actions + 1,)
        observation_shapes = (data_attrs.shape[1] + 1 +n_actions,)
        self.n_actions = n_actions
        # the rest is default

        # init default (some shapes will be overloaded below
        BaseEnvironment.__init__(self,
                                 env_state_shapes,
                                 observation_shapes)

    """reading/loading data"""

    def get_dataset(self,dataset_path = dataset_path):
        print dataset_path
        df = pd.read_csv(dataset_path)
        attr_columns = list(df.columns[:-1])
        df_attributes = df[attr_columns]

        from sklearn.preprocessing import OneHotEncoder
        df_stages = OneHotEncoder(sparse=False).fit_transform(df.histologic_stage_of_disease[:, None])
        stage_columns = ["diagnosis:stage%i"%(stage+1) for stage in range(df_stages.shape[1]) ]

        return df_attributes,df_stages,attr_columns+stage_columns
    def load_data_batch(self, attrs_batch, stages_batch):
        """load data into model"""
        set_shared(self.attributes,attrs_batch)
        set_shared(self.disease_stages, stages_batch)
        
    def load_random_batch(self,df_attributes, df_stages, batch_size=10):
        """load batch_size random samples from given data attributes and categories"""
        df_attributes, df_stages = map(np.array, [df_attributes, df_stages])
        assert len(df_attributes) == len(df_stages)

        batch_ids = np.random.randint(0,len(df_attributes),batch_size)
        self.load_data_batch(df_attributes[batch_ids],df_stages[batch_ids])

         
    
    def get_whether_alive(self,observation_tensors):
        """Given observations, returns whether session has or has not ended.
        Returns uint8 [batch,time_tick] where 1 means session is alive and 0 means session ended already.
        Note that session is considered still alive while agent is committing terminal action
        """
        observation_tensors = check_list(observation_tensors)
        return T.eq(observation_tensors[0][:,:,-1],0)


    
    """agent interaction"""
    
    def get_action_results(self,last_states,actions,**kwargs):
        
        #unpack state and action
        last_state = check_list(last_states)[0]
        action = check_list(actions)[0]
        
        #state is a boolean vector: whether or not i-th action
        #was tried already during this session
        #last output[:,end_code] always remains 1 after first being triggered
        

        

        #whether session was active before tick
        session_active =  T.eq(last_state[:,-1],0)
        #whether session was terminated by the end of this tick
        session_terminated = T.or_(T.eq(session_active,0),in1d(action, self.terminal_action_ids))

        batch_range = T.arange(action.shape[0])
        state_after_action = T.set_subtensor(last_state[batch_range,action],1)
        state_after_action = T.set_subtensor(state_after_action[:,-1],session_terminated)

        new_state = T.switch(
            session_active.reshape([-1,1]),
            state_after_action,
            last_state
        )


        #if allowed to see attribute
        observed_attrs = T.switch(state_after_action[:,:self.attributes.shape[1]],
                                  self.attributes,
                                  -1
                                )


        observation = T.concatenate([
                observed_attrs,#float32[batch,1] response
                T.extra_ops.to_one_hot(action,self.joint_data.shape[1]), #what action was commited
                session_terminated.reshape([-1, 1]),  # whether session is terminated by now
        ],axis=1)
        
        return new_state, observation

    def get_reward(self, session_states, session_actions, batch_id):
        """
        WARNING! this runs on a single session, not on a batch
        reward given for taking the action in current environment state
        arguments:
            session_states float[time, memory_id]: environment state before taking action
            session_actions int[time]: agent action at this tick
        returns:
            reward float[time]: reward for taking action from the given state
        """
        #unpach states and actions
        session_states = check_list(session_states)[0]
        session_actions = check_list(session_actions)[0]
        
        
        time_range = T.arange(session_actions.shape[0])
        

        has_tried_already = session_states[time_range,session_actions]
        session_is_active = T.eq(session_states[:,-1],0)
        

        action_is_terminal= in1d(session_actions, self.terminal_action_ids)

        at_least_one_terminal_action = T.gt(T.cumsum(action_is_terminal,axis=0),0)

        has_finished_now = T.set_subtensor(action_is_terminal[-1],1)
        end_tick = has_finished_now.nonzero()[0][0]

        
        #categorical and attributes
        reward_for_intermediate_action= T.switch(
                has_tried_already,
                self.rw["repeated_poll"],
                self.rw["tick_reward"],
            )



        correct_stage = T.argmax(self.disease_stages[batch_id])+1
        predicted_stage = session_actions - self.attributes.shape[1]

        exaggeration_penalty = T.maximum(predicted_stage - correct_stage,0)*\
                               self.rw["stage_exaggerated_penalty_per_point"]
        underestimation_penalty = T.maximum(correct_stage - predicted_stage,0)*\
                                  self.rw["stage_underestimated_penalty_per_point"]

        diagnosis_reward = self.rw["end_action_reward"] + exaggeration_penalty + underestimation_penalty



        #ending session
        reward_for_end_action = T.switch(at_least_one_terminal_action, #if at least 1 diagnosis chosen
                                         diagnosis_reward,   # than score diagnosis
                                         self.rw["unfinished"])  #else punish for no diagnosis
        
        #include end action
        reward_for_action = T.switch(
            has_finished_now,
            reward_for_end_action,
            reward_for_intermediate_action,
        )

        final_reward = T.switch(
            session_is_active,
            reward_for_action,
            0,
        )
        
        
        return final_reward.astype(theano.config.floatX)
    
    
