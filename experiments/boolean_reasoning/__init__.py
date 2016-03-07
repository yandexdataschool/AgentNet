

__doc__="""
This is a dummy experiment setup that requires agent to make advantage of
a simple logical formula in order to maximize expected reward.

The world agent exists in has a number of boolean hidden factors:
X1~3, Y1, Y2.

The factors are not independent. Namely,
 - Y1 = (not X1) and X2
 - Y2 = not Y1

In the initial moment of time, agent knows nothing about any of them.
At each turn, agent may decide to
 - "open" one of the hidden factors.
   - if the factor turns out to be 1, agent receives +1 reward for X*, +3 for Y*
   - Otherwise, the reward equals -1 for X*, -3 for Y*
 - decide to quit session
   - yields reward of 0 and ends the interaction.
   - all farther actions will have no effect until next session

It is expected, that in order to maximize it's expected reward, the agent
will converge to a strategy of polling X for as long as they yield information
on Y, and than pick one of two Ys which is 1 and quit session. 

The experiment setup contains a single class BooleanReasoningEnvironment that
implements both BaseEnvironment and BaseObjective.
"""



import pandas as pd
import numpy as np

import theano
import theano.tensor as T


from AgentNet.objective import BaseObjective
from AgentNet.environment import BaseEnvironment

from AgentNet.utils.tensor_ops import in1d
from AgentNet.utils import create_shared,set_shared


class BooleanReasoningEnvironment(BaseObjective,BaseEnvironment):


    def __init__(self,batch_size = 10):
        
        n_attrs = 3
        n_categories = 2
        
        #fill shared variables with dummy values
        self.attributes = create_shared("X_attrs_data",np.zeros([batch_size,n_attrs]),'uint8')
        self.categories =  create_shared("categories_data",np.zeros([batch_size,n_categories]),'uint8')
        self.batch_size = self.attributes.shape[0]
        
        

        #"end_session_now" action
        end_action = T.zeros([self.batch_size,1], dtype='uint8')

        #concatenate data and cast it to float to avoid gradient problems
        self.joint_data = T.concatenate([self.attributes,
                                          self.categories,
                                          end_action,
                                         ],axis=1).astype(theano.config.floatX)
    
        #indices
        self.category_action_ids = T.arange(
            self.attributes.shape[1],
            self.attributes.shape[1]+self.categories.shape[1]
        )
        
        #last action id corresponds to the "end session" action
        self.end_action_id = self.joint_data.shape[1]-1
        
        #fill in one data sample
        self.generate_new_data_batch(batch_size)
    """generating data"""
    
    feature_names = ["X1","X2","X3","Y1","Y2","End_session_now"]
    n_actions = 6
       
    def _generate_data(self,batch_size):
        """generates data batch"""
        df = pd.DataFrame(np.random.randint(0,2,[batch_size,3]),columns=["X1","X2","X3"])
        df["Y1"] = np.logical_and(np.logical_not(df.X1), df.X2).astype(int)
        df["Y2"] = 1 - df["Y1"]
    
        return df[["X1","X2","X3"]].values, df[["Y1","Y2"]].values
    

    def set_data_batch(self,attrs_batch,categories_batch):
        """load data into model"""
        set_shared(self.attributes,attrs_batch)
        set_shared(self.categories,categories_batch)

    def generate_new_data_batch(self,batch_size=10):
        """this method generates new data batch and loads it into the environment. Returns None"""
        
        attrs_batch,categories_batch = self._generate_data(batch_size)
        
        self.set_data_batch(attrs_batch,categories_batch)
        
    """dimensions"""
    
    @property
    def observation_size(self):
        return int((self.joint_data.shape[1]+2).eval())
    @property
    def state_size(self):
        return int(self.joint_data.shape[1].eval())
    
    
    def get_whether_alive(self,observations_tensor):
        """Given observations, returns whether session has or has not ended.
        Returns uint8 [batch,time_tick] where 1 means session is alive and 0 means session ended already.
        Note that session is considered still alive while agent is commiting end_action
        """
        return T.eq(observations_tensor[:,:,1],0)
    """agent interaction"""
    
    def get_action_results(self,last_state,action,time_i):
        
        #state is a boolean vector: whether or not i-th action
        #was tried already during this session
        #last output[:,end_code] always remains 1 after first being triggered
        
        
        batch_range = T.arange(action.shape[0])

        session_active = T.eq(last_state[:,self.end_action_id],0)
        
        state_after_action = T.set_subtensor(last_state[batch_range,action],1)
        
        new_state = T.switch(
            session_active.reshape([-1,1]),
            state_after_action,
            last_state
        )
        
        
        
        observation = T.concatenate([
                self.joint_data[batch_range,action,None],#uint8[batch,1]
                ~session_active.reshape([-1,1]), #whether session has been terminated by now
                T.extra_ops.to_one_hot(action,self.joint_data.shape[1]),
            ],axis=1)
        
        return new_state, observation

    def get_reward(self,session_states,session_actions,batch_i):
        """
        WARNING! this runs on a single session, not on a batch
        reward given for taking the action in current environment state
        arguments:
            session_states float[batch_id, memory_id]: environment state before taking action
            session_actions int[batch_id]: agent action at this tick
        returns:
            reward float[batch_id]: reward for taking action from the given state
        """
        time_range = T.arange(session_actions.shape[0])
        

        has_tried_already = session_states[time_range,session_actions]
        session_is_active = T.eq(session_states[:,self.end_action_id],0)
        has_finished_now = T.eq(session_actions,self.end_action_id)
        action_is_categorical = in1d(session_actions, self.category_action_ids)
        
        response = self.joint_data[batch_i,session_actions].ravel()
        
        #categorical and attributes
        reward_for_intermediate_action = T.switch(
            action_is_categorical,
            response*6-3,
            response*2-1
        )
        #include end action
        reward_for_action = T.switch(
            has_finished_now,
            0,
            reward_for_intermediate_action,
        )
        
        reward_if_first_time = T.switch(
                has_tried_already,
                0,
                reward_for_action,
            )
        
        final_reward = T.switch(
            session_is_active,
            reward_if_first_time,
            0,

            
        )
        
        
        return final_reward.astype(theano.config.floatX)