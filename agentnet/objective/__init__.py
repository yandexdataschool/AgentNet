import theano.tensor as T
import theano
import numpy as np


__doc__="""Base agent objective class that defines when does agent get reward"""

class BaseObjective:
    """
    instance, that:
        - determines rewards for all actions agent takes given environment state and agent action,
    """
    def __init__(self):
        raise NotImplemented
    def reset(self,batch_size):
        """
        performs this action each time a new session [batch] is loaded
            batch size: size of the new batch
        """
        pass
    def get_reward(self,last_environment_state,agent_action,batch_i):
        """
        WARNING! this function is computed on a session, not on a batch!
        reward given for taking the action in current environment state
        arguments:
            last_environment_state float[time_i, memory_id]: environment state before taking action
            action int[time_i]: agent action at this tick
        returns:
            reward float[time_i]: reward for taking action
        """
        
        return T.zeros_like(agent_action).astype(theano.config.floatX)

    def get_reward_sequences(self,env_state_sessions,agent_action_sessions):
        """
        computes the rewards given to agent at each time step for each batch
        parameters:
            env_state_seq - environment state [batch_i,seq_i,state_units] history for all sessions
            agent_action_seq - int[batch_i,seq_i]
        returns:
            rewards float[batch_i,seq_i] - what reward was given to an agent for corresponding action from state in that batch

        """
        
        
        
        def compute_reward(batch_i,session_states,session_actions):
            return self.get_reward(session_states,session_actions,batch_i)



        sequences = [
            T.arange(env_state_sessions.shape[0],),
            env_state_sessions,
            agent_action_sessions,
        ]

        rewards,updates = theano.map(compute_reward,
                              sequences=sequences)
        assert len(updates)==0
        return rewards.reshape(agent_action_sessions.shape) #reshape bach to original
    
