from theano import tensor as T

class BaseEnvironment:
    def __init__(self):
        raise NotImplemented

    def reset(self,batch_size):
        """
        performs this action each time a new session [batch] is loaded
            batch size: size of the new batch
        """
        pass
    @property 
    def state_size(self):
        """must return integer(not symbolic)"""
        raise NotImplemented
    @property 
    def observation_size(self):
        """must return integer(not symbolic)"""
        raise NotImplemented
    
    def get_action_results(self,last_state,action,time_i):
        """
        computes environment state after processing agent's action
        arguments:
            last_state float[batch_id, memory_id0,[memory_id1],...]: environment state on previous tick
            action int[batch_id]: agent action after observing last state
        returns:
            new_state float[batch_id, memory_id0,[memory_id1],...]: environment state after processing agent's action
            observation float[batch_id,n_agent_inputs]: what agent observes after commiting the last action
        """

        
        new_state = self.get_first_state(last_state.shape[0])
        observation = new_state #mdp with full observability
        return new_state, observation
        
        