from theano import tensor as T

__doc__ = """Base environment class that all experiments must inherit from (or mimic methods, if they feel cocky)"""

class BaseEnvironment:
    def __init__(self):
        """Create an environment setup. Please implement all the BaseEnvironment methods or make sure you understand
        why default implementations are okay for you
        Warning!
        if you keep experiencing a *.grad illegally returned an integer-valued variable. (Input index *, dtype *),
        please make sure that any non-float environment states are excluded from gradient computation or are behind the
        ConsiderConstant op. To find out which variable causes the problem, find all expressions of the dtype mentioned in
        the expression and then iteratively replace their type with a similar one (like int8 -> uint8 or int32) until the
        error message dtype changes. Once id does, you have found the cause of the exception.
        """
        raise NotImplemented

    @property 
    def state_size(self):
        """Environment state size"""
        raise NotImplemented
    @property 
    def observation_size(self):
        """Single observation size"""
        #base: mdp with full observability
        return self.state_size()
    
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

        #a dummy update rule where new state is equal to last state
        new_state = self.get_first_state(last_state.shape[0])
        observation = new_state #mdp with full observability
        return new_state, observation
        
        