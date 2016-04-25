from ..environment import BaseEnvironment


class FeedbackEnvironment(BaseEnvironment):
    def __init__(self):
        """
        A generic pseudo-environment that sends back whatever agent's action was chosen.
        Has no state, no is_alive, nothing else.
        Does not implement BaseObjective or provide any rewards.
        """
        
    @property 
    def state_size(self):
        """Environment state size"""
        return []

    @property 
    def observation_size(self):
        """Single observation size"""
        raise NotImplemented, "FeedbackEnvironment does not have a pre-defined observation size."
    
    def get_action_results(self,last_states,actions,time_i):
        """
        computes environment state after processing agent's action
        arguments:
            last_state float[batch_id, memory_id0,[memory_id1],...]: environment state on previous tick
            action int[batch_id]: agent action after observing last state
        returns:
            new_state float[batch_id, memory_id0,[memory_id1],...]: environment state after processing agent's action
            observation float[batch_id,n_agent_inputs]: what agent observes after commiting the last action
        """
        return [],actions
        
