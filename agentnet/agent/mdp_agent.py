import lasagne                 
from lasagne.utils import unroll_scan
from theano import tensor as T
from ..utils import insert_dim

from ..utils.format import supported_sequences

from .base import BaseAgent

class MDPAgent(BaseAgent):
    def __init__(self,
                 observation_layers,
                 memory_dict,
                 policy,
                 resolver,
                ):
        """
        A generic agent within MDP abstraction,
        
            memory_dict - OrderedDict{ memory_output: memory_input}, where
                memory_output: lasagne layer
                    - generates first (a-priori) agent state
                    - determines new agent state given previous agent state and an observation
                    
                memory_input: lasagne.layers.InputLayer that is used as "previous state" input for memory_output
            
            policy - lasagne.Layer child instance that
                - determines Q-values or probabilities for all actions given current agent state and current observation,
                - can .get_output_for(hidden_state)
            resolver - resolver.BaseResolver child instance that
                - determines agent's action given Q-values for all actions
        """
        
        
        self.single_resolver = type(resolver) not in supported_sequences
        self.single_policy = type(policy) not in supported_sequences
        self.single_observation = type(observation_layers) not in supported_sequences
            
        
        super(MDPAgent, self).__init__(observation_layers,resolver,memory_dict,policy)
        
        


    def get_sessions(self, 
                     environment,
                     session_length = 10,
                     batch_size = None,
                     initial_env_state = 'zeros',initial_observation = 'zeros',initial_hidden = 'zeros',
                     **flags
                     ):
        """returns history of agent interaction with environment for given number of turns:
        parameters:
            environment - an environment to interact with (BaseEnvironment instance)
            session_length - how many turns of interaction shall there be for each batch
            batch_size - [required parameter] amount of independed sessions [number or symbolic].Irrelevant if you manually set all initial_*.
            
            initial_<something> - initial values for all variables at 0-th time step
            Unless you are doing something nasty, initial policy (qvalues) and actions will not matter at all
            'zeros' default means filling variable with zeros
            Initial values are NOT included in history sequences
            additional_output_layers - any layers of a network which outputs need to be added to the outputs
            flags: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)


        returns:
            state_seq,observation_seq,hidden_seq,action_seq,policy_seq, [additional_output_0, additional_output_1]
            for environment state, observation, hidden state, agent policy and chosen actions respectively
            each of them having dimensions of [batch_i,seq_i,...]
            
            
            time synchronization policy:
                state_seq,observation_seq correspond to observation BASED ON WHICH agent generated hidden_seq,policy_seq,action_seq
            
        """
        
        groups = super(MDPAgent, self).get_sessions(environment=environment,
                                                  session_length=session_length,
                                                  batch_size = batch_size,
                                                  initial_env_states=initial_env_state,
                                                  initial_observations=initial_observation,
                                                  initial_state_variables=initial_hidden,
                                                  **flags
                                                  )
        env_states,observations,agent_states,actions,policy = groups
        
        if type(environment.state_size) not in supported_sequences:
            env_states = env_states[0]
        if self.single_observation:
            observations = observations[0]
        if self.single_resolver:
            actions = actions[0]
        if self.single_policy:
            policy = policy[0]
        return env_states,observations,agent_states,actions,policy
        