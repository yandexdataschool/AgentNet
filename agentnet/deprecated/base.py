import lasagne                 
from lasagne.utils import unroll_scan
from theano import tensor as T
from ..utils import insert_dim

import numpy as np
from collections import OrderedDict
from warnings import warn


from ..utils.format import check_list,check_ordict,unpack_list


from warnings import warn


class BaseAgent(object):
    def __init__(self,
                 observation_layers = [],
                 action_layers = [],
                 state_variables = OrderedDict(),
                 tracked_outputs = [],
                ):
        """
        write me someone
        
        action_layers - anything that should be passed to the environment
        observation_layers - anything that should be taken from environtment
        state_variables - anything that is carried on from i-1'th state to i'th
                - maps output to next input.
                  Format: {lasagne layer for output: lasagne InputLayer for "prev state" input}
                  
        tracked_outputs - anything that one should keep track of along iterations
        """
        warn("BaseAgent class is deprecated and will be removed in 0.0.9.\n"\
             "Consider using agentnet.agent.Recurrence with all BaseAgent features and more capabilities.")
        
        self.observation_layers = check_list(observation_layers)
        self.action_layers = check_list(action_layers)
        self.tracked_outputs = check_list(tracked_outputs)
        
        
        self.state_variables = state_variables 
        if type(state_variables) is not OrderedDict:
            
            self.state_variables = check_ordict(self.state_variables)
            
            if len(self.state_variables) >1:
                warn("It is recommended that state_variables is an ordered dict.\n"\
                     "Otherwise, order of agent state outputs from get_sessions and get_agent_reaction methods\n"\
                     "may depend on python configuration. Current order is:"+ str(list(self.state_variables.keys()))+"\n"\
                     "You may find OrderedDict in standard collections module: from collections import OrderedDict")
            
        
        for memory_in, memory_out in list(self.state_variables.items()):
            assert tuple(memory_in.output_shape) == tuple(memory_out.output_shape)

        
    def get_agent_reaction(self,prev_states={},current_observations=[],additional_outputs=[],**flags):
        """
            prev_states: a dict [memory output: prev state]
            current_observations: a list of inputs where i-th input corresponds to 
                i-th input slot from self.observations
            flags: any flag that should be passed to the lasagne network for lasagne.layers.get_output method
            additional outputs: more "tracked_outputs" to be appended to original ones
            
            
            returns:
                actions: a list of all action layer outputs
                new_states: a list of all new_state values, where i-th element corresponds
                        to i-th self.state_variables key
            
        """

        
            
                 
        #standartize prev_states to a dicitonary
        if not hasattr(prev_states,'keys'):
            #if only one layer given, make a single-element list of it
            prev_states = check_list(prev_states)
            prev_states = OrderedDict(list(zip(list(self.state_variables.keys()),prev_states)))
        else:
            prev_states = check_ordict(prev_states)
            
        ##check that the size matches
        assert len(prev_states) == len(self.state_variables)
        
        #standartize current observations to a list
        current_observations = check_list(current_observations)
        ##check that the size matches
        assert len(current_observations) == len(self.observation_layers)
        
            
            
            
        #compose input map
        
        ##state input layer: prev state
        prev_states_kv = [(self.state_variables[s],prev_states[s]) for s in list(self.state_variables.keys())] #prev states
        
        ##observation input layer: observation value
        observation_kv = list(zip(self.observation_layers,current_observations))
        
        input_map = OrderedDict(prev_states_kv + observation_kv)
        
        #compose output_list
        output_list = list(self.action_layers)+list(self.state_variables.keys()) + list(self.tracked_outputs)
        
        #call get output
        results = lasagne.layers.get_output(
            layer_or_layers=output_list,
            inputs= input_map,
            **flags
          )
        
        #parse output array
        n_actions = len(self.action_layers)
        n_states = len(self.state_variables)
        n_outputs = len(self.tracked_outputs)
        
        new_actions,new_states,new_outputs = unpack_list(results,n_actions,n_states,n_outputs)
        
        return new_actions,new_states,new_outputs

    def get_sessions(self, 
                     environment = None,
                     session_length = 10,
                     batch_size = None,
                     initial_env_states = 'zeros',
                     initial_observations = 'zeros',
                     initial_state_variables = 'zeros',
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
            state_seq,observation_seq,hidden_seq,policy_seq,action_seq, [additional_output_0, additional_output_1]
            for environment state, observation, hidden state, agent policy and chosen actions respectively
            each of them having dimensions of [batch_i,seq_i,...]
            
            
            time synchronization policy:
                state_seq[:,i],observation_seq[:,i] correspond to observation BASED ON WHICH 
                agent generated hidden_seq[:,i],policy_seq[:,i],action_seq[:,i],
                 also using his previous memory hidden_seq[:,i-1]
            
        """
        env = environment
        
        #assert that environment is None if and only if there are no observations
        assert (env is None) == (len(self.observation_layers) == 0)
        
        if env is not None:
            
            if initial_env_states == 'zeros':
                
                initial_env_states = [T.zeros((batch_size,)+size,
                                             dtype=dtype) 
                                      for size,dtype in zip(check_list(env.state_shapes),check_list(env.state_dtypes))]
            else:
                initial_env_states = check_list(initial_env_states)

            if initial_observations == 'zeros':
                initial_observations = [T.zeros((batch_size,)+tuple(obs_layer.shape[1:])) 
                                        for obs_layer in self.observation_layers]
            else:
                initial_observations = check_list(initial_observations)
                
                
        else:
            initial_env_states = initial_observations = []
            
            
            
        if initial_state_variables == 'zeros':
            initial_state_variables = []
            for memory in self.state_variables:
                
                state_shape = lasagne.layers.get_output_shape(memory)[1:] #drom batch_i dimension
                initial_state = T.zeros((batch_size,)+tuple(state_shape))
                initial_state_variables.append(initial_state)
        
        

        #recurrent step function
        #during SCAN, time synchronization is reverse: state_1 came after action_1 based on observation_0 from state_0
        def step(time_tick,*args):

            
            #slice previous: they contain 
            #[*env_states_if_any, *observations, *state_variables, *prev_actions, *prev_outputs, *rubbish]
            # we only need env state, prev observation and agent state to iterate on
            
            if env is not None:
                n_env_states = len(check_list(env.state_shapes))
            else:
                n_env_states = 0
            
            n_observations = len(self.observation_layers)
            
            n_memories = len(self.state_variables)
            
            env_states,observations,prev_agent_states = unpack_list(args,n_env_states,n_observations,n_memories)
            
            
            prev_states_dict = OrderedDict(list(zip(list(self.state_variables.keys()),prev_agent_states)))
            
            new_actions,new_agent_states,new_outputs = self.get_agent_reaction(prev_states_dict,observations,**flags)
            
            
            if env is not None: 
                new_env_states,new_observations = env.get_action_results(env_states,new_actions)#,time_tick)
                new_env_states = check_list(new_env_states)
                new_observations = check_list(new_observations)
            else:
                new_env_states = new_observations = []

            return new_env_states + new_observations + new_agent_states + new_actions + new_outputs

        #main recurrent loop configuration
        outputs_info = initial_env_states+initial_observations + initial_state_variables+\
                       [None]*(len(self.action_layers)+len(self.tracked_outputs))
        
        
        
                
        time_ticks = T.arange(session_length)
        sequences = [time_ticks]
        
        
        history = unroll_scan(step,
            sequences = sequences,
            outputs_info = outputs_info,
            non_sequences = [],
            n_steps = session_length
        )

        #for the record
        self.last_history = history
        #from [time,batch,...] to [batch,time,...]
        history = [ (var.swapaxes(1,0) if var.ndim >1 else var) for var in history]
        
        
        
        groups = unpack_list(history, 
                             len(initial_env_states),len(self.observation_layers),
                             len(self.state_variables),len(self.action_layers),
                             len(self.tracked_outputs))
        
        env_state_sequences, observation_sequences, agent_state_sequences,\
            action_sequences, output_sequences = groups
        
        
        agent_state_dict = OrderedDict(list(zip(list(self.state_variables.keys()),agent_state_sequences)))
        
        #allign time axes: actions come AFTER states with the same index
        #add first env turn, crop to session length
        env_state_sequences = [
            T.concatenate([insert_dim(initial_env_state,1),
                           state_seq[:,:-1]],axis=1)
            for state_seq, initial_env_state in 
                zip(env_state_sequences, initial_env_states)
        ]

        observation_seqs = [
            T.concatenate([insert_dim(initial_observation,1),
                           observation_seq[:,:-1]],axis=1)
            for observation_seq, initial_observation in 
                zip(observation_sequences, initial_observations)
        ]
            
        
        
        return env_state_sequences, observation_sequences, agent_state_dict,action_sequences, output_sequences