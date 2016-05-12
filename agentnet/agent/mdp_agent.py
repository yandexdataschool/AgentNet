from theano import tensor as T

import lasagne                 
from lasagne.layers import InputLayer,Layer

from .recurrence import Recurrence
from ..utils.format import supported_sequences,unpack_list,check_list,check_tuple,check_ordict

from collections import OrderedDict

class MDPAgent(object):
    def __init__(self,
                 observation_layers,
                 state_variables,
                 policy,
                 action_layers,
                ):
        """
        A generic agent within MDP abstraction,
        
            state_variables - OrderedDict{ memory_output: memory_input}, where
                memory_output: lasagne layer
                    - generates first (a-priori) agent state
                    - determines new agent state given previous agent state and an observation
                    
                memory_input: lasagne.layers.InputLayer that is used as "previous state" input for memory_output
            
            policy - whatever determines agent policy, lasagne.Layer child instance (e.g. Q-values) 
                    or a tuple of such instances (state value + action probabilities)
                     
            action_layers - agent's action, represented by resolver.BaseResolver child instance or any appropriate layer
                or a tuple of such, that can be fed into environment to get next state and observation.
                Basicly, whatever is fed into your environment as agent actions.
        """
        
        
        self.single_action = type(action_layers) not in supported_sequences
        self.single_policy = type(policy) not in supported_sequences
        self.single_observation = type(observation_layers) not in supported_sequences
        
        
        self.observation_layers = check_list(observation_layers)
        self.state_variables = check_ordict(state_variables)
        self.policy = check_list(policy)
        self.action_layers = check_list(action_layers)
        
        


    def as_recurrence(self, 
                      environment,
                      session_length = 10,
                      batch_size = None,
                      initial_env_states = 'zeros',
                      initial_observations = 'zeros',
                      initial_hidden = 'zeros',
                      **kwargs
                      ):
        """returns a Recurrence lasagne layer that contains :
        parameters:
            environment - an environment to interact with (BaseEnvironment instance)
            session_length - how many turns of interaction shall there be for each batch
            batch_size - [required parameter] amount of independed sessions [number or symbolic].
                rrelevant if there's at least one input or if you manually set any initial_*.
            
            initial_<something> - layers providing initial values for all variables at 0-th time step
                'zeros' default means filling variables with zeros
            Initial values are NOT included in history sequences
            flags: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)


        returns:
            
            an agentnet.agent.recurrence.Recurrence instance that returns
              [agent memory states] + [env states] + [env_observations] [agent policy] + [action_layers outputs]
                  all concatenated into one list
            
        """
        env = environment
        
        assert len(check_list(env.observation_shapes)) == len(self.observation_layers)
        
        #initialize prev states
        prev_env_states = [InputLayer((None,)+check_tuple(shape), 
                                        name="env.prev_state[%i]"%(i))
                             for i, shape in enumerate(check_list(env.state_shapes))]
        
        #apply environment changes after agent action
        new_state_outputs, new_observation_outputs = env.as_layers(prev_env_states,self.action_layers)
        
        
        
        #add state changes to memory dict
        all_state_pairs = self.state_variables.items() +\
                          zip(new_state_outputs, prev_env_states) +\
                          zip(new_observation_outputs, self.observation_layers)
        
        
        #compose state initialization dict
        state_init_pairs = []
        for initializers,layers in zip(
                    [initial_env_states,initial_observations, initial_hidden],
                    [new_state_outputs, new_observation_outputs, self.state_variables.keys()]
            ):
            if initializers == "zeros":
                continue  
            elif isinstance(initializers,dict):
                state_init_pairs += initializers.items()
            else:
                initializers = check_list(initializers)
                assert len(initializers) == len(layers)
                
                for layer,init in zip(layers,initializers):
                    if init is not None:
                        state_init_pairs.append([layer,init])
        
        #convert all inits into layers:
        for i in range(len(state_init_pairs)):
            layer,init = state_init_pairs[i]
            
            #replace theano variables with input layers for them
            if not isinstance(init, lasagne.layers.Layer):
                init_layer = InputLayer(layer.output_shape,
                                        name = "env.initial_values_for."+(layer.name or "state"),
                                        input_var = init)
                state_init_pairs[i][1] = init_layer
            
        
        #create the recurrence
        recurrence = Recurrence(
            state_variables = OrderedDict(all_state_pairs),
            state_init = OrderedDict(state_init_pairs),
            tracked_outputs = self.policy + self.action_layers,
            n_steps = session_length,
            batch_size = batch_size,
            delayed_states = new_state_outputs+ new_observation_outputs,
            **kwargs
        )
        
        
        
        return recurrence

    def get_sessions(self, 
                     environment,
                     session_length = 10,
                     batch_size = None,
                     initial_env_states = 'zeros',
                     initial_observations = 'zeros',
                     initial_hidden = 'zeros',
                     **kwargs
                  ):
        """returns history of agent interaction with environment for given number of turns:
        parameters:
            environment - an environment to interact with (BaseEnvironment instance)
            session_length - how many turns of interaction shall there be for each batch
            batch_size - [required parameter] amount of independed sessions [number or symbolic].Irrelevant if you manually set all initial_*.
            
            initial_<something> - layers or lists of layers or theano expressions providing initial <something> 
                for the first tick of recurrent applier. 
                If return_layers is True, they must be layers, if False - theano expressions
                Default 'zeros' means filling variables/layers with zeros of appropriate shape/type.
            
            return_layers - if True, works with lasagne layers and returns a bunch of them.
                    Otherwise (default) returns symbolic expressions.
                    Turning this on may be useful when traing nested MDP agents
            
            kwargs: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)


        returns:
        
        
            state_seq,observation_seq,hidden_seq,action_seq,policy_seq,
            for environment state, observation, hidden state, chosen actions and agent policy respectively
            each of them having dimensions of [batch_i,seq_i,...]

                    
            an agentnet.agent.recurrence.Recurrence instance that returns
              - state sequences - [agent memory states] + [env states] + [env_observations] concatenated into one list
              - output sequences - [agent policy] + [action_layers outputs] concatenated into one list
            
            
            time synchronization policy:
                env_states[:,i] was observed as observation[:,i] BASED ON WHICH agent generated his
                policy[:,i], resulting in action[:,i], and also updated his memory from hidden[:,i-1] to hiden[:,i]
            
        """
        env = environment
        
        
        
        #create recurrence
        recurrence = self.as_recurrence(environment=environment,
                                        session_length=session_length,
                                        batch_size = batch_size,
                                        initial_env_states=initial_env_states,
                                        initial_observations=initial_observations,
                                        initial_hidden=initial_hidden,
                                        **kwargs
                                       )
        state_layers_dict, output_layers = recurrence.get_sequence_layers()
        
        #convert sequence layers into actual theano varaibles
        theano_expressions = lasagne.layers.get_output(list(state_layers_dict.values())+output_layers)
        
        n_states = len(state_layers_dict)
        states_list, outputs = theano_expressions[:n_states], theano_expressions[n_states:]
        
        #sort sequences into categories
        agent_states,env_states,observations = unpack_list(states_list,
                                                           len(self.state_variables),
                                                           len(env.state_shapes),
                                                           len(env.observation_shapes))
        
        
        policy,actions = unpack_list(outputs,
                                     len(self.policy),
                                     len(self.action_layers),
                                    )
        
        agent_states = OrderedDict(zip(self.state_variables.keys(),agent_states))        
        
        #if user asked for single value and not one-element list, unpack the list
        if type(environment.state_shapes) not in supported_sequences:
            env_states = env_states[0]
        if self.single_observation:
            observations = observations[0]
        if self.single_action:
            actions = actions[0]
        if self.single_policy:
            policy = policy[0]
            
        
        return env_states,observations,agent_states,actions,policy
    
    
    def get_agent_reaction(self,prev_states={},current_observations=[],**flags):
        """
            prev_states: a dict [memory output: prev state]
            current_observations: a list of inputs where i-th input corresponds to 
                i-th input slot from self.observations
            flags: any flag that should be passed to the lasagne network for lasagne.layers.get_output method
            
            returns:
                actions: a list of all action layer outputs
                new_states: a list of all new_state values, where i-th element corresponds
                        to i-th self.state_variables key
            
        """

        
            
                 
        #standartize prev_states to a dicitonary
        if not hasattr(prev_states,'keys'):
            #if only one layer given, make a single-element list of it
            prev_states = check_list(prev_states)
            prev_states = OrderedDict(zip(self.state_variables.keys(),prev_states))
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
        prev_states_kv = [(self.state_variables[s],prev_states[s]) for s in self.state_variables.keys()] #prev states
        
        ##observation input layer: observation value
        observation_kv = zip(self.observation_layers,current_observations)
        
        input_map = OrderedDict(prev_states_kv + observation_kv)
        
        #compose output_list
        output_list = list(self.action_layers)+list(self.state_variables.keys()) + self.policy + self.action_layers
        
        #call get output
        results = lasagne.layers.get_output(
            layer_or_layers=output_list,
            inputs= input_map,
            **flags
          )
        
        #parse output array
        n_actions = len(self.action_layers)
        n_states = len(self.state_variables)
        n_outputs = len(self.policy + self.action_layers)
        
        new_actions,new_states,new_outputs = unpack_list(results,n_actions,n_states,n_outputs)
        
        return new_actions,new_states,new_outputs
    
    
   
