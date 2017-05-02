"""
MDPAgent provides a high-level interface for quick implementation of classic MDP agents with continuous,
discrete or mixed action space, arbitrary recurrent agent memory and decision making policy.

If you are up to something more sophisticated, try agentnet.agent.recurrence.Recurrence,
 which is a lasagne layer for custom recurrent networks.


"""
from __future__ import division, print_function, absolute_import

from collections import OrderedDict
from itertools import chain

import numpy as np
import theano
import lasagne
from lasagne.layers import InputLayer
from theano import tensor as T

from .recurrence import Recurrence
from ..environment import SessionPoolEnvironment, SessionBatchEnvironment, BaseEnvironment
from ..utils.format import supported_sequences, unpack_list, check_list, check_tuple, check_ordered_dict
from ..utils.layers import get_layer_dtype
from ..utils.logging import warn





class MDPAgent(object):
    """
    A generic agent within MDP (markov decision process) abstraction.
    Basically wraps Recurrence layer to interact between agent and environment.
    Note for developers: if you want to get acquainted with this code, we suggest reading 
    [Recurrence](http://agentnet.readthedocs.io/en/master/modules/agent.html#module-agentnet.agent.recurrence) first.

    :param observation_layers: agent observation(s)
    :type observation_layers: lasagne.layers.InputLayer or a list of such

    :param action_layers: agent's action(s), or whatever is fed into your environment as agent actions.
    :type action_layers: resolver.BaseResolver child instance or any appropriate layer
            or a tuple of such, that can be fed into environment to get next state and observation.

    :param agent_states: OrderedDict{ memory_output: memory_input}, where
        memory_output: lasagne layer
        generates first agent state (before any interaction)
        determines new agent state given previous agent state and an observation
        memory_input: lasagne.layers.InputLayer that is used as "previous state" input for memory_output

    :type agent_states: collections.OrderedDict or dict

    :param policy_estimators: whatever determines agent policy (or whatever you want to work with later).
        - Q_values (and target network q-values) for q-learning
        - action probabilities for reinforce
        - action probabilities and state values (also possibly target network) for actor-critic
        - whatever intermediate state you want. e.g. if you want to penalize network for activations
        of layer `l_dense_1` later, you will need to add it to policy_estimators.
    :type policy_estimators: lasagne.Layer child instance (e.g. Q-values) or a tuple of such instances
            (e.g. state value + action probabilities for a2c)



    """
    def __init__(self,
                 observation_layers=(),
                 agent_states={},
                 policy_estimators=(),
                 action_layers=(),
                 ):


        self.single_action = type(action_layers) not in supported_sequences
        self.single_policy = type(policy_estimators) not in supported_sequences
        self.single_observation = type(observation_layers) not in supported_sequences

        self.observation_layers = check_list(observation_layers)
        self.agent_states = check_ordered_dict(agent_states)
        self.policy = check_list(policy_estimators)
        self.action_layers = check_list(action_layers)
        
    def get_react_function(self, output_flags={},
                           function_flags={'allow_input_downcast': True}):
        """
        compiles and returns a function that performs one step of agent network

        :returns: a theano function.
            The returned function takes all observation inputs, followed by all agent memories.
            It's outputs are all actions, followed by all new agent memories
            By default, the function will have allow_input_downcast=True, you can override it in function parameters
        :rtype: theano.function


        :Example:

        The regular use case would look something like this:
        (assuming agent is an MDPagent with single observation, single action and 2 memory slots)
        >> react = agent.get_react_function
        >> action, new_mem0, new_mem1 = react(observation, mem0, mem1)


        """
        # observation variables
        applier_observations = [layer.input_var for layer in self.observation_layers]

        # inputs to all agent memory states (using lasagne defaults, may use any theano inputs)
        applier_memories = OrderedDict([(new_st, prev_st.input_var)
                                        for new_st, prev_st in list(self.agent_states.items())
                                        ])

        # one step function
        res = self.get_agent_reaction(applier_memories,
                                      applier_observations,
                                      **output_flags  # disable dropout here. Only enable in experience replay
                                      )

        # unpack
        applier_actions, applier_new_states, applier_policy = res

        # compile
        applier_fun = theano.function(applier_observations + list(applier_memories.values()),
                                      applier_actions + applier_new_states,
                                      **function_flags)

        # return
        return applier_fun

    def get_zeros_like_memory(self, batch_size=1):
        """
        Returns a list of tensors matching initial agent memory, filled with zeros
        :param batch_size: how many parallel session memories to store
        :return: list of numpy arrays filled with zeros zeros with shape/dtype matching agent memory
        """
        return [np.zeros((batch_size,) + tuple(mem.output_shape[1:]),
                         dtype=get_layer_dtype(mem))
                for mem in self.agent_states]

    def get_all_params(self, **kwargs):
        """calls lasagne.layers.get_all_params(all_agent_layers,**kwargs)"""
        layers = list(self.agent_states) + self.policy + self.action_layers
        return lasagne.layers.get_all_params(layers, **kwargs)

    def get_all_param_values(self, **kwargs):
        """calls lasagne.layers.get_all_param_values(all_agent_layers,**kwargs)"""
        layers = list(self.agent_states) + self.policy + self.action_layers
        return lasagne.layers.get_all_param_values(layers, **kwargs)

    def set_all_param_values(self, values, **kwargs):
        """calls lasagne.layers.set_all_param_values(all_agent_layers,values,**kwargs)"""
        layers = list(self.agent_states) + self.policy + self.action_layers
        return lasagne.layers.set_all_param_values(layers, values, **kwargs)

    def get_sessions(self,
                     environment,
                     session_length=10,
                     batch_size=None,
                     initial_env_states='zeros',
                     initial_observations='zeros',
                     initial_hidden='zeros',
                     experience_replay=False,
                     unroll_scan=True,
                     return_automatic_updates=False,
                     optimize_experience_replay=None,
                     **kwargs
                     ):
        """
        Returns history of agent interaction with environment for given number of turns:
        
        :param environment: an environment to interact with
        :type environment: BaseEnvironment
        :param session_length: how many turns of interaction shall there be for each batch
        :type session_length: int
        :param batch_size: amount of independent sessions [number or symbolic].
            irrelevant if experience_replay=True (will be inferred automatically
            also irrelevant if there's at least one input or if you manually set any initial_*.

        :type batch_size: int or theano.tensor.TensorVariable

        :param experience_replay: whether or not to use experience replay if True, assumes environment to have
            a pre-defined sequence of observations and actions (as env.observations etc.)
            The agent will then observe sequence of observations and will be forced to take recorded actions
            via get_output(...,{action_layer=recorded_action}
            Saves some time by directly using environment.observations (list of sequences) instead of calling
            environment.get_action_results via environment.as_layers(...).
            Note that if this parameter is false, agent will be allowed to pick any actions during experience replay

        :type experience_replay: bool

        :param unroll_scan: whether use theano.scan or lasagne.utils.unroll_scan
        :param return_automatic_updates: whether to append automatic updates to returned tuple (as last element)

        
        :param kwargs: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)
        :type kwargs: several kw flags (flag=value,flag2=value,...)
        
        :param initial_something: layers providing initial values for all variables at 0-th time step
            'zeros' default means filling variables with zeros
            Initial values are NOT included in history sequences

        :param optimize_experience_replay: deprecated, use experience_replay

        :returns: state_seq,observation_seq,hidden_seq,action_seq,policy_seq,
            for environment state, observation, hidden state, chosen actions and agent policy respectively
            each of them having dimensions of [batch_i,seq_i,...]
            time synchronization policy:
            env_states[:,i] was observed as observation[:,i] BASED ON WHICH agent generated his
            policy[:,i], resulting in action[:,i], and also updated his memory from hidden[:,i-1] to hiden[:,i]

        :rtype: tuple of Theano tensors



        """

        if optimize_experience_replay is not None:
            experience_replay = optimize_experience_replay
            warn("optimize_experience_replay is deprecated and will be removed in 1.0.2. Use experience_replay parameter.")

        env = environment

        if experience_replay:
            if not hasattr(env, "observations") or not hasattr(env, "actions"):
                raise ValueError(
                    'if optimize_experience_replay is turned on, one must provide an environment with .observations'
                    'and .actions properties containing observations and actions to replay.')
            if initial_env_states != 'zeros' or initial_observations != 'zeros':
                warn("In experience replay mode, initial env states and initial observations parameters are unused",
                     verbosity_level=2)

            if initial_hidden == 'zeros' and hasattr(env,"preceding_agent_memories"):
                initial_hidden = getattr(env,"preceding_agent_memories")

            # create recurrence
            self.recurrence = self.as_replay_recurrence(environment=environment,
                                                   session_length=session_length,
                                                   initial_hidden=initial_hidden,
                                                   unroll_scan=unroll_scan,
                                                   **kwargs
                                                   )

        else:
            if isinstance(env, SessionPoolEnvironment) or isinstance(env, SessionBatchEnvironment):
                warn("You are using experience replay environment as normal environment. "
                     "This will work, but you can get a free performance boost "
                     "by using passing optimize_experience_replay = True to .get_sessions",
                     verbosity_level=2)

            # create recurrence in active mode (using environment.get_action_results)
            self.recurrence = self.as_recurrence(environment=environment,
                                            session_length=session_length,
                                            batch_size=batch_size,
                                            initial_env_states=initial_env_states,
                                            initial_observations=initial_observations,
                                            initial_hidden=initial_hidden,
                                            unroll_scan=unroll_scan,
                                            **kwargs
                                            )
        state_layers_dict, output_layers = self.recurrence.get_sequence_layers()

        # convert sequence layers into actual theano variables
        theano_expressions = lasagne.layers.get_output(list(state_layers_dict.values()) + list(output_layers))

        n_states = len(state_layers_dict)
        states_list, outputs = theano_expressions[:n_states], theano_expressions[n_states:]

        if experience_replay:
            assert len(states_list) == len(self.agent_states)
            agent_states = states_list
            env_states = [T.arange(session_length)]
            observations = env.observations
        else:
            # sort sequences into categories
            agent_states, env_states, observations = \
                unpack_list(states_list, [len(self.agent_states), len(env.state_shapes), len(env.observation_shapes)])

        policy, actions = unpack_list(outputs, [len(self.policy), len(self.action_layers)])

        agent_states = OrderedDict(list(zip(list(self.agent_states.keys()), agent_states)))

        # if user asked for single value and not one-element list, unpack the list
        if type(environment.state_shapes) not in supported_sequences:
            env_states = env_states[0]
        if self.single_observation:
            observations = observations[0]

        if self.single_action:
            actions = actions[0]
        if self.single_policy:
            policy = policy[0]

        ret_tuple = env_states, observations, agent_states, actions, policy

        if return_automatic_updates:
            ret_tuple += (self.get_automatic_updates(),)

        if unroll_scan and return_automatic_updates:
            warn("return_automatic_updates useful when and only when unroll_scan == False",verbosity_level=2)

        return ret_tuple

    def get_automatic_updates(self,recurrent=True):
        """
        Gets all random state updates that happened inside scan.
        :param recurrent: if True, appends automatic updates from previous layers
        :return: theano.OrderedUpdates with all automatic updates
        """
        return self.recurrence.get_automatic_updates(recurrent=recurrent)

    ###auxilaries for as_recurrence methods
    @staticmethod
    def _check_layer(inner_graph_layer, value, sequence_length=-1, prefix="agent.linker_layer."):
        """
        auxilary function that makes sure v is a lasagne layer with shapes matching inner_graph_layer
        :param value: - lasagne layer or theano tensor or whatever that needs to be a lasagne layer
        :param inner_graph_layer: a layer in agent's inner graph to be matched against v
        :param sequence_length: if None, assumes v is not a sequence of values for inner_graph_layer,
            if not None, assumes it to be a sequence of given sequence_length"""
        shape = tuple(inner_graph_layer.output_shape)
        if sequence_length != -1:
            shape = shape[:1] + (sequence_length,) + shape[1:]

        if not isinstance(value, lasagne.layers.Layer):
            assert value.ndim == len(shape)
            return InputLayer(shape,
                              name=prefix + (inner_graph_layer.name or ""),
                              input_var=value)
        else:
            # v is lasagne.layers.Layer
            assert tuple(value.output_shape)[1:] == shape[1:]
            return value

    @staticmethod
    def _check_init_pairs(layers,initial_values):
        """convert whatever user sends into a list of pairs to an OrderedDict{layer:initializer layer}"""
        if initial_values == "zeros":
            return OrderedDict()
        elif isinstance(initial_values, dict):
            for key,val in initial_values.items():
                assert key in layers
            return check_ordered_dict(initial_values)
        else:
            #type(initialization) is list of inits for each layer
            initial_values = check_list(initial_values)
            assert len(initial_values) == len(layers)
            return OrderedDict([(layer,init) for layer,init in zip(layers, initial_values) if init is not None])




    def as_recurrence(self,
                      environment,
                      session_length=10,
                      batch_size=None,
                      initial_env_states='zeros',
                      initial_observations='zeros',
                      initial_hidden='zeros',
                      recurrence_name='AgentRecurrence',
                      unroll_scan=True,
                      ):

        """
        Returns a Recurrence lasagne layer that contains :

        :param environment: an environment to interact with
        :type environment: BaseEnvironment
        :param session_length: how many turns of interaction shall there be for each batch
        :type session_length: int
        :param batch_size: amount of independent sessions [number or symbolic].
            irrelevant if there's at least one input or if you manually set any initial_*.
        :type batch_size: int or theano.tensor.TensorVariable

        :param initial_something: layers providing initial values for all variables at 0-th time step
            'zeros' default means filling variables with zeros
            Initial values are NOT included in history sequences
            flags: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)

        :param unroll_scan: whether use theano.scan or lasagne.utils.unroll_scan

        :returns: Recurrence instance that returns
              [agent memory states] + [env states] + [env_observations] + [agent policy] + [action_layers outputs]
              all concatenated into one list
        :rtype: agentnet.agent.recurrence.Recurrence

        """
        env = environment

        assert len(check_list(env.observation_shapes)) == len(self.observation_layers)

        # initialize prev states
        prev_env_states = [InputLayer((None,) + check_tuple(shape), name="env.prev_state[{i}]".format(i=i))
                           for i, shape in enumerate(check_list(env.state_shapes))]

        # apply environment changes after agent action
        new_state_outputs, new_observation_outputs = env.as_layers(prev_env_states, self.action_layers)

        # add state changes to memory dict
        all_state_pairs = OrderedDict(chain(self.agent_states.items(),
                                            zip(new_state_outputs, prev_env_states),
                                            zip(new_observation_outputs, self.observation_layers)))

        # compose a dict of {state:initialization}
        state_init_pairs = OrderedDict(chain(
            self._check_init_pairs(new_state_outputs,initial_env_states).items(),
            self._check_init_pairs(new_observation_outputs, initial_observations).items(),
            self._check_init_pairs(list(self.agent_states.keys()), initial_hidden).items()
        ))

        #convert state initializations to layers
        for layer, init in list(state_init_pairs.items()):
            state_init_pairs[layer] = self._check_layer(layer,init)

        # create the recurrence
        recurrence = Recurrence(
            state_variables=all_state_pairs,
            state_init=state_init_pairs,
            tracked_outputs=self.policy + self.action_layers,
            n_steps=session_length,
            batch_size=batch_size,
            delayed_states=new_state_outputs + new_observation_outputs,
            name=recurrence_name,
            unroll_scan=unroll_scan
        )

        return recurrence


    def as_replay_recurrence(self,
                             environment,
                             session_length=10,
                             initial_hidden='zeros',
                             recurrence_name='ReplayRecurrence',
                             unroll_scan=True,
                             ):
        """
        returns a Recurrence lasagne layer that contains.

        :param environment: an environment to interact with
        :type environment: SessionBatchEnvironment or SessionPoolEnvironment
        :param session_length: how many turns of interaction shall there be for each batch
        :type session_length: int

        :param initial_something: layers providing initial values for all variables at 0-th time step
            'zeros' default means filling variables with zeros
            Initial values are NOT included in history sequences
            flags: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)

        :param unroll_scan: whether use theano.scan or lasagne.utils.unroll_scan



        returns:

            an agentnet.agent.recurrence.Recurrence instance that returns
              [agent memory states] + [env states] + [env_observations] [agent policy] + [action_layers outputs]
                  all concatenated into one list

        """
        env = environment

        assert len(check_list(env.observation_shapes)) == len(self.observation_layers)

        initial_hidden = self._check_init_pairs(list(self.agent_states.keys()),initial_hidden)
        #convert state initializations to layers
        for layer, init in list(initial_hidden.items()):
            initial_hidden[layer] = self._check_layer(layer,init)

        # handle observation sequences
        observation_sequences = OrderedDict(zip(self.observation_layers, check_list(env.observations)))
        for layer,sequence in observation_sequences.items():
            observation_sequences[layer] = self._check_layer(layer,sequence,sequence_length=session_length)

        #handle action sequences
        action_sequences = OrderedDict(zip(self.action_layers, check_list(env.actions)))
        for layer, sequence in action_sequences.items():
            action_sequences[layer] = self._check_layer(layer, sequence, sequence_length=session_length)

        all_sequences = OrderedDict(chain(observation_sequences.items(),action_sequences.items()))

        #assert observation and action layers do not intersect
        assert len(all_sequences) == len(observation_sequences) + len(action_sequences)

        # create the recurrence
        recurrence = Recurrence(
            state_variables=self.agent_states,
            state_init=initial_hidden,
            input_sequences=all_sequences,
            tracked_outputs=self.policy + self.action_layers,
            n_steps=session_length,
            name=recurrence_name,
            unroll_scan=unroll_scan
        )
        return recurrence

    def get_agent_reaction(self,
                           prev_states={},
                           current_observations=tuple(),
                           **kwargs):
        """
        Symbolic expression for a one-tick agent reaction
        
        :param prev_states: values for previous states
        :type prev_states: a dict [memory output: prev memory state value]
        
        :param current_observations: agent observations at this step
        :type current_observations: a list of inputs where i-th input corresponds to 
            i-th input slot from self.observations

        :param kwargs: any flag that should be passed to the lasagne network for lasagne.layers.get_output method
        
        :return: a tuple of [actions, new agent states]
            actions: a list of all action layer outputs
            new_states: a list of all new_state values, where i-th element corresponds
            to i-th self.state_variables key
        :rtype: the return type description
        
            
        """

        # standardize prev_states to a dictionary
        if not hasattr(prev_states, 'keys'):
            # if only one layer given, make a single-element list of it
            prev_states = check_list(prev_states)
            prev_states = OrderedDict(zip(self.agent_states.keys(), prev_states))
        else:
            prev_states = check_ordered_dict(prev_states)

        # check that the size matches
        assert len(prev_states) == len(self.agent_states)

        # standardize current observations to a list
        current_observations = check_list(current_observations)
        # check that the size matches
        assert len(current_observations) == len(self.observation_layers)

        # compose input map

        # state input layer: prev state
        prev_states_kv = [(self.agent_states[s], prev_states[s]) for s in
                          list(self.agent_states.keys())]  # prev states

        # observation input layer: observation value
        observation_kv = list(zip(self.observation_layers, current_observations))

        input_map = OrderedDict(prev_states_kv + observation_kv)

        # compose output_list
        output_list = list(self.action_layers) + list(self.agent_states.keys()) + self.policy + self.action_layers

        # call get output
        results = lasagne.layers.get_output(
            layer_or_layers=output_list,
            inputs=input_map,
            **kwargs
        )

        # parse output array
        n_actions = len(self.action_layers)
        n_states = len(self.agent_states)
        n_outputs = len(self.policy + self.action_layers)

        new_actions, new_states, new_outputs = unpack_list(results, [n_actions, n_states, n_outputs])

        return new_actions, new_states, new_outputs


