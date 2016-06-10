"""
MDPAgent provides a high-level interface for quick implementation of classic MDP agents with continuous,
discrete or mixed action space, arbitrary recurrent agent memory and decision making policy.

If you are up to something more sophisticated, try agentnet.agent.recurrence.Recurrence,
 which is a lasagne layer for custom recurrent networks.


"""
from __future__ import division, print_function, absolute_import

from collections import OrderedDict
from itertools import chain
from warnings import warn

import lasagne
import theano
from lasagne.layers import InputLayer
from theano import tensor as T

from .recurrence import Recurrence
from ..environment import SessionPoolEnvironment, SessionBatchEnvironment, BaseEnvironment
from ..utils.format import supported_sequences, unpack_list, check_list, check_tuple, check_ordered_dict
from ..deprecated import deprecated


class MDPAgent(object):
    """
        A generic agent within MDP (markov decision process) abstraction.

        :param agent_states: OrderedDict{ memory_output: memory_input}, where
                memory_output: lasagne layer
                    - generates first agent state (before any interaction)
                    - determines new agent state given previous agent state and an observation

                memory_input: lasagne.layers.InputLayer that is used as "previous state" input for memory_output
         :type agent_states: collections.OrderedDict or dict

         :param policy_estimators: whatever determines agent policy
         :type policy_estimators: lasagne.Layer child instance (e.g. Q-values) or a tuple of such instances
                 (e.g. state value + action probabilities for a2c)

         :param action_layers: agent's action(s), or whatever is fed into your environment as agent actions.
         :type action_layers: resolver.BaseResolver child instance or any appropriate layer
                 or a tuple of such, that can be fed into environment to get next state and observation.


        """
    def __init__(self,
                 observation_layers,
                 agent_states,
                 policy_estimators,
                 action_layers,
                 ):


        self.single_action = type(action_layers) not in supported_sequences
        self.single_policy = type(policy_estimators) not in supported_sequences
        self.single_observation = type(observation_layers) not in supported_sequences

        self.observation_layers = check_list(observation_layers)
        self.agent_states = check_ordered_dict(agent_states)
        self.policy = check_list(policy_estimators)
        self.action_layers = check_list(action_layers)
        
    @property
    @deprecated(".agent_states")
    def state_variables(self):
        return self.agent_states


    def get_sessions(self,
                     environment,
                     session_length=10,
                     batch_size=None,
                     initial_env_states='zeros',
                     initial_observations='zeros',
                     initial_hidden='zeros',
                     optimize_experience_replay=False,
                     unroll_scan=True,
                     return_automatic_updates=False,
                     **kwargs
                     ):
        """
        Returns history of agent interaction with environment for given number of turns:
        
        :param environment: an environment to interact with
        :type environment: BaseEnvironment
        :param session_length: how many turns of interaction shall there be for each batch
        :type session_length: int
        :param batch_size: amount of independent sessions [number or symbolic].
            irrelevant if there's at least one input or if you manually set any initial_*.
        :type batch_size: int or theano.tensor.TensorVariable

        initial_<something> - layers providing initial values for all variables at 0-th time step
                'zeros' default means filling variables with zeros
        Initial values are NOT included in history sequences
        
        
        :param optimize_experience_replay: whether or not to optimize for experience replay
           if True, assumes environment to have a pre-defined sequence of observations(as env.observations).
                Saves some time by directly using environment.observations (list of sequences) instead of calling
                    environment.get_action_results via environment.as_layers(...).
                Note that this parameter is not required since experience replay environments have everythin required to
                behave as regular environments
        :type optimize_experience_replay: bool

        :param unroll_scan: whether use theano.scan or lasagne.utils.unroll_scan
        :param return_automatic_updates: whether to append automatic updates to returned tuple (as last element)

        
        :param kwargs: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)
        :type kwargs: several kw flags (flag=value,flag2=value,...)
        
        :returns: state_seq,observation_seq,hidden_seq,action_seq,policy_seq,
            for environment state, observation, hidden state, chosen actions and agent policy respectively
            each of them having dimensions of [batch_i,seq_i,...]
            time synchronization policy:
                env_states[:,i] was observed as observation[:,i] BASED ON WHICH agent generated his
                policy[:,i], resulting in action[:,i], and also updated his memory from hidden[:,i-1] to hiden[:,i]
        :rtype: tuple of Theano tensors

        """
        env = environment

        if optimize_experience_replay:
            if not hasattr(env, "observations"):
                raise ValueError(
                    'if optimize_experience_replay is turned on, one must provide an environment with .observations'
                    'property containing tensor [batch,tick,observation_size] of a list of such (in case of several'
                    'observation layers)')
            if initial_env_states != 'zeros' or initial_observations != 'zeros':
                warn("In experience replay mode, initial env states and initial observations parameters are unused")

            # create recurrence
            self.recurrence = self.as_replay_recurrence(environment=environment,
                                                   session_length=session_length,
                                                   initial_hidden=initial_hidden,
                                                   unroll_scan=unroll_scan,
                                                   **kwargs
                                                   )

        else:
            if isinstance(env, SessionPoolEnvironment) or isinstance(env, SessionBatchEnvironment):
                warn(
                    "You are using experience replay environment as normal environment. "
                    "This will work, but you can get a free performance boost "
                    "by using passing optimize_experience_replay = True to .get_sessions")

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

        if optimize_experience_replay:
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

        if unroll_scan == return_automatic_updates:
            warn("return_automatic_updates useful when and only when unroll_scan == False")

        return ret_tuple

    def get_automatic_updates(self,recurrent=True):
        """
        Gets all random state updates that happened inside scan.
        :param recurrent: if True, appends automatic updates from previous layers
        :return: theano.OrderedUpdates with all automatic updates
        """
        return self.recurrence.get_automatic_updates(recurrent=recurrent)

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

            initial_<something> - layers providing initial values for all variables at 0-th time step
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
        all_state_pairs = list(chain(self.agent_states.items(),
                                     zip(new_state_outputs, prev_env_states),
                                     zip(new_observation_outputs, self.observation_layers)))

        # compose state initialization dict
        state_init_pairs = []
        for initialization, layers in zip(
                [initial_env_states, initial_observations, initial_hidden],
                [new_state_outputs, new_observation_outputs, list(self.agent_states.keys())]
        ):
            if initialization == "zeros":
                continue
            elif isinstance(initialization, dict):
                state_init_pairs += list(initialization.items())
            else:
                initialization = check_list(initialization)
                assert len(initialization) == len(layers)

                for layer, init in zip(layers, initialization):
                    if init is not None:
                        state_init_pairs.append([layer, init])

        # convert all initializations into layers:
        for i in range(len(state_init_pairs)):
            layer, init = state_init_pairs[i]

            # replace theano variables with input layers for them
            if not isinstance(init, lasagne.layers.Layer):
                init_layer = InputLayer(layer.output_shape,
                                        name="env.initial_values_for." + (layer.name or "state"),
                                        input_var=init)
                state_init_pairs[i][1] = init_layer

        # create the recurrence
        recurrence = Recurrence(
            state_variables=OrderedDict(all_state_pairs),
            state_init=OrderedDict(state_init_pairs),
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
                             recurrence_name='AgentRecurrence',
                             unroll_scan=True,
                             ):
        """
        returns a Recurrence lasagne layer that contains.

        :param environment: an environment to interact with
        :type environment: BaseEnvironment
        :param session_length: how many turns of interaction shall there be for each batch
        :type session_length: int

            batch_size - [required parameter] amount of independent sessions [number or symbolic].
                rrelevant if there's at least one input or if you manually set any initial_*.

            initial_<something> - layers providing initial values for all variables at 0-th time step
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

        # state initialization dict item pairs
        if initial_hidden == "zeros":
            initial_hidden = {}
        elif isinstance(initial_hidden, dict):
            for layer in list(initial_hidden.keys()):
                assert layer in list(self.agent_states.keys())
            initial_hidden = check_ordered_dict(initial_hidden)

        else:
            initial_hidden = check_list(initial_hidden)
            assert len(initial_hidden) == len(self.agent_states)

            state_init_pairs = []
            for layer, init in zip(list(self.agent_states.keys()), initial_hidden):
                if init is not None:
                    state_init_pairs.append([layer, init])
            initial_hidden = OrderedDict(state_init_pairs)

        for layer, init in list(initial_hidden.items()):
            # replace theano variables with input layers for them
            if not isinstance(init, lasagne.layers.Layer):
                init_layer = InputLayer(layer.output_shape,
                                        name="agent.initial_values_for." + (layer.name or "state"),
                                        input_var=init)
                initial_hidden[layer] = init_layer

        # handle observation sequences
        observation_sequences = check_list(env.observations)
        for i, obs_seq in enumerate(observation_sequences):

            # replace theano variables with input layers for them
            if not isinstance(obs_seq, lasagne.layers.Layer):
                layer = self.observation_layers[i]
                obs_shape = tuple(layer.output_shape)
                obs_seq_shape = obs_shape[:1] + (session_length,) + obs_shape[1:]

                obs_seq = InputLayer(obs_seq_shape,
                                     name="agent.initial_values_for." + (layer.name or "state"),
                                     input_var=obs_seq)

                observation_sequences[i] = obs_seq
        observation_sequences = OrderedDict(list(zip(self.observation_layers, observation_sequences)))

        # create the recurrence
        recurrence = Recurrence(
            state_variables=self.agent_states,
            state_init=initial_hidden,
            input_sequences=observation_sequences,
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
        
        :param flags: any flag that should be passed to the lasagne network for lasagne.layers.get_output method
        
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

    def get_react_function(self):
        """
        compiles and returns a function that performs one step of agent network

        :returns: a theano function.
            The returned function takes all observation inputs, followed by all agent memories.
            It's outputs are all actions, followed by all new agent memories
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
                                      deterministic=True  # disable dropout here. Only enable in experience replay
                                      )

        # unpack
        applier_actions, applier_new_states, applier_policy = res

        # compile
        applier_fun = theano.function(applier_observations + list(applier_memories.values()),
                                      applier_actions + applier_new_states)

        # return
        return applier_fun
