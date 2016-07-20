# What's what

Here's a brief description of AgentNet structure, intended to help you understand where to find any particular functionality.

Using AgentNet usually involves designing agent architecture, interacting with an environment and training using reinforcement learning techniques.

The first stage (designing agent) requires you to build a neural network (or any Lasagne thing) representing __a single tick__ of agent in a decision process.
This process is described in `agentnet.agent` section below.



## `agentnet.agent`

Core `Agent` abstraction to build your agent around. Lower level `Recurrence` to design custom recurrent architectures.

A tick in agent's life consists of
- seeing observation
- remembering previous memory state [if any]
- committing action
- updating memory state [if any]

Thus, the minimalistic Q-learning agent with no memory should look like this

```
import lasagne,agentnet
#where current observation arrives
observation_layer = lasagne.layers.InputLayer([None,n_observations], name="observation_input")

# q_values layer (estimated using linear model)
q_values = lasagne.layers.DenseLayer(observation_layer,num_units=n_actions,nonlinearity=None)

#a layer that picks action in an epsilon-greedy manner (epsilon = 0 -> greedy manner)
action_resolver = agentnet.resolver.EpsilonGreedyResolver(q_values, epsilon=0, name="resolver")

# packing this into agent
agent = agentnet.agent.Agent(observation_layer,
                             policy_estimators=q_values,
                             action_layers=action_resolver)
```

One can that use `agent.get_sessions(...)` to produce sessions of environment interaction.

To use the trained agent, one can use `agent.get_react_function()` that compiles a one-step function
 that takes observations and previous memory states (if any) and returns actions and updates memory (if any).

To see these methods in action, take a look at some of the examples.
The `agent.get_sessions(...)` is used everywhere and the `agent.get_react_function` is present in any of the Atari examples.

The agent supports arbitrary lasagne network architecture, so this is only the most basic usage example.

### `agentnet.resolver`

The action-picker layers. These layers implement some action picking strategy (e.g. epsilon-greedy) fiven agent policy.

They are generally fed with action Qvalues or probabilities, predicted by the agent.

Unlike most lasagne layers, their output is usually of type int32 (for discrete action space problems), meaning the IDs of actions picked.

Here's a code snippet for epsilon-greedy resolver.

```
#a layer that estimates Qvalues. Note that there's no nonlinearity since Qvalues can be arbitrary.
q_values = lasagne.layers.DenseLayer(<some_other_layers>,
                                     num_units=<n_actions>,
                                     nonlinearity=None)

# epsilon-greedy resolver. Returns a batch of action codes, representing actions picked at this turn.
action_resolver = EpsilonGreedyResolver(q_values, name="action-picker")

#One can change the "epsilon" (probability of picking random action instead of optimal one) like this
action_resolver.epsilon.set_value(0.5)

```


### `agentnet.memory`

Memory layers used to give your agent a recurrent memory (e.g. LSTM) that can be trained via backpropagation through time.

Unlike `lasagne.layers.recurrent`, `agentnet.memory` layers provide a one-step update.
For example, a `GRUCell` layer takes GRU cell from previous tick and input layer(s) and outputs an updated GRU cell state.

To add recurrent memory layers to a one-step network graph, one should
  * Define where does the memory state from last tick go (typically InputLayer of your network).
  * Define a layer that provides the "new state" of the recurrent memory.
  * Connect these two layers when creating Agent (or Recurrence)

Here's an example of adding one RNNCell from the basic tutorial.
```
#layer where current observation goes
observation_layer = lasagne.layers.InputLayer(observation_size, name="observation_input")

#layer for previous memory state (first dot)
prev_state_layer = lasagne.layers.InputLayer([None, n_hidden_neurons], name="prev_state_input")

# new memory state (second dot)
rnn = agentnet.memory.RNNCell(prev_state_layer, observation_layer, name="rnn0")

#<... define Qvalues, resolver, etc>

# packing this into agent
agent = agentnet.agent.Agent(<...all inputs,actions,etc...>,
              agent_states={rnn:prev_state_layer})
```


## `agentnet.environment`

`SessionPoolEnvironment` used to train on recorded sessions from any external environment. Also facilitates experience replay.

When using any external environment (e.g. OpenAI gym), one can go with this kind of environment alone.

If you want to implement Experience Replay-based training, take a closer look to the docs of `agentnet.environment.SessionPoolEnvironment`.

In case you want to implement a custom theano-based environment from scratch to train directly, use `agentnet.environment.BaseEnvironment` to inherit from.

## `agentnet.learning`

A set of reinforcement learning objectives one can use to train `Agent`.

These objectives can be optimized using any optimization tool like `lasagne.updates` (see any of the examples).

## `agentnet.target_network`

This module allows you to define the so called Target Networks - copies of your agent (or some parts of it) that use older
weights from N epochs ago that slowly update towards the agent's current weights.

More details can be found in the module itself.


## utils

This module stores a lot of helper functions, used in other AgentNet submodules.

It also contains a number of generally useful utility functions.

* `agentnet.utils.persistence` - contains `save` and `load` functions used to save all agent params to a file or read them from a previously saved file.
* `agentnet.utils.clone` - a function that allows you to quickly clone a subnetwork or apply some layers to a different input. Useful when sharing params.

That's it for the basics.
To see this architecture in action, we recommend viewing __examples__ section.