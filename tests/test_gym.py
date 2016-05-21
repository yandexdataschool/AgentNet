"""
a minimal test case for agentnet and openai gym
"""
from __future__ import print_function

# import matplotlib.pyplot as plt
import numpy as np
import gym
import lasagne
from lasagne.layers import InputLayer, DimshuffleLayer
from agentnet.memory import WindowAugmentation
from agentnet.resolver import EpsilonGreedyResolver
from lasagne.layers import DropoutLayer, DenseLayer, ExpressionLayer
from agentnet.learning import qlearning
import theano
from agentnet.display import Metrics
from lasagne.regularization import regularize_network_params, l2
from agentnet.experiments.openai_gym.pool import GamePool
from agentnet.agent import Agent
from agentnet.environment import SessionPoolEnvironment

__author__ = 'Alex Rogozhnikov'


def test_space_invaders(game_title='SpaceInvaders-v0',
                        n_parallel_games=3,
                        replay_seq_len=2,
                        ):
    """
    :param game_title: name of atari game in Gym
    :param n_parallel_games: how many games we run in parallel
    :param replay_seq_len: how long is one replay session from a batch
    """

    atari = gym.make(game_title)
    atari.reset()

    # Game Parameters
    n_actions = atari.action_space.n
    observation_shape = (None,) + atari.observation_space.shape
    action_names = atari.get_action_meanings()
    del atari
    # ##### Agent observations

    # image observation at current tick goes here
    observation_layer = InputLayer(observation_shape, name="images input")

    # reshape to [batch, color, x, y] to allow for convolutional layers to work correctly
    observation_reshape = DimshuffleLayer(observation_layer, (0, 3, 1, 2))

    # Agent memory states
    window_size = 3

    # prev state input
    prev_window = InputLayer((None, window_size) + tuple(observation_reshape.output_shape[1:]),
                             name="previous window state")

    # our window
    window = WindowAugmentation(observation_reshape,
                                prev_window,
                                name="new window state")

    memory_dict = {window: prev_window}

    # ##### Neural network body
    # you may use any other lasagne layers, including convolutions, batch_norms, maxout, etc

    # pixel-wise maximum over the temporal window (to avoid flickering)
    window_max = ExpressionLayer(window,
                                 lambda a: a.max(axis=1),
                                 output_shape=(None,) + window.output_shape[2:])

    # a simple lasagne network (try replacing with any other lasagne network and see what works best)
    nn = DenseLayer(window_max, num_units=500, name='dense0')
    nn = DropoutLayer(nn, name="dropout", p=0.05)  # will get deterministic during evaluation
    nn = DenseLayer(nn, num_units=300, name='dense1')

    # Agent policy and action picking
    q_eval = DenseLayer(nn,
                        num_units=n_actions,
                        nonlinearity=lasagne.nonlinearities.linear,
                        name="QEvaluator")

    # resolver
    resolver = EpsilonGreedyResolver(q_eval, epsilon=0.1, name="resolver")

    # agent
    agent = Agent(observation_layer,
                  memory_dict,
                  q_eval, resolver)

    # Since it's a single lasagne network, one can get it's weights, output, etc
    weights = lasagne.layers.get_all_params(resolver, trainable=True)

    # Agent step function
    print('compiling react')
    applier_fun = agent.get_react_function()

    # a nice pythonic interface
    def step(observation, prev_memories='zeros', batch_size=n_parallel_games):
        """ returns actions and new states given observation and prev state
        Prev state in default setup should be [prev window,]"""
        # default to zeros
        if prev_memories == 'zeros':
            prev_memories = [np.zeros((batch_size,) + tuple(mem.output_shape[1:]),
                                      dtype='float32')
                             for mem in agent.agent_states]
        res = applier_fun(np.array(observation), *prev_memories)
        action = res[0]
        memories = res[1:]
        return action, memories

    # # Create and manage a pool of atari sessions to play with

    pool = GamePool(game_title, n_parallel_games)

    observation_log, action_log, reward_log, _, _, _ = pool.interact(step, 50)

    print(np.array(action_names)[np.array(action_log)[:3, :5]])

    # # experience replay pool
    # Create an environment with all default parameters
    env = SessionPoolEnvironment(observations=observation_layer,
                                 actions=resolver,
                                 agent_memories=agent.agent_states)

    def update_pool(env, pool, n_steps=100):
        """ a function that creates new sessions and ads them into the pool
        throwing the old ones away entirely for simplicity"""

        preceding_memory_states = list(pool.prev_memory_states)

        # get interaction sessions
        observation_tensor, action_tensor, reward_tensor, _, is_alive_tensor, _ = pool.interact(step, n_steps=n_steps)

        # load them into experience replay environment
        env.load_sessions(observation_tensor, action_tensor, reward_tensor, is_alive_tensor, preceding_memory_states)

    # load first  sessions
    update_pool(env, pool, replay_seq_len)

    # A more sophisticated way of training is to store a large pool of sessions and train on random batches of them.
    # ### Training via experience replay

    # get agent's Q-values obtained via experience replay
    _env_states, _observations, _memories, _imagined_actions, q_values_sequence = agent.get_sessions(
        env,
        session_length=replay_seq_len,
        batch_size=env.batch_size,
        optimize_experience_replay=True,
    )

    # Evaluating loss function

    scaled_reward_seq = env.rewards
    # For SpaceInvaders, however, not scaling rewards is at least working


    elwise_mse_loss = qlearning.get_elementwise_objective(q_values_sequence,
                                                          env.actions[0],
                                                          scaled_reward_seq,
                                                          env.is_alive,
                                                          gamma_or_gammas=0.99, )

    # compute mean over "alive" fragments
    mse_loss = elwise_mse_loss.sum() / env.is_alive.sum()

    # regularize network weights
    reg_l2 = regularize_network_params(resolver, l2) * 10 ** -4

    loss = mse_loss + reg_l2

    # Compute weight updates
    updates = lasagne.updates.adadelta(loss, weights, learning_rate=0.01)

    # mean session reward
    mean_session_reward = env.rewards.sum(axis=1).mean()

    # # Compile train and evaluation functions

    print('compiling')
    train_fun = theano.function([], [loss, mean_session_reward], updates=updates)
    evaluation_fun = theano.function([], [loss, mse_loss, reg_l2, mean_session_reward])
    print("I've compiled!")

    # # Training loop

    for epoch_counter in range(10):
        update_pool(env, pool, replay_seq_len)
        loss, avg_reward = train_fun()
        full_loss, q_loss, l2_penalty, avg_reward_current = evaluation_fun()

        print("epoch %i,loss %.5f, rewards: %.5f " % (
            epoch_counter, full_loss, avg_reward_current))
        print("rec %.3f reg %.3f" % (q_loss, l2_penalty))

