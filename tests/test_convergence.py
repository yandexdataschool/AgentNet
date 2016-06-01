"""
i test whether learning algorithms converge on boolean reasoning problem
"""
from __future__ import print_function

import numpy as np

import theano
import lasagne
from lasagne.layers import InputLayer, DimshuffleLayer
from lasagne.layers import DropoutLayer, DenseLayer, ExpressionLayer
from lasagne.regularization import regularize_network_params, l2

from agentnet.agent import Agent
from agentnet.resolver import EpsilonGreedyResolver,ProbabilisticResolver
from agentnet.memory.rnn import RNNCell
import agentnet.experiments.boolean_reasoning as experiment
from agentnet.learning import qlearning, sarsa,qlearning_n_step, a2c_n_step
from agentnet.display import Metrics



from nose_parameterized import parameterized

@parameterized([
    (25, qlearning),
    (25, qlearning_n_step),
    (25, sarsa),

])
def test_reasoning_value_based(n_parallel_games=25,
                   algo = qlearning,
                  ):
    """
    :param game_title: name of atari game in Gym
    :param n_parallel_games: how many games we run in parallel
    :param algo: training algorithm to use (module)
    """
    # instantiate an experiment environment with default parameters
    env = experiment.BooleanReasoningEnvironment()

    # hidden neurons
    n_hidden_neurons = 64

    observation_size = (None,) + tuple(env.observation_shapes)

    observation_layer = lasagne.layers.InputLayer(observation_size, name="observation_input")
    prev_state_layer = lasagne.layers.InputLayer([None, n_hidden_neurons], name="prev_state_input")

    # memory layer (this isn't the same as lasagne recurrent units)
    rnn = RNNCell(prev_state_layer, observation_layer, name="rnn0")

    # q_values (estimated using very simple neural network)
    q_values = lasagne.layers.DenseLayer(rnn,
                                         num_units=env.n_actions,
                                         nonlinearity=lasagne.nonlinearities.linear,
                                         name="QEvaluator")

    # resolver uses epsilon - parameter which defines a probability of randomly taken action.
    epsilon = theano.shared(np.float32(0.1), name="e-greedy.epsilon")
    resolver = EpsilonGreedyResolver(q_values, epsilon=epsilon, name="resolver")


    # packing this into agent
    agent = Agent(observation_layer,
                  agent_states={rnn:prev_state_layer},
                  policy_estimators=q_values, 
                  action_layers=resolver)
    
    # Since it's a lasagne network, one can get it's weights, output, etc
    weights = lasagne.layers.get_all_params(resolver,trainable=True)
    weights
    
    
    # produce interaction sequences of length <= 10
    (state_seq,), observation_seq, agent_state, action_seq, qvalues_seq = agent.get_sessions(
        env,
        session_length=10,
        batch_size=env.batch_size,
    )

    hidden_seq = agent_state[rnn]

    # get rewards for all actions
    rewards_seq = env.get_reward_sequences(state_seq, action_seq)

    # get indicator whether session is still active
    is_alive_seq = env.get_whether_alive(observation_seq)
    
    

    # gamma - delayed reward coefficient - what fraction of reward is retained if it is obtained one tick later
    gamma = theano.shared(np.float32(0.99), name='q_learning_gamma')

    squarred_Qerror = algo.get_elementwise_objective(
        qvalues_seq,
        action_seq,
        rewards_seq,
        is_alive_seq,
        gamma_or_gammas=gamma)

    # take sum over steps, average over sessions
    mse_Qloss = squarred_Qerror.sum(axis=1).mean()
    
    
    # impose l2 regularization on network weights
    reg_l2 = regularize_network_params(resolver, l2) * 10**-3

    loss = mse_Qloss + reg_l2
    
    
    # compute weight updates
    updates = lasagne.updates.adadelta(loss, weights, learning_rate=0.1)
    # take sum over steps, average over sessions
    mean_session_reward = rewards_seq.sum(axis=1).mean()

    train_fun = theano.function([], [loss, mean_session_reward], updates=updates)

    compute_mean_session_reward = theano.function([], mean_session_reward)


    score_log = Metrics()
        
    for epoch in range(5000):        

        # update resolver's epsilon (chance of random action instead of optimal one)
        # epsilon decreases over time
        current_epsilon = 0.05 + 0.95 * np.exp(-epoch / 2500.)
        resolver.epsilon.set_value(np.float32(current_epsilon))

        # train
        env.generate_new_data_batch(n_parallel_games)
        loss, avg_reward = train_fun()

        # show current learning progress
        if epoch % 100 == 0:
            print(epoch),

            # estimate reward for epsilon-greedy strategy
            avg_reward_current = compute_mean_session_reward()
            score_log["expected epsilon-greedy reward"][epoch] = avg_reward_current

            # estimating the reward under assumption of greedy strategy
            resolver.epsilon.set_value(0)
            avg_reward_greedy = compute_mean_session_reward()
            score_log["expected greedy reward"][epoch] = avg_reward_greedy
            
            
            if avg_reward_greedy > 2:
                print("converged")
                break
    else:
        print("diverged")
        raise ValueError("Algorithm diverged")
            