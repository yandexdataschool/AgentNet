
# Documentation and tutorials

One can find more-or-less structured documentation pages on AgentNet functionality here.

AgentNet also has full embedded documentation, so calling `help(some_function_or_object)` or
pressing shift+tab in IPython yields a description of object/function.

A standard pipeline of AgentNet experiment is shown in following examples:
* [Simple Deep Recurrent Reinforcement Learning setup](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Basic%20tutorial%20on%20Boolearn%20Reasoning%20problem.ipynb) .
    * Most basic demo, if a bit boring. Covers the problem of learning "If X1 than Y1 Else Y2".
    * Uses a single RNN memory and Q-learning algorithm

* [Playing Atari SpaceInvaders with Convolutional NN via OpenAI Gym](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Playing%20Atari%20with%20Deep%20Reinforcement%20Learning%20%28OpenAI%20Gym%29.ipynb) .
  * Step-by-step explanation of what you need to do to recreate DeepMind Atari DQN
  * Written in a generic way, so that adding recurrent memory or changing learning algorithm could be done in a couple of lines



# Demos

__If you wish to get acquainted with the current library state, view some of the ./examples__
* [Playing Atari with Convolutional NN via OpenAI Gym](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Playing%20Atari%20with%20Deep%20Reinforcement%20Learning%20%28OpenAI%20Gym%29.ipynb)
  * Can switch to any visual game thanks to awesome Gym interface
  * Very simplistic, non-recurrent suffering from atari flickering, etc.
* [Deep Recurrent Kung-Fu training with GRUs and actor-critic](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Deep%20Kung-Fu%20with%20GRUs%20and%20A2c%20algorithm%20%28OpenAI%20Gym%29.ipynb)
  * Uses the "Playing atari" example with minor changes
  * Trains via Advantage actor-critic (value+policy-based)
* [Simple Deep Recurrent Reinforcement Learning setup](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Basic%20tutorial%20on%20Boolearn%20Reasoning%20problem.ipynb)
  * Trying to guess the interconnected hidden factors on a synthetic problem setup
* [Stack-augmented GRU generator](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Stack%20RNN%20for%20formal%20sequence%20modelling.ipynb)
  * Reproducing http://arxiv.org/abs/1503.01007 with less code
* [MOAR deep recurrent value-based LR for wikipedia facts guessing](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Advanced%20MDP%20tools%20and%20wikicat.ipynb)
  * Trying to figure a policy on guessing musician attributes (genres, decades active, instruments, etc)
  * Using several hidden layers and 3-step Q-learning
* More to come


AgentNet is under active construction, so expect things to change.
If you wish to join the development, we'd be happy to accept your help.



