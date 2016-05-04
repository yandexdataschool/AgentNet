# AgentNet

A lightweight library to build and train neural networks for reinforcement learning using Theano+Lasagne

[![Build Status](https://travis-ci.org/yandexdataschool/AgentNet.svg?branch=master)](https://travis-ci.org/yandexdataschool/AgentNet)
[![Gitter](https://badges.gitter.im/yandexdataschool/AgentNet.svg)](https://gitter.im/yandexdataschool/AgentNet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=body_badge)


## Installation

[Here's an installation guide](https://github.com/yandexdataschool/AgentNet/wiki/Installing-AgentNet)

In short, 
 * Manual install
   * Install [bleeding edge theano/lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version)
   * `[sudo] pip install --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip`
 * [Here's the docker container](https://hub.docker.com/r/justheuristic/agentnet/)
   * `[sudo] docker run -d -p 1234:8888 justheuristic/agentnet`
   * Access via localhost:1234 or whatever port you chose


# Documentation and tutorials
AgentNet is using embedded documentation, so calling `help(some_function_or_object)` or pressing shift+tab in Ipython will yield description of what that thing is supposed to do.

A standard pipeline of AgentNet experiment can be found among examples
* [Playing Atari SpaceInvaders with Convolutional NN via OpenAI Gym](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Playing%20Atari%20with%20Deep%20Reinforcement%20Learning%20%28OpenAI%20Gym%29.ipynb)
  * Step-by-step explaination of what you need to do to recreate DeepMind Atari DQN
  * Written in a generic way, so that adding recurrent memory or changing learning algorithm could be done in a couple of lines
* [Simple Deep Recurrent Reinforcement Learning setup](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Basic%20tutorial%20on%20Boolearn%20Reasoning%20problem.ipynb)
  * Most basic demo, if a bit boring. Covers the problem of learning "If X1 than Y1 Else Y2".
  * Only required if SpaceInvaders left you confused.



# Demos
##### If you wish to get acquainted with the current library state, view some of the ./examples
* [Playing Atari with Convolutional NN via OpenAI Gym](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Playing%20Atari%20with%20Deep%20Reinforcement%20Learning%20%28OpenAI%20Gym%29.ipynb)
  * Can switch to any visual game thanks to their awesome interface
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

If you wish to join the development, we would be eager to accept your help. Current priority development anchors are maintained at the bottom of this readme. 



