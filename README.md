# AgentNet
A lightweight library to build and train neural networks for reinforcement learning using Theano+Lasagne

## Warning
The library is in active development. We maintain a set of runnable examples and a fixed interface, but it may still change once in a while.

## Linux and Mac OS Installation
This far the instalation was only tested on Ubuntu, yet an experienced user is unlikely to have problems installing it onto other Linux or Mac OS Machine
Currently the minimal dependencies are bleeding edge Theano and Lasagne.
You can find a guide to installing them here 
* http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version

If you have both of them, you can install agentnet with these commands
```
 git clone https://github.com/justheuristic/AgentNet
 cd AgentNet
 python setup.py install
``` 

## Windows installation
Technically if you managed to get Lasagne working on Windows, you can follow the Linux instruction.
However, we cannot guarantee that this will work consistently.


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

If you wish to contribute your own architecture or experiment, please contact me via github or justheuristic@gmail.com. In fact, please contact me if you have any questions, or ideas, i'd be eager to see them.

## What?

The final framework is planned to be built on and fully compatible with awesome Lasagne[6] with some helper functions to facilitate learning.

The main objectives are:
* easy way of tinkering with reinforcement learning architectures
* just as simple prototyping of Attention and Long Term Memory architectures
* ease of experiment conduction and reproducibility
* full integration with Lasagne and Theano



## Why?

[long story short: create a platform to play with *QN, attentive and LTM architectures without spending months reading code]

[short story long:

The last several years have marked the rediscovery of neural networks applied to Reinforcement Learning domain. The idea has first been introduced in early 90's [0] or even earlier, but was mostly forgotten soon afterwards. 

Years later, these methods were reborn under Deep Learning sauce and popularized by Deepmind [1,2]. Several other researchers have already jumped into the domain with their architectures [3,4] and even dedicated playgrounds [5] to play with them.

The problem is that all these models exist in their own problem setup and implementation bubbles. Simply comparing your new architecture the ones you know requires 
* 10% implementing architecture
* 20% implementing experiment setup
* 70% reimplementing all the other network architectures

This process is not only inefficient, but also very unstable, since a single mistake while implementing 'other' architecture can lead to incorrect results.

So here we are, attempting to build yet another bridge between eager researchers [primarily ourselves so far] and deep reinforcement learning. 

The key objective is to make it easy to build new architectures and test is against others on a number of problems. The easier it is to reproduce the experiment setup, the simpler it is to architect something new and wonderful, the quicker we get to solutions directly applicable to real world problems.

]

* [0] an dusty old journal issue - https://books.google.ru/books?id=teHhVHk3a54C&printsec=frontcover#v=onepage&q&f=false
* [1] DQN by DeepMind - http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html 
* [2] DQN explained - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
* [4] deep recurrent  - http://arxiv.org/abs/1507.06527
* [3] attentive DRQN - http://arxiv.org/pdf/1512.01693.pdf
* [5] MazeBaze by Facebook - http://arxiv.org/pdf/1511.07401.pdf



## Current state & priorities
The library is currently in active development and there is much to be done yet.

[priority] Component; no priority means "done"

* Core components
 * Environment
 * Objective
 * Agent architecture
   * MDP (RL) agent
   * Generator
   * Fully customizable agent
 * Experiment platform
   * [high] Experiment setup zoo
   * [medium] Pre-trained model zoo
   * [medium] quick experiment running (
         * experiment is defined as (environment, objective function, NN architecture, training algorithm)

* Layers 
 * Memory 
    * Simple RNN done as Lasagne.layers.DenseLayer
    * One-step GRU memory 
    * [half-done, medium] Custom LSTM-like constructor
    * Stack Augmentation
    * [low] List augmentation
    * [low] Neural Turing Machine controller
 * Resolvers
    * Greedy resolver (as BaseResolver) 
    * Epsilon-greedy resolver
    * Probablistic resolver

* Learning objectives algorithms
  * Q-learning
  * SARSA
  * k-step learning
  * k-step Advantage Actor-critic methods
  * Can use any theano/lasagne expressions for loss, gradients and updates
  * Experience replay pool

* Experiment setups
  * boolean reasoning - basic "tutorial" experiment about learning to exploit variable dependencies
  * Wikicat - guessing person's traits based on wikipedia biographies
  * [half-done] 2048 in the browser - playing 2048 using Selenium only
  * [high] openAI gym training/evaluation api and demos
  * [medium] KSfinder - detecting particle decays in Large Hadron Collider beauty experiment 

* Visualization tools
  * basic monitoring tools 
  * [medium] generic tunable session visualizer

* Explanatory material
 * [medium] readthedocs pages
 * [global] MOAR sensible examples
 * [medium] report on basic research (optimizer comparison, training algorihtm comparison, layers, etc)
