# AgentNet
A lightweight library to build and train neural networks for reinforcement learning using Theano+Lasagne

## Warning
The library is halfway through development. We maintain a set of runnable examples, but some parts are still missing and others may change significantly with new versions.

__If you wish to get aquainted with the current library state, view some of the ./examples __
https://github.com/BladeCarrier/AgentNet/blob/master/examples/simple%20synthetic%20problem%20-%20default%20setup.ipynb

If you wish to join the development, we would be eager to accept your help. Current priority development anchors are maintained at the bottom of this readme. 

If you wish to contribute your own architecture or experiment, please contact me via github or justheuristic@gmail.com. In fact, please contact me if you have any questions, or ideas, i'd be eager to see them.

##What

The final framework is planned to be built on and fully compatible with awesome Lasagne[6] with some helper functions to facilitate learning.

The main objectives are:
* simple way of tinkering with reinforcement learning architectures
* simple experiment design and reproducibility
* seamless compatibility with Lasagne and Theano



##Why?

[long story short: create a platform to play with *QN architectures without spending months reading code]

The last several years have marked the rediscovery of neural networks applied to Reinforcement Learning domain. The idea has first been introduced in early 90's [0] or even earlier, but was mostly forgotten soon afterwards. 

Years later, these methods were reborn under Deep Learning sauce and popularized by Deepmind [1,2]. Several other researchers have already jumped into the domain with their architectures [3,4] and even dedicated playgrounds [5] to play with them.

The problem is that all these models exist in their own problem setup and implementation bubbles. Simply comparing your new architecture the ones you know requires 
* 10% implementing architecture
* 20% implementing experiment setup
* 70% reimplementing all the other network architectures

This process is not only inefficient, but also very unstable, since a single mistake while implementing 'other' architecture can lead to incorrect results.

So here we are, attempting to build yet another bridge between eager researchers [primarily ourselves so far] and deep reinforcement learning. 

The key objective is to make it easy to build new architectures and test is against others on a number of problems. The easier it is to reproduce the experiment setup, the simpler it is to architect something new and wonderful, the quicker we get to solutions directly applicable to real world problems.

* [0] an dusty old journal issue - https://books.google.ru/books?id=teHhVHk3a54C&printsec=frontcover#v=onepage&q&f=false
* [1] DQN by DeepMind - http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html 
* [2] DQN explained - https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
* [4] deep recurrent  - http://arxiv.org/abs/1507.06527
* [3] attentive DRQN - http://arxiv.org/pdf/1512.01693.pdf
* [5] MazeBaze by Facebook - http://arxiv.org/pdf/1511.07401.pdf



##Current state
The library is currently halfway through creation and there is much to be done yet.

[priority] Component

* Core components
 * [done] Environment
 * [done] Objective
 * [done] Agent architecture
 * Experiment platform
  * [global] Experiment setup zoo
  * [global] Pre-trained model zoo
  * [medium] one-row experiment running
* Layers 
 * Memory 
  * Simple RNN done as Lasagne.layers.DenseLayer
  * [done] One-step GRU memory 
  * [medium] LSTM
  * [medium] Custom memory layer
 * Resolvers
  * [done] Greedy resolver (as BaseResolver) 
  * [done] Epsilon-greedy resolver
  * [low] Softmax resolver
 * Q-evaluator
  * Supports any lasagne architecture 
* Loss functions and training curriculums
 * Can use any theano/lasagne expressions for loss, gradients and updates
 * [high] Training on interesting sessions pool
 * [low] policy gradient training
* Experiment setups
 * [done] Wikicat - guessing person's traits based on wikipedia biographies
 * [high] KSfinder - detecting particle decays in Large Hadron Collider beauty experiment 
* Visualization tools
 * [medium] generic tunable session visualizer 
* Explanatory material
 * [medium] readthedocs pages
 * [global] moar sensible examples
