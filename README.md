# AgentNet

A lightweight library to build and train deep reinforcement learning and custom recurrent networks using Theano+Lasagne
[![Build Status](https://travis-ci.org/yandexdataschool/AgentNet.svg?branch=master)](https://travis-ci.org/yandexdataschool/AgentNet)
[![Docs badge](https://readthedocs.org/projects/agentnet/badge)](http://agentnet.readthedocs.org/en/latest/)
[![Gitter](https://badges.gitter.im/yandexdataschool/AgentNet.svg)](https://gitter.im/yandexdataschool/AgentNet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=body_badge)


![img](https://cs.hse.ru/mirror/pubs/share/thumb/150729584:c570x570+185+148:r150x150!)

# What is AgentNet?

<img src='http://s33.postimg.org/ytx63kwcv/whatis_agentnet_png.png' 
     alt='agentnet structure' 
     title='agentnet structure' width=600 />

No time to play games? Let machines do this for you!

AgentNet is a deep reinforcement learning framework, 
which is designed for ease of research and prototyping of Deep Learning models for Markov Decision Processes.

All techno-babble set aside, you can use it to __train your pet neural network to play games!__ [e.g. OpenAI Gym]

We have a full in-and-out support for __Lasagne__ deep learning library, granting you access to all convolutions, maxouts, poolings, dropouts, etc. etc. etc.

__AgentNet__  handles both discrete and continuous control problems and supports hierarchical reinforcement learning [experimental].

List of already implemented reinforcement techniques:
- Q-learning (or deep Q-learning, since we support arbitrary complexity of network)
- N-step Q-learning
- SARSA
- N-step Advantage Actor-Critic (A2c)
- N-step Deterministic Policy Gradient (DPG)

As a side-quest, we also provide a boilerplate to custom long-term memory network architectures (see examples).

## Installation

[Detailed installation guide](https://github.com/yandexdataschool/AgentNet/wiki/Installing-AgentNet)

### Try without installing
* [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/yandexdataschool/dive_to_dqn)
* If you use other similar tools, see [Repo with a dockerfile](https://github.com/yandexdataschool/dive_to_dqn)

### Quick install
* install [bleeding edge lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version)
* [sudo] pip install --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip


### Full install (with examples)

1. Clone this repository: `git clone https://github.com/yandexdataschool/AgentNet.git && cd AgentNet`
2. Install dependencies: `pip install -r requirements.txt`
3. Install library itself: `pip install -e .`

### Docker container

On Windows/OSX install Docker [Kitematic](https://kitematic.com/), 
then simply run `justheuristic/agentnet` container and click on 'web preview'.

On other linux/unix systems: 

1. install [Docker](http://docs.docker.com/installation/),
2. make sure `docker` daemon is running (`sudo service docker start`)
3. make sure no application is using port 1234 (this is the default port that can be changed)
4. `[sudo] docker run -d -p 1234:8888 justheuristic/agentnet`
5. Access from browser via localhost:1234 
  


# Documentation and tutorials

A quick dive-in can be found here:
* Click [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/yandexdataschool/dqn_binder)
* classwork.ipynb = your tutorial
* classwork_solution.ipynb = a fully implemented version with simple CNN for reference


(incomplete) Documentation pages can be found [here](http://agentnet.readthedocs.io/en/latest/).

AgentNet also has full embedded documentation, so calling `help(some_function_or_object)` or 
pressing shift+tab in IPython yields a description of object/function.

A standard pipeline of AgentNet experiment is shown in following examples:
* [Simple Deep Recurrent Reinforcement Learning setup](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Basic%20tutorial%20on%20Boolearn%20Reasoning%20problem.ipynb)
  * Most basic demo, if a bit boring. Covers the problem of learning "If X1 than Y1 Else Y2".
  * Uses a single RNN memory and Q-learning algorithm

* [Playing Atari SpaceInvaders with Convolutional NN via OpenAI Gym](https://github.com/yandexdataschool/AgentNet/blob/master/examples/Playing%20Atari%20with%20Deep%20Reinforcement%20Learning%20%28OpenAI%20Gym%29.ipynb)
  * Step-by-step explanation of what you need to do to recreate DeepMind Atari DQN
  * Written in a generic way, so that adding recurrent memory or changing learning algorithm could be done in a couple of lines



# Advanced examples

##### If you wish to get acquainted with the current library state, view some of the ./examples
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

