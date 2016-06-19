# Installing AgentNet

At this point AgentNet is supported for Linux and Mac OS.
We also provide a platform-agnostic Docker container with AgentNet and most popular analytical libraries.

Any assistance with AgentNet installation, as well as your feedback, ideas and bug reports are very welcome.

## Linux and Mac OS Installation
This far the installation was only tested on Ubuntu, yet an experienced user is unlikely to have problems installing it onto other Linux or Mac OS Machine
Currently the minimal dependencies are __bleeding edge__ Theano and Lasagne.
You can find a guide to installing them here
* http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version

If you have both of them, you can install agentnet with
* `[sudo] pip install --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip`

However, we recommend you to install with these commands
```
 git clone https://github.com/justheuristic/AgentNet
 cd AgentNet
 python setup.py install
```


## [Docker container](https://hub.docker.com/r/justheuristic/agentnet/)

We use [Yandex REP](https://github.com/yandex/rep) container to provide data analysis tools (Jupyter, Matplotlib, Pandas, etc)

##### To download/install/run the container, use
 1. install [Docker](http://docs.docker.com/installation/),
 2. make sure `docker` daemon is running (`sudo service docker start`)
 3. make sure no application is using port 1234 (this is the default port that can be changed)
 4. `[sudo] docker run -d -p 1234:8888 justheuristic/agentnet`
 5. Access via localhost:1234 or whatever port you chose
This installation contains an installation of AgentNet, along with latest Theano and Lasagne libraries.



## Windows installation
We recommend running the docker container, using docker-machine (see docker install above).

Technically if you managed to get Lasagne working on Windows, you can follow the Linux instruction.
However, we cannot guarantee that this will work consistently.

A guide on how to install lasagne on windows can be found in [this awesome tutorial](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne).
If you managed to make it running, please contribute your solution (even if it was straightforward).


## Under construction
Our current priorities are
 * Ensuring AgentNet works with python3.*
 * TensorFlow compatibility (via TensorFuse+Lasagne-tf or manually)
 * Writing ReadTheDocs pages


