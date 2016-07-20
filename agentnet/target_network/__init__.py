"""
Implements the target network techniques in deep reinforcement learning.
In short, the idea is to estimate reference Qvalues not from the current agent state, but
from an earlier snapshot of weights. This is done to decorrelate target and predicted Qvalues/state_values
and increase stability of learning algorithm.

Some notable alterations of this technique:
- Standard approach with older NN snapshot
-- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

- Moving average of weights
-- http://arxiv.org/abs/1509.02971

- Double Q-learning and other clever ways of training with target network
-- http://arxiv.org/pdf/1509.06461.pdf

Here we implement a generic TargetNetwork class that supports both standard and moving
average approaches through "moving_average_alpha" parameter of "load_weights".

"""

from ..utils.clone import clone_network
import lasagne
import theano.tensor as T
import theano
from collections import OrderedDict

class TargetNetwork(object):
    """
    A generic class for target network techniques.
    Works by creating a deep copy of the original network and synchronizing weights through
    "load_weights" method.

    If you just want to duplicate lasagne layers with or without sharing params, use agentnet.utils.clone.clone_network

    :param original_network_outputs: original network outputs to be cloned for target network
    :type original_network_outputs: lasagne.layers.Layer or a list/tuple of such
    :param bottom_layers: the layers that should be shared between networks.
    :type bottom_layers: lasagne.layers.Layer or a list/tuple/dict of such.
    :param share_inputs: if True, all InputLayers will still be shared even if not mentioned in bottom_layers
    :type share_inputs: bool


    :snippet:

    #build network from lasagne.layers
    l_in = InputLayer([None,10])
    l_d0 = DenseLayer(l_in,20)
    l_d1 = DenseLayer(l_d0,30)
    l_d2 = DenseLayer(l_d1,40)
    other_l_d2 = DenseLayer(l_d1,41)

    # TargetNetwork that copies all the layers BUT FOR l_in
    full_clone = TargetNetwork([l_d2,other_l_d2])
    clone_d2, clone_other_d2 = full_clone.output_layers

    # only copy l_d2 and l_d1, keep l_d0 and l_in from original network, do not clone other_l_d2
    partial_clone = TargetNetwork(l_d2,bottom_layers=(l_d0))
    clone_d2 = partial_clone.output_layers

    do_something_with_l_d2_weights()

    #synchronize parameters with original network
    partial_clone.load_weights()

    #OR set clone_params = 0.33*original_params + (1-0.33)*previous_clone_params
    partial_clone.load_weights(0.33)

    """
    def __init__(self,original_network_outputs,bottom_layers=(),share_inputs=True,name="target_net."):
        self.output_layers = clone_network(original_network_outputs,
                                           bottom_layers,
                                           share_inputs=share_inputs,
                                           name_prefix=name)
        self.original_network_outputs = original_network_outputs
        self.bottom_layers = bottom_layers
        self.name = name

        #get all weights that are not shared between networks
        all_clone_params = lasagne.layers.get_all_params(self.output_layers)
        all_original_params = lasagne.layers.get_all_params(self.original_network_outputs)

        #a dictionary {clone param -> original param}
        self.param_dict = OrderedDict({clone_param : original_param
                           for clone_param, original_param in zip(all_clone_params,all_original_params)
                           if clone_param != original_param})

        if len(self.param_dict) ==0:
            raise ValueError("Target network has no loadable. "
                             "Either it consists of non-trainable layers or you messed something up "
                             "(e.g. hand-crafted layers with no automatic params)."
                             "In case you simply want to clone network, use agentnet.utils.clone.clone_network")

        self.load_weights_hard = theano.function([],updates=self.param_dict)

        self.alpha = alpha = T.scalar('moving average alpha',dtype=theano.config.floatX)
        self.param_updates_with_alpha = OrderedDict({ clone_param:  (1-alpha)*clone_param + (alpha)*original_param
                                                     for clone_param,original_param in self.param_dict.items()
                                                    })
        self.load_weights_moving_average = theano.function([alpha],updates=self.param_updates_with_alpha)



    def load_weights(self,moving_average_alpha=1):
        """
        Loads the weights from original network into target network. Should usually be called whenever
        you want to synchronize the target network with the one you train.

        When using moving average approach, one should specify which fraction of new weights is loaded through
        moving_average_alpha param (e.g. moving_average_alpha=0.1)

        :param moving_average_alpha: If 1, just loads the new weights.
            Otherwise target_weights = alpha*original_weights + (1-alpha)*target_weights
        """
        assert 0<=moving_average_alpha<=1

        if moving_average_alpha == 1:
            self.load_weights_hard()
        else:
            self.load_weights_moving_average(moving_average_alpha)

