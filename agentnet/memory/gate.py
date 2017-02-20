"""
An overly generic layer that implements a custom [stacked] gate to be used in all of your gated recurrences
"""

from functools import reduce
from operator import add
from collections import OrderedDict

import theano.tensor as T
from lasagne import nonlinearities, init
from lasagne.layers import flatten

from ..utils.format import is_layer, check_list, supported_sequences
from ..utils.layers import DictLayer, get_layer_dtype
from ..utils.logging import warn

class GateLayer(DictLayer):
    """
        An overly generic interface for one-step gate, stacked gates or gate applier.
        If several channels are given, stacks them for quicker execution.

        :param gate_controllers: a single layer or a list/tuple of such
            layers that gate depends on (for most RNNs, that's input and previous memory state)

        :param channels: a single layer or integer or a list/tuple of layers/integers
            if a layer, that defines a layer that should be multiplied by the gate output
            if an integer that defines a number of units of a gate -- and these are the units to be returned

        :param gate_nonlinearities: a single function or a list of such(channel-wise),
            - defining nonlinearities for gates on corresponding channels

        :param bias_init: - an initializer or a list (channel-wise) of initializers for bias(b) parameters
            - (None, lasagne.init, theano variable or numpy array)
            - None means no bias
        :param weight_init: - an initializer OR a list of initializers for  (channel-wise)
            - OR a list of lists of initializers (channel, controller)
            - (lasagne.init, theano variable or numpy array)
        """
    def __init__(self,
                 gate_controllers,
                 channels,
                 gate_nonlinearities=nonlinearities.sigmoid,
                 bias_init=init.Constant(),
                 weight_init=init.Normal(),
                 channel_names=None,
                 **kwargs):

        self.channels = check_list(channels)
        self.gate_controllers = check_list(gate_controllers)

        # separate layers from non-layers
        self.channel_layers = list(filter(is_layer, self.channels))
        self.channel_ints = [v for v in self.channels if not is_layer(v)]


        # check channel types
        for chl in self.channels:
            assert is_layer(chl) or (type(chl) == int)

        # check channel names or auto-assign them
        if channel_names is None:
            channel_names = [None] * len(channels)
        assert len(channel_names) == len(channels)

        for i, chl in enumerate(channels):
            if is_layer(chl):
                channel_names[i] = channel_names[i] or chl.name
            channel_names[i] = channel_names[i] or "gate%i" % i

        # flatten gate layers to 2 dimensions
        for i in range(len(self.channel_layers)):
            layer = self.channel_layers[i]
            if type(layer) == int:
                continue

            lname = layer.name or ""
            if len(layer.output_shape) != 2:
                warn("One of the channels (name='%s') has an input dimension of %s and will be flattened." % (
                    lname, layer.output_shape),verbosity_level=2)
                self.channel_layers[i] = flatten(layer,
                                                 outdim=2,
                                                 name=lname)
                assert len(self.channel_layers[i].output_shape) == 2

        # flatten controller layers to 2 dimensions
        for i in range(len(self.gate_controllers)):
            layer = self.gate_controllers[i]
            lname = layer.name or ""
            if len(layer.output_shape) != 2:
                warn("One of the gate controllers (name='%s') has an input dimension of %s and will be flattened." % (
                    lname, layer.output_shape),verbosity_level=2)
                self.gate_controllers[i] = flatten(layer,
                                                   outdim=2,
                                                   name=lname)
                assert len(self.gate_controllers[i].output_shape) == 2

        # initialize merge layer
        incomings = self.channel_layers + self.gate_controllers

        # default name
        output_names = [(kwargs["name"] or "") + channel_name for channel_name in channel_names]

        # determine whether or not user defined a fixed batch size
        batch_sizes = [chl.output_shape[0] for chl in filter(is_layer, self.channels)]
        batch_size = reduce(lambda a,b: a or b, batch_sizes,None)

        
        output_shapes = [ chl.output_shape if is_layer(chl) else (batch_size,chl) for chl in self.channels]
        output_shapes = OrderedDict(zip(output_names,output_shapes))
        
        output_dtypes = [ get_layer_dtype(chl) for chl in self.channels]
        output_dtypes = OrderedDict(zip(output_names,output_dtypes))
        
        
        super(GateLayer, self).__init__(incomings, 
                                        output_shapes=output_shapes,
                                        output_dtypes=output_dtypes,
                                        **kwargs)

        # nonlinearities
        self.gate_nonlinearities = check_list(gate_nonlinearities)
        self.gate_nonlinearities = [(nl if (nl is not None) else (lambda v: v))
                                    for nl in self.gate_nonlinearities]
        # must be either one common nonlinearity or one per channel
        assert len(self.gate_nonlinearities) in (1, len(self.channels))

        if len(self.gate_nonlinearities) == 1:
            self.gate_nonlinearities *= len(self.channels)

        # cast bias init to a list
        bias_init = check_list(bias_init)
        assert len(bias_init) in (1, len(self.channels))
        if len(bias_init) == 1:
            bias_init *= len(self.channels)

        # cast weight init to a list of lists [channel][controller]
        weight_init = check_list(weight_init)
        assert len(weight_init) in (1, len(self.channels))
        if len(weight_init) == 1:
            weight_init *= len(self.channels)

        for i in range(len(self.channels)):
            weight_init[i] = check_list(weight_init[i])
            assert len(weight_init[i]) in (1, len(self.gate_controllers))
            if len(weight_init[i]) == 1:
                weight_init[i] *= len(self.gate_controllers)

        self.gate_b = []  # a list of biases for channels
        self.gate_W = [list() for _ in self.gate_controllers]  # a list of lists of weights [controller][channel]

        for channel,channel_name,b_init,channel_w_inits in zip(self.channels,
                                                               channel_names,
                                                               bias_init,
                                                               weight_init
                                                               ):

            if is_layer(channel):
                channel_n_units = channel.output_shape[1]
            else:
                channel_n_units = channel

            # add bias
            if b_init is not None:
                self.gate_b.append(
                    self.add_param(
                        spec=b_init,
                        shape=(channel_n_units,),
                        name="b_%s" % (channel_name)
                    )
                )
            else:
                self.gate_b.append(T.zeros((channel_n_units,)))

            # add weights
            for ctrl_i, (controller, w_init) in enumerate(zip(self.gate_controllers,
                                                              channel_w_inits
                                                              )):
                ctrl_name = controller.name or "ctrl" + str(ctrl_i)
                # add bias
                self.gate_W[ctrl_i].append(
                    self.add_param(
                        spec=w_init,
                        shape=(controller.output_shape[1], channel_n_units),
                        name="W_%s_%s" % (ctrl_name, channel_name)
                    ))

        # a list where i-th element contains weights[i-th_gate_controller] for all outputs stacked
        self.gate_W_stacked = [T.concatenate(weights, axis=1) for weights in self.gate_W]
        # a list of biases for the respective outputs stacked
        self.gate_b_stacked = T.concatenate(self.gate_b)


    def get_output_for(self, inputs, **kwargs):
        """
            Symbolic  output for the layer.
            parameters:
                inputs - a list of [all layer-defined channels,]+[all_gate_controllers]
                    layer-defined channels are those defined lasagne.layers.Layer and not just a number of inputs
        """
        assert len(inputs) == len(self.channel_layers) + len(self.gate_controllers)

        given_channels, controllers = inputs[:len(self.channel_layers)], inputs[len(self.channel_layers):]

        def slice_w(x_stacked):
            """i slice stacked weights back into components"""
            cumsum = 0
            unpacked = []
            for chl in self.channels:
                n_units = chl.output_shape[1] if is_layer(chl) else chl
                unpacked.append(x_stacked[:, cumsum:cumsum + n_units])
                cumsum += n_units
            return unpacked

        # compute stacked gate->all_outputs contributions

        gate_dot_stacks = [T.dot(ctrl_inp, w_stack) for ctrl_inp, w_stack in zip(controllers, self.gate_W_stacked)]

        # Wx+b stacked for for all channels
        stacked_gate_sums = reduce(add,
                                   gate_dot_stacks,
                                   self.gate_b_stacked[None, :])

        channel_gate_sums = slice_w(stacked_gate_sums)

        # apply nonlinearities
        channel_gates = [nonlinearity(gate_sum) for nonlinearity, gate_sum in zip(self.gate_nonlinearities,
                                                                                  channel_gate_sums)]

        # align channels back into the original order (undo sorting them into ints and layers)
        # and also compute gated outputs on the fly
        gated_channels = []
        next_layer_id = 0  # the index of first unused layer

        for chl_layer_or_int, gate in zip(self.channels, channel_gates):

            if is_layer(chl_layer_or_int):
                chl_value = given_channels[next_layer_id]
                next_layer_id += 1
                gated_channels.append(gate * chl_value)
            else:
                assert type(chl_layer_or_int) == int
                gated_channels.append(gate)
        
        # otherwise return list
        return OrderedDict(zip(self.keys(),gated_channels))

