__doc__ = """an overly generic layer that implements a custom [stacked] gate to be used in all of your gated recurrences"""

import theano
import theano.tensor as T
import lasagne
import numpy as np

from lasagne.layers import InputLayer,flatten
from lasagne import nonlinearities,init

from ..utils.layers import TupleLayer
from ..utils.format import is_layer,check_list,check_tuple,supported_sequences

from warnings import warn
from operator import add


class GateLayer(TupleLayer):
    def __init__(self,
                 gate_controllers,
                 channels,
                 gate_nonlinearities=nonlinearities.sigmoid,
                 bias_init = init.Constant(),
                 weight_init = init.Normal(),

                 
                 **kwargs):
        """
        An overly generic interface for one-step gate, stacked gates or gate applier.
        If several channels are given, stacks them for quicker execution. 
        
        gate_controllers - a single layer or a list/tuple of such
            layers that gate depends on (for most RNNs, that's input and previous memory state)
        
        channels - a single layer or integer or a list/tuple of layers/integers
            if a layer, that defines a layer that should be multiplied by the gate output
            if an integer - that defines a number of units of a gate -- and these are the units to be returned
            
        gate_nonlinearities - a single function or a list of such(channel-wise), 
            - defining nonlinearities for gates on corresponding channels
        
        bias_init - an initializer or a list (channel-wise) of initializers for bias(b) parameters 
            - (None, lasagne.init, theano variable or numpy array) 
            - None means no bias
        weight init - an initializer OR a list of initializers for  (channel-wise) 
            - OR a list of lists of initializers (channel, controller) 
            - (lasagne.init, theano variable or numpy array) 
        """
        
        #remember if user wants us to only handle a single channel (in that case, return one output instead of list
        self.single_channel = type(channels) not in supported_sequences
        
        self.channels = check_list(channels)
        self.gate_controllers = check_list(gate_controllers)
        
        #check channel types
        for chl in self.channels:
            assert is_layer(chl) or (type(chl) in (int,long))
        
        
        
        #separate layers from non-layers
        self.channel_layers = filter(is_layer, self.channels)
        self.channel_ints = filter(lambda v: not is_layer(v),self.channels)
        
        
        
        #flatten layers to 2 dimensions
        for i in range(len(self.channel_layers)):
            layer = self.channel_layers[i]
            if type(layer) in (int,long):
                continue
                
            lname = layer.name or ""
            if len(layer.output_shape) != 2:
                warn("One of the channels (name='%s') has an input dimension of %s and will be flattened."%(
                        lname, layer.output_shape))
                self.channel_layers[i] = flatten(layer, 
                                                 outdim=2, 
                                                 name=lname)
                assert len(self.channel_layers[i].output_shape)==2
        
        #flatten layers to 2 dimensions
        for i in range(len(self.gate_controllers)):
            layer = self.gate_controllers[i]
            lname = layer.name or ""
            if len(layer.output_shape) != 2:
                warn("One of the gate controllers (name='%s') has an input dimension of %s and will be flattened."%(
                        lname, layer.output_shape))
                self.gate_controllers[i] = flatten(layer, 
                                                   outdim=2, 
                                                   name=lname)
                assert len(self.gate_controllers[i].output_shape)==2
        
        

        #initialize merge layer
        incomings = self.channel_layers + self.gate_controllers
        
        #default name
        kwargs["name"] = kwargs.get("name","YetAnother"+self.__class__.__name__)
        
        
        super(GateLayer, self).__init__(incomings, **kwargs)
        
        
        #nonlinearities
        self.gate_nonlinearities = check_list(gate_nonlinearities)
        self.gate_nonlinearities = [(nl if (nl is not None) else (lambda v:v))
                                       for nl in self.gate_nonlinearities]
        #must be either one common nonlinearity or one per channel
        assert len(self.gate_nonlinearities) in (1, len(self.channels))
        
        if len(self.gate_nonlinearities) == 1:
            self.gate_nonlinearities *= len(self.channels)

        
        #cast bias init to a list 
        bias_init = check_list(bias_init)
        assert len(bias_init) in (1, len(self.channels))
        if len(bias_init) == 1:
            bias_init *= len(self.channels)
            
        #cast weight init to a list of lists [channel][controller]
        weight_init = check_list(weight_init)
        assert len(weight_init) in (1,len(self.channels))
        if len(weight_init) ==1:
            weight_init *= len(self.channels)
            
        for i in range(len(self.channels)):
            weight_init[i] = check_list(weight_init[i])
            assert len(weight_init[i]) in (1,len(self.gate_controllers))
            if len(weight_init[i]) ==1:
                weight_init[i] *= len(self.gate_controllers)
        
        
        
        
        
        
        self.gate_b = [] # a list of biases for channels
        self.gate_W = [list() for i in self.gate_controllers] # a list of lists of weights [controller][channel]
        
        for chl_i,(channel,b_init,channel_w_inits) in enumerate(zip(self.channels,
                                                                     bias_init,
                                                                     weight_init
                                                                    )):
            
            if is_layer(channel):
                channel_name = channel.name or "chl"+str(chl_i)
                channel_n_units = channel.output_shape[1]
            else:
                channel_name = "chl"+str(chl_i)
                channel_n_units = channel
            
            #add bias
            if b_init is not None:
                self.gate_b.append(
                    self.add_param(
                        spec= b_init,
                        shape = (channel_n_units,),
                        name = "b_%s"%(channel_name)
                    )              
                )
            else:
                self.gate_b.append(T.zeros((channel_n_units,)))
                
            
            #add weights
            for ctrl_i,(controller,w_init) in enumerate(zip(self.gate_controllers,
                                                         channel_w_inits
                                                        )):
                ctrl_name = controller.name or "ctrl"+str(ctrl_i)
                #add bias
                self.gate_W[ctrl_i].append(
                    self.add_param(
                        spec= w_init,
                        shape = (controller.output_shape[1], channel_n_units),
                        name = "W_%s_%s"%(ctrl_name,channel_name)
                    ))
            
                
        #a list where i-th element contains weights[i-th_gate_controller] for all outputs stacked
        self.gate_W_stacked = map(lambda weights:T.concatenate(weights,axis=1),self.gate_W)        
        #a list of biases for the respective outputs stacked
        self.gate_b_stacked = T.concatenate(self.gate_b)
        
    
    @property
    def disable_tuple(self):
        return self.single_channel


    def get_output_for(self, inputs, **kwargs):
        """
            Symbolic  output for the layer.
            parameters:
                inputs - a list of [all layer-defined channels,]+[all_gate_controllers]
                    layer-defined channels are those defined lasagne.layers.Layer and not just a number of inputs
        """
        assert len(inputs) == len(self.channel_layers)+len(self.gate_controllers)
        
        given_channels,controllers= inputs[:len(self.channel_layers)],inputs[len(self.channel_layers):]
        
        
        def slice_w(x_stacked):
            """i slice stacked weights back into components"""
            cumsum = 0
            unpacked = []
            for chl in self.channels:
                n_units = chl.output_shape[1] if is_layer(chl) else chl
                unpacked.append(x_stacked[:,cumsum:cumsum+n_units])
                cumsum += n_units
            return unpacked
        
        
        #compute stacked gate->all_outputs contributions
        
        gate_dot_stacks = [T.dot(ctrl_inp,w_stack) for ctrl_inp, w_stack in zip(controllers,self.gate_W_stacked)]

        # Wx+b stacked for for all channels
        stacked_gate_sums = reduce(add,
                              gate_dot_stacks,
                              self.gate_b_stacked[None,:])

        
        channel_gate_sums = slice_w(stacked_gate_sums)
        
        #apply nonlinearities
        channel_gates = [nonlinearity(gate_sum) for nonlinearity,gate_sum in zip(self.gate_nonlinearities,
                                                                                 channel_gate_sums)]
        
        #align channels channels back into the original order (undo sorting them into ints and layers)
        #and also compute gated outputs on the fly
        gated_channels = []
        next_layer_id = 0 #the index of first unused layer
        
        for chl_layer_or_int, gate in zip(self.channels,channel_gates):
            
            if is_layer(chl_layer_or_int):
                chl_value = given_channels[next_layer_id]
                next_layer_id+=1
                gated_channels.append(gate*chl_value)
            else:
                assert type(chl_layer_or_int) in (int,long)
                gated_channels.append(gate)
        #if user only wants one channel, give him that channel instead of a one-item list
        if self.single_channel:
            gated_channels = gated_channels[0]
        
        #otherwise return list
        return gated_channels

    def get_output_shape_for(self, input_shapes):
        """
        a list of shapes of all gates
        parameters: 
            shapes of all incomings [gate controllers and channels (only those defined by layers, not n_units)]
            -- Actually this parameter is unused, so you may provide anything (like None)
        returns:
           list of tuples of shapes of all inputs
        """
        #determine whether or not user defined a fixed batch size
        batch_size = None
        for chl in filter(is_layer, self.channels):
            if chl.output_shape[0] is not None:
                batch_size = chl.output_shape[0]
                break

        channel_shapes = [
            chl.output_shape if is_layer(chl) else (batch_size,chl)
                for chl in self.channels
        ]
        
        #if user wants a single channel, return single channel shape
        if self.single_channel:
            channel_shapes= channel_shapes[0]
        
        #otherwise return a list of shapes
        return channel_shapes


