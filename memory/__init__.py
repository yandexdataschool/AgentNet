import theano
import theano.tensor as T
import lasagne
import numpy as np



class BaseMemory(lasagne.layers.MergeLayer):
    def __init__(self,
                 observation_input,
                 prev_state_input = None,
                 num_units = None,
                 name="YetAnotherMemory",
                 **kwargs):
        """

        instance that
        - generates first (a-priori) agent state
        - determines new agent state given previous agent state and an observation

        parameters: 
            observation_input - a lasagne layer that provides
            float[batch_id, input_id]: input observation at this tick
            -- as an output.
            prev_state_input [optional] - a lasagne layer that generates the previous batch
            of hidden states (in case you wish several layers to handle the same sequence)


        """
        
        
        
        if prev_state_input is None:
            
            batch_size = observation_input.output_shape[0]
            
            if num_units is None:
                raise ValueError, "please provide either num_units or prev_state_input"
            assert type(num_units) is int
            
            prev_state_input = lasagne.layers.InputLayer([batch_size,num_units],
                                                         name=name+".prev_state_input")
            
        elif num_units is None:
            num_units = prev_state_input.shape[1:]

        
        self.num_units = num_units
        self.prev_state_input = prev_state_input
        self.observation_input = observation_input
        incomings = [
            prev_state_input,
            observation_input
        ]
                
        kwargs["name"]=name
        super(BaseMemory, self).__init__(incomings, **kwargs)
        
    def reset(self,batch_size):
        """
        performs this action each time a new session [batch] is loaded
            batch size: size of the new batch
        """
        pass
    def get_first_state(self,batch_size=1):
        """
        computes agent's state before observing any action
        arguments:
            batch_size scalar int: amount of events in current batch 
        returns:
            apriori_state float[batch_id,memory_id]: agent's state
        """
        return T.zeros([batch_size,1])
    
    def get_output_shape_for(self, input_shapes):
        """
        shape of agent's output
        input_shapes: (last_memory_state.shape,observation.shape)
        returns:
               output shape
        """
        last_memory_state_shape,observation_shape = input_shapes
        
        return last_memory_state_shape

    def get_output_for(self, inputs, **kwargs):
        """
        computes agent's memory state after processing observation given last state
        inputs: (last_memory_state,observation)
            last_memory_state float[batch_id, memory_id]: agent's memory state on previous tick
            observation float[batch_id, input_id]: input observation at this tick
        returns:
            memory_state float[batch_id, memory_id]: agent's memory state at this tick
        """
        last_memory_state, observation = inputs
        
        return self.get_first_state(last_memory_state.shape[0])
    def default_input_map(self,last_hidden,observation):
        """
        returns a dict to be used for lasagne.layers.get_output_for
        when predicting agent reaction on observation given his memory state
        """
        return {
            self.prev_state_input: last_hidden,
            self.observation_input:observation,
            }
    
    



from lasagne.layers import Gate,InputLayer
from lasagne import nonlinearities,init
class GRUMemoryLayer(BaseMemory):
    def __init__(self,
                 num_units,
                 observation_input,
                 prev_state_input = None,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 
                 
                 
                 grad_clipping=5.,
                 **kwargs):
        """
        a Gated Recurrent Unit implementation of a memory layer.

        Unlike lasagne.layers.GRUlayer, this layer does not produce the whole time series at a time, 
        but yields it's next state given last state and observation one tick at a time.
        This is done to simplify usage within external loops along with other MDP components.
    
        parameters: 
            - num_units: amount of units in the hidden state. 
                - If you are using prev_state_input, put anything here.
            - observation_input - a lasagne layer that provides
            float[batch_id, input_id]: input observation at this tick
            -- as an output.
            - prev_state_input [optional] - a lasagne layer that generates the previous batch
            of hidden states (in case you wish several layers to handle the same sequence)
            - concatenate_input: if true, appends observation_input of current tick to own activation at this tick

        instance that
        - generates first (a-priori) agent state
        - determines new agent state given previous agent state and an observation|previous input

        """

        
        #default name
        if "name" not in kwargs:
            kwargs["name"] = "YetAnother"+self.__class__.__name__
        
               
        super(GRUMemoryLayer, self).__init__(observation_input,prev_state_input,num_units, **kwargs)
        self.grad_clipping = grad_clipping
        
            
        # Retrieve the dimensionality of the incoming layer
        last_state_shape, observation_shape = self.input_shapes
        
        # Input dimensionality is the output dimensionality of the input layer
        last_num_units = np.prod(last_state_shape[1:])
        num_inputs = np.prod(observation_shape[1:])
        #hidden shapes must match
        assert last_num_units == num_units
        
        
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)
        
        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update, 'hidden_update')
        
        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        self.W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        self.W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        self.b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)



    def get_output_for(self, inputs, **kwargs):
        """
        computes agent's memory state after processing observation given last state
        inputs: (last_memory_state,observation)
            last_memory_state float[batch_id, memory_id]: agent's memory state on previous tick
            observation float[batch_id, input_id]: input observation at this tick
        returns:
            memory_state float[batch_id, memory_id]: agent's memory state at this tick
        """
        last_memory_state, input_data = inputs
        
        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate

        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        
        
        # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
        hid_input = T.dot(last_memory_state, self.W_hid_stacked)

        if self.grad_clipping:
            input_data = theano.gradient.grad_clip(
                input_data, -self.grad_clipping, self.grad_clipping)
            hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

        # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
        input_n = T.dot(input_data, self.W_in_stacked) + self.b_stacked

        # Reset and update gates
        resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
        updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
        resetgate = self.nonlinearity_resetgate(resetgate)
        updategate = self.nonlinearity_updategate(updategate)

        # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
        hidden_update_in = slice_w(input_n, 2)
        hidden_update_hid = slice_w(hid_input, 2)
        hidden_update = hidden_update_in + resetgate*hidden_update_hid
            
        if self.grad_clipping:
            hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
        hidden_update = self.nonlinearity_hid(hidden_update)

        # Compute (1 - u_t)h_{t - 1} + u_t c_t
        hid = (1 - updategate)*last_memory_state + updategate*hidden_update

        
        
        return hid


