import lasagne                 
from lasagne.utils import unroll_scan
from lasagne.layers import InputLayer
from ..utils.layers import TupleLayer, get_layer_dtype


import theano
from theano import tensor as T

import numpy as np
from collections import OrderedDict
from warnings import warn


from ..utils.format import check_list,check_ordict,unpack_list, supported_sequences
from ..utils import insert_dim


class Recurrence(TupleLayer):
    def __init__(self,
                 input_nonsequences = OrderedDict(),
                 input_sequences = OrderedDict(),
                 tracked_outputs = [],
                 state_variables = OrderedDict(),
                 state_init = 'zeros',
                 n_steps = 10,
                 batch_size = None,
                 delayed_states = [],
                 strict = True,
                 **kwargs
                ):
        """
        A generic recurrent unit that works with a custom graph
        Takes:
        
            input_nonsequences: inputs that are same at each time tick.
                Technically it's a dictionary that maps InputLayer from one-step graph
                to layers from the outer graph.
                
            input_sequences: layers that represent time sequences, fed into graph tick by tick.
                This has to be a dict (one-step input -> sequence layer).
                All such sequences are iterated over FIRST AXIS (axis=1),
                since we consider their shape to be [batch, time, whatever_else...]
            
            tracked_outputs: any layer from the one-state graph which outputs should be
                recorded at every time tick.
                Note that all state_variables are tracked separately, so their inclusion is not needed.


            state_variables: a dictionary that maps next state variables to their respective previous state
                keys (new states) must be lasagne layers and values (previous states) must be InputLayers
                
                Note that state dtype is defined thus:
                 - if state key layer has output_dtype, than that type is used for the entire state
                 - otherwise, theano.config.floatX is used
            
            state_init: what are the default values for state_variables. In other words, what is
                prev_state for the first iteration. By default it's T.zeros of the appropriate shape.
                Can be a dict mapping state OUTPUTS to their initializers.
                    - if so, any states not mentioned in it will be considered zeros
                Can be a list of initializer layers for states in the order of dict.items()
                    - if so, it's length must match len(state_variables)
                    
                
            n_steps - how many time steps will the recurrence unroll for
            
            batch_size - if the process has no inputs, this expression (int or theano scalar),
                this variable defines the batch size
                
            delayed_states - any states mentioned in this list will be shifted 1 turn backwards 
                    - from init to n_steps -1. They will be padded with their initial values
                    This is intended to allow flipping the recurrence graph to synchronize corresponding values. 
                    E.g. for MDP, if environment reaction follows agent action,  synchronize observations with actions
                    [at i-th turn agent sees i-th observation, than chooses i-th action and gets i-th reward]
            
            strict - whether to assert that all inner graph input layers are registered for the recurrence
                as inputs or prev states and all inputs/prev states are actually needed to compute next states/outputs.
                NOT the same as theano.scan(strict=True).
                
            **kwargs - layer name and other lasagne layer attributes you want that suit the MergeLayer
        
        Outputs:
            returns a tuple of sequences with shape [batch,tick, ...]
                - state variable sequences in order of dict.items()
                - tracked_outputs in given order
            
            WARNING! can not be used further as an atomic lasagne layer.
            Instead, consider calling .get_sequences() or unpacking it
            
            state_sequence_layers, output_sequence_layers = Recurrence(...).get_sequences()
            (see .get_sequences help for more info)
            
            OR
            
            state_seq_layer, ... , output1_seq_layer, output2_seq_layer, ... = Recurrence(...)
                        
            
        """
        
        #default name
        if "name" not in kwargs:
            kwargs["name"] = "YetAnother"+self.__class__.__name__

            

        self.n_steps = n_steps

        
        self.input_nonsequences = check_ordict(input_nonsequences)
        self.input_sequences = check_ordict(input_sequences)

        
        self.tracked_outputs = check_list(tracked_outputs)
         
        
        self.state_variables = check_ordict(state_variables) 
        if type(state_variables) is not OrderedDict:
            if len(self.state_variables) >1:
                warn("State_variables recommended type is OrderedDict.\n"\
                     "Otherwise, order of agent state outputs from get_sessions and get_agent_reaction methods\n"\
                     "may depend on python configuration.\n Current order is:"+ str(list(self.state_variables.keys()))+"\n"\
                     "You may find OrderedDict in standard collections module: from collections import OrderedDict")
        
        self.delayed_states = delayed_states
        
        #initial states
        
        #convert zeros to empty dict
        if state_init == "zeros":
            state_init = OrderedDict()
        #convert list to dict
        elif state_init in supported_sequences:
            assert len(state_init) == len(self.state_variables)
            state_init = OrderedDict(list(zip(list(self.state_variables.keys()), state_init)))
            
        #cast to dict and save
        self.state_init = check_ordict(state_init)
        
        
        #init base class
        incomings = list(self.state_init.values()) +\
                    list(self.input_nonsequences.values()) +\
                    list(self.input_sequences.values())
        
                
        #if we have no inputs or inits, make sure batch_size is specified
        if len(incomings) ==0:
            assert batch_size is not None
        self.batch_size = batch_size

                
        super(Recurrence,self).__init__(incomings, **kwargs)
        
        
        ### assertions and only assertions. From now this function only asserts stuff.
        
        if strict:        
            #verifying graph topology (assertions)

            all_inputs = list(self.state_variables.values()) +\
                         list(self.state_init.keys()) +\
                         list(self.input_nonsequences.keys()) +\
                         list(self.input_sequences.keys())
                        
            #all recurrent graph inputs and prev_states are unique (no input/prev_state is used more than once)
            assert len(all_inputs) == len(set(all_inputs))

            #all state_init correspond to defined state variables
            for state_out in list(self.state_init.keys()):
                assert state_out in list(self.state_variables.keys())

            #all new_state+output dependencies (input layers) lie inside one-step recurrence
            all_outputs = list(self.state_variables.keys()) + self.tracked_outputs
            all_inputs = set(all_inputs)
            
            for layer in lasagne.layers.get_all_layers(all_outputs):
                if type(layer) is InputLayer:
                    if layer not in all_inputs:
                        raise ValueError("One of your network dependencies (%s) is not mentioned "\
                              "as a Recurrence inputs"%(str(layer.name)))
        
        #verifying shapes (assertions)
        
        nonseq_pairs = list(self.state_variables.items()) +\
                       list(self.state_init.items()) +\
                       list(self.input_nonsequences.items())
                
        for layer_out, layer_in in nonseq_pairs:
            assert tuple(layer_in.output_shape) == tuple(layer_out.output_shape)
        
            
        for seq_onestep, seq_input in list(self.input_sequences.items()):
            seq_shape = tuple(seq_input.output_shape)
            step_shape = seq_shape[:1] + seq_shape[2:]
            assert tuple(seq_onestep.output_shape) == step_shape
            
            seq_len = seq_shape[1]
            assert seq_len is None or seq_len>=n_steps
            if seq_len is None:
                warn("You are giving Recurrence an input sequence of undefined length (None).\n"\
                     "Make sure it is always above {}(n_steps) you specified for recurrence".format(n_steps))
            
            
            
    def get_params(self,**kwargs):
        """returns all params. If include_recurrence is set ot True, includes recurrent params from one-step network"""
        
        params = super(Recurrence,self).get_params(**kwargs)
        
        #include inner recurrence params
        outputs = list(self.state_variables.keys()) + self.tracked_outputs
        inner_params = lasagne.layers.get_all_params(outputs,**kwargs)
        params += inner_params
        
        return params
        

    def get_output_for(self,inputs,recurrence_flags = {},**kwargs):
        """
        returns history of agent interaction with environment for given number of turns.
        
        parameters:
            inputs - [state init]  + [input_nonsequences] + [input_sequences]
                Each part is a list of theano expressions for layers in the order they were
                provided when creating this layer.
            recurrence_flags - a set of flags to be passed to the one step agent (anything that lasagne supports)
                e.g. {deterministic=True}
        returns:
            [state_sequences] + [output sequences] - a list of all states and all outputs sequences
            Shape of each such sequence is [batch, tick, shape_of_one_state_or_output...]
        """
        
        #set batch size
        if len(inputs) != 0:
            batch_size = inputs[0].shape[0]
        else:
            batch_size = self.batch_size
        
        
        #parse inputs
        input_layers = list(self.input_nonsequences.keys()) + list(self.input_sequences.keys())

        n_states = len(self.state_variables)
        n_state_inits = len(self.state_init)
        n_input_nonseq = len(self.input_nonsequences)
        n_input_seq = len(self.input_sequences)
        n_outputs = len(self.tracked_outputs)
        
        initial_states, nonsequences, sequences = unpack_list(inputs, n_state_inits,n_input_nonseq,n_input_seq)
        
        
        # reshape sequences from [batch, time, ...] to [time,batch,...] to fit scan
        sequences = [seq.swapaxes(1,0) for seq in sequences]
        
        
        # create outputs_info for scan
        initial_states = OrderedDict(list(zip(self.state_init,initial_states)))
        
        def get_initial_state(state_out_layer):
            """ pick dedicated initial state or create zeros of appropriate shape and dtype"""
            #if we have a dedicated init
            if state_out_layer in initial_states:
                
                #use it
                initial_state = initial_states[state_out_layer]
                                    
            #otherwise initialize with zeros
            else:
                initial_state = T.zeros((batch_size,)+tuple(state_out_layer.output_shape[1:]), 
                                    dtype = get_layer_dtype(state_out_layer))
            return initial_state
        
        
        initial_state_variables = list(map(get_initial_state, self.state_variables))
        
        outputs_info = initial_state_variables + [None]*len(self.tracked_outputs)
        
        #recurrent step function
        def step(*args):

            sequence_slices,prev_states,prev_outputs,nonsequences = unpack_list(args,
                                                                               n_input_seq,
                                                                               n_states, 
                                                                               n_outputs,
                                                                               n_input_nonseq,
                                                                               )

            #make dicts of prev_states and inputs
            prev_states_dict = OrderedDict(list(zip(list(self.state_variables.keys()),prev_states)))
            
            input_layers = list(self.input_nonsequences.keys()) + list(self.input_sequences.keys())
            
            assert len(input_layers)== len(nonsequences+sequence_slices)
            
            inputs_dict = OrderedDict(list(zip(input_layers,nonsequences+sequence_slices)))
            
            #call one step recurrence
            new_states, new_outputs = self.get_one_step(prev_states_dict, inputs_dict,**recurrence_flags)
            
            return new_states + new_outputs
        
        
        #call scan itself (unroll it to avoid randomness issues that may have ceased to exist already)
        
        history = unroll_scan(step,
            sequences = sequences,
            outputs_info = outputs_info,
            non_sequences = nonsequences,
            n_steps = self.n_steps
        )


        #from [time,batch,...] to [batch,time,...]
        history = [ (var.swapaxes(1,0) if var.ndim >1 else var) for var in history]
        
        
        state_seqs, output_seqs = unpack_list(history, 
                                            n_states,
                                            n_outputs)
        
        #handle delayed_states
        #selectively shift state sequences by 1 tick into the past, padding with their initializers 
        for i in range(len(state_seqs)):
            if list(self.state_variables.keys())[i] in self.delayed_states:
                
                state_seq = state_seqs[i]
                state_init = initial_state_variables[i]
                state_seq = T.concatenate([
                                    insert_dim(state_init,1),
                                    state_seq[:,:-1]
                                    ],
                                    axis = 1)
                
                
                state_seqs[i] = state_seq    
        
        
        
        
        return state_seqs+ output_seqs
                        
                        
    @property
    def output_shapes(self):
        """returns shapes of each respective output"""
        shapes = [ tuple(layer.output_shape) for layer in list(self.state_variables.keys()) + self.tracked_outputs]
        return [ shape[:1]+(self.n_steps,)+shape[1:] for shape in shapes]
        
                            
                            
    def get_one_step(self,prev_states={},current_inputs={},**flags):
        """
        Applies one-step recurrence.
        parameters:
            prev_states: a dict {memory output: prev state} 
                    or a list of theano expressions for each prev state
            
            current_inputs: a dictionary of inputs that maps {input layers -> theano expressions for them},
                        Alternatively, it can be a list where i-th input corresponds to 
                        i-th input slot from concatenated sequences and nonsequences 
                        self.input_nonsequences.keys() + self.input_sequences.keys()
            
            flags: any flag that should be passed to the lasagne network for lasagne.layers.get_output method
            
            
            returns:
                new_states: a list of all new_state values, where i-th element corresponds
                        to i-th self.state_variables key
                new_outputs: a list of all outputs  where i-th element corresponds
                        to i-th self.tracked_outputs key
            
        """


            
                 
        #standartize prev_states to a dicitonary
        if not isinstance(prev_states,dict):
            #if only one layer given, make a single-element list of it
            prev_states = check_list(prev_states)
            prev_states = OrderedDict(list(zip(list(self.state_variables.keys()),prev_states)))
        else:
            prev_states = check_ordict(prev_states)
        
        assert len(prev_states) == len(self.state_variables)
        
        #input map
        ##prev state input layer: prev state expression
        prev_states_kv = [(self.state_variables[s],prev_states[s]) 
                          for s in list(self.state_variables.keys())] #prev states


        
        #prepare both sequence and nonsequence inputs in one dict
        input_layers = list(self.input_nonsequences.keys()) + list(self.input_sequences.keys())

            
        #standartize current_inputs to a dictionary

        if not isinstance(current_inputs,dict):
            
            #if only one layer given, make a single-element list of it
            current_inputs = check_list(current_inputs)
            current_inputs = OrderedDict(list(zip(input_layers,current_inputs)))
        else:
            current_inputs = check_ordict(current_inputs)
            
        
        assert len(current_inputs) == len(input_layers)

        #second half of input map
        ##external input layer: input expression
        inputs_kv = list(current_inputs.items())

        
        #compose input map
        input_map = OrderedDict(prev_states_kv + inputs_kv)
        
        #compose output_list
        output_list = list(self.state_variables.keys()) + self.tracked_outputs

        
        #call get output
        results = lasagne.layers.get_output(
            layer_or_layers=output_list,
            inputs= input_map,
            **flags
          )
        
        #parse output array
        n_states = len(self.state_variables)
        n_outputs = len(self.tracked_outputs)
        
        new_states,new_outputs = unpack_list(results,n_states,n_outputs)
        
        return new_states,new_outputs

    
    
    def get_sequence_layers(self):
        """
        returns history of agent interaction with environment for given number of turns.
            [state_sequences] , [output sequences] - a list of all state sequences and  a list of all output sequences
            Shape of each such sequence is [batch, tick, shape_of_one_state_or_output...]
        """

        outputs = list(self)
        
        n_states = len(self.state_variables)
        
        state_dict = OrderedDict(list(zip( self.state_variables,outputs[:n_states])))
        
        return state_dict, outputs[n_states:]
        
    
