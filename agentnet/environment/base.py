import theano
from theano import tensor as T

from ..utils.format import check_list,check_tuple
from ..utils.layers import TupleLayer
from lasagne.layers import InputLayer


__doc__ = """Base environment class that all experiments must inherit from (or mimic methods, if they feel cocky)"""

class BaseEnvironment:
    """
    Base for environment layers.
    This is the class you want to inherit from when designing your own custom environments.
    
    To define an environment, one has to describe
        - it's internal state(s),
        - observations it sends to agent
        - actions it accepts from agent
        - environment inner logic

    States, observations and actions are theano tensors (matrices, vectors, etc),
    their shapes should be defined via state_shape, state_dtype, action_shape, etc.

    The default dtypes are floatX for state and observation, int32 for action.
    This suits most of the cases, one can usually use inherited dtypes.

    Finally, one has to implement get_action_results, which converts
    a tuple of (old environment state, agent action) -> (new state, new observation)





    Developer tips:
    [when playing with non-float observations and states]
    if you implemented a new environment, but keep getting a *.grad illegally returned an
    integer-valued variable exception. (Input index *, dtype *), please make sure that any non-float environment
    states are excluded from gradient computation or are cast to floatX.

    To find out which variable causes the problem, find all expressions of the dtype mentioned in
    the expression and then iteratively replace their type with a similar one (like int8 -> uint8 or int32) 
    until the error message dtype changes. Once id does, you have found the cause of the exception.
    """
        
    #shapes

    @property 
    def state_shapes(self):
        """Environment state size: a single shape tuple or several such tuples in a list/tuple """
        raise []
        
    @property 
    def observation_shapes(self):
        """Single observation size: a single shape tuple or several such tuples in a list/tuple """
        #base: mdp with full observability
        return []
    
    @property
    def action_shapes(self):
        """Single agent action size: a single shape tuple or several such tuples in a list/tuple """
        return [1]
    
    
    ##types
    
    @property
    def state_dtypes(self):
        """ types of respective observations"""
        return [theano.config.floatX for obs in check_list(self.observation_shapes)]
    
    @property
    def observation_dtypes(self):
        """type of observations. Most cases require floatX"""
        return [theano.config.floatX for obs in check_list(self.observation_shapes)]

    @property
    def action_dtypes(self):
        """type of an action: a single theano-compatible dtype or several such dtypes in a list/tuple """
        return ["int32" for a in check_list(self.action_shapes)]
    
    ## interaction with agent (one step)
    
    def get_action_results(self,prev_states,actions,**kwargs):
        """
        computes environment state after processing agent's action
        parameters:
            prev_states list(float[batch_id, memory_id0,[memory_id1],...]): environment state on previous tick
            actions list(int[batch_id]): agent action after observing last state
        returns:
            new_states list(float[batch_id, memory_id0,[memory_id1],...]): environment state after processing agent's action
            observations list(float[batch_id,n_agent_inputs]): what agent observes after commiting the last action
        """

        #a dummy update rule where new state is equal to last state
        new_states = last_states
        observations = new_states #mdp with full observability
        return last_states, observations
    

    #---------------------------------------------------
    # Here begin Lasagne Layer compatibility methods
    # Understanding these is not required when implementing your own environments.
    
    
    def as_layers(self, prev_state_layers = None, action_layers = None, **kwargs):
        """Creates a lasagne layer that makes one step environment updates given agent actions.


            parameters:
                prev_state_layers: a layer or a list of layers that provide previous environment state
                        None means create InputLayers automatically
                action_layers: a layer or a list of layers that provide agent's chosen action
                        None means create InputLayers automatically
                name: layer's name
                
            returns:
                [new_states], [observations]: 2 lists of Lasagne layers
                new states - all states in the same order as in self.state_shapes
                observations - all observations in the order of self.observation_shapes
                
        """
        outputs = EnvironmentStepLayer(self,prev_state_layers,action_layers,**kwargs)
        
        pivot = len(self.state_shapes)
        return outputs[:pivot], outputs[pivot:]
    




class EnvironmentStepLayer(TupleLayer):
    
    def __init__(self,
                 environment,
                 prev_state_layers = None,
                 action_layers = None,
                 **kwargs):
        """Creates a lasagne layer that makes one step environment updates given agent actions.


            parameters:
                prev_state_layers: a layer or a list of layers that provide previous environment state
                        None means create InputLayers automatically
                action_layers: a layer or a list of layers that provide agent's chosen action
                        None means create InputLayers automatically
                name: layer's name

            """
        self.env = environment

        #default name
        if "name" not in kwargs:
            kwargs["name"] = self.env.__class__.__name__+".OneStep"

        name = kwargs["name"]

        #create default prev_state and action inputs, if none provided
        if prev_state_layers is None:
            prev_state_layers = [InputLayer((None,)+check_tuple(shape), 
                                            name=name+".prev_state[%i]"%(i))
                                 for i, shape in enumerate(check_list(self.env.state_shapes))]
        
        if action_layers is None:
            action_layers = [InputLayer((None,)+check_tuple(shape), 
                                        name=name+".agent_action_input[%i]"%(i))
                             for i, shape in enumerate(check_list(self.env.action_shapes))]


        self.prev_state_layers = check_list(prev_state_layers)
        self.action_layers = check_list(action_layers)

        incomings = prev_state_layers + action_layers
            

        super(EnvironmentStepLayer,self).__init__(incomings, **kwargs)



    def get_output_for(self,inputs,**kwargs):
        """
        Computes get_action_results as a lasagne layer.
        parameters:
            inputs - a list/tuple that contains previous states and agent actions
                - first, all previous states, that all actions in the original order
        returns
            a list of [all new states, than all observations ] 
        """

        #slice inputs into prev states and actions
        pivot = len(self.prev_state_layers)
        prev_states, actions = inputs[:pivot],inputs[pivot:]

        new_states,observations = self.env.get_action_results(prev_states,actions)

        return check_list(new_states) + check_list(observations)

    def get_output_shape_for(self,input_shapes,**kwargs):
        """Returns a tuple of shapes for every output layer"""
        state_shapes = [ (None,)+check_tuple(shape) for shape in check_list(self.env.state_shapes)]
        observation_shapes =[ (None,)+check_tuple(shape) for shape in check_list(self.env.observation_shapes)]
        
        return state_shapes + observation_shapes
    
    @property
    def output_dtype(self):
        """Returns dtype of output tensors"""
        return check_list(self.env.state_dtypes) + check_list(self.env.observation_dtypes)
