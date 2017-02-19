"""
A few layers that implement simple switching and counting ops
"""
from lasagne.layers import Layer,MergeLayer
import theano.tensor as T
from ..utils.layers import get_layer_dtype


class CounterLayer(Layer):
    """
    A simple counter Layer that increments it's state by 1 each turn and loops each k iterations

    :param prev_counter: previous state of counter
    :type prev_counter: lasagne.layers.Layer, normally InputLayer
    :param k: if not None, resets counter to zero each k timesteps

    :returns: incremented counter
    :rtype: lasagne.layers.Layer

    """
    def __init__(self,prev_counter,k=None,name=None):
        super(CounterLayer,self).__init__(prev_counter,name=name)
        self.k=k
    def get_output_for(self,prev_state,**kwargs):
        return prev_state+1 if self.k is None else (prev_state+1)%self.k
    @property
    def output_dtype(self):
        return 'int32'
    
class SwitchLayer(MergeLayer):
    """a simple layer that implements an 'if-than-else' logic

    :param condition: a layer with [batch_size] boolean conditions (dtype int*)
    :type condition: lasagne.layers.Layer
    :param than_branch: branch that happens if condition != 0 for particular element of a batch
    :type condition: lasagne.layers.Layer
    :param else_branch: branch that happens if condition == 0 for particular element of a batch
    :type condition: lasagne.layers.Layer

    Shapes and dtypes of the two branches must match.

    :returns: a layer where i-th batch sample will take than_branch value if condition, else else_branch value
    :rtype: lasagne.layers.Layer


    """
    def __init__(self,condition,than_branch,else_branch,name=None):
        super(SwitchLayer,self).__init__(incomings=[condition,than_branch,else_branch], name=name)
        
        assert tuple(than_branch.output_shape) == tuple(else_branch.output_shape)
        assert get_layer_dtype(than_branch) == get_layer_dtype(else_branch)
        
        self.output_dtype = get_layer_dtype(than_branch)
        
    def get_output_for(self,inputs,**kwargs):
        """
        :param inputs: a tuple of [condition,than,else]
        """
        
        cond, than_branch, else_branch = inputs
        return T.switch(cond,than_branch,else_branch)
    
    def  get_output_shape_for(self,input_shapes):
        return input_shapes[-1]
    