__doc__="""base (greedy) action resolver"""

import theano.tensor as T

import lasagne
class BaseResolver(lasagne.layers.Layer):
    """
    special Lasagne Layer instance, that:
        - determines Q-values for all actions given current agent state and current observation,
    """
    def __init__(self,incoming,*args,**kwargs):
        super(BaseResolver, self).__init__(incoming, **kwargs)

    
    def get_output_for(self,Qvalues,**kwargs):
        """
        picks the action based on Qvalues
        arguments:
            Qvalues float[batch_id, action_id]: Qvalues for all actions
        returns:
            actions int[batch_id]: ids of actions picked  
        """
        
        return T.argmax(Qvalues,axis=1)
    def get_output_shape_for(self,input_shape):
        """
        returns output shape [batch_id, 1]
        """
        batch_size = input_shape[0]
        return [batch_size,1]
               