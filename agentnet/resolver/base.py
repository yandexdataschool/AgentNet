__doc__="""base (greedy) action resolver"""

import theano.tensor as T

import lasagne
class BaseResolver(lasagne.layers.Layer):
    """
    special Lasagne Layer instance, that:
        - determines actions agent takes given policy (e.g. Qvalues),
    """
    def __init__(self,incoming,*args,**kwargs):
        super(BaseResolver, self).__init__(incoming, **kwargs)

    
    def get_output_for(self,policy,**kwargs):
        """
        picks the action based on Qvalues
        arguments:
            policy float[batch_id, action_id]: policy values for all actions (e.g. Qvalues of probabilities)
        returns:
            actions int[batch_id]: ids of actions picked  
        """
        
        return T.argmax(policy,axis=1)
    def get_output_shape_for(self,input_shape):
        """
        returns output shape [batch_id, 1]
        """
        batch_size = input_shape[0]
        return [batch_size,1]
               