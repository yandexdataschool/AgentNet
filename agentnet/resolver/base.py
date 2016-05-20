"""
Greedy action resolver
"""

import theano.tensor as T

import lasagne


class BaseResolver(lasagne.layers.Layer):
    """
    Special Lasagne Layer instance,
    that determines actions agent takes given policy (e.g. Q-values),
    """

    def __init__(self, incoming, name='BaseResolver'):
        super(BaseResolver, self).__init__(incoming, name=name)

    def get_output_for(self, policy, **kwargs):
        """
        picks the action based on Q-values
        arguments:
            policy float[batch_id, action_id]: policy values for all actions (e.g. Qvalues of probabilities)
        returns:
            actions int[batch_id]: ids of actions picked  
        """

        return T.argmax(policy, axis=1)

    def get_output_shape_for(self, input_shape):
        """
        output shape is [batch_size]
        """
        batch_size = input_shape[0]
        return (batch_size, )

    @property
    def output_dtype(self):
        """
        returns dtype of output tensor. If not implemented, assumes floatX
        """
        return "int32"
