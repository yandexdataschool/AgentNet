"""
An augmentation that holds K previous states in memory,
used in DeepMind Atari architecture from original article
"""
from lasagne.layers.base import MergeLayer

import theano.tensor as T

from ..utils import insert_dim


class WindowAugmentation(MergeLayer):
    """
    An augmentation that holds K previous items in memory,
    used in DeepMind Atari architecture from original article

    Supports a window of K last values of new_value_input.
    Each time new element is pushed into the window, the oldest one gets out.

    :param new_value_input: a newest item to be stored in the window
    :type new_value_input: lasagne.layers.Layer, shape must be compatible with prev_state_input (see next)
    :param prev_state_input: previous window state of shape [batch, window_length, item_size]
    :type prev_state_input: lasagne.layers.Layer, normally InputLayer


    Shapes of new_value_input and prev_state_input must match:
    if new_value_input has shape (batch, a, b, c)
    than prev_state_input must have shape (batch, window_size, a, b, c)

    where a,b,c stands for arbitrary shapes (e.g. channel, width and height of an image).
    An item can have arbitrary number of dimensions as long as first one is batch_size



    Window shape and K are defined as prev_state_input.output_shape

    The state shape consists of (batch_i, relative_time_inverted, new_value shape)
    So, last inserted value would be at state[:,0],
    pre-last - at state[:,1]
    etc.

    And yes, K = prev_state_input.output_shape[1].
    """

    def __init__(self,
                 new_value_input,
                 prev_state_input,
                 **kwargs):

        # default name
        if "name" not in kwargs:
            kwargs["name"] = "YetAnother" + self.__class__.__name__

        super(WindowAugmentation, self).__init__([new_value_input, prev_state_input], **kwargs)

    def get_output_for(self, inputs, **kwargs):
        """
        pushes new state into window
        """

        # unpacking
        new_state, prev_window = inputs

        # insert time axis
        new_state = insert_dim(new_state, 1)

        assert prev_window.ndim == new_state.ndim

        new_window = T.concatenate([new_state, prev_window[:, :-1]], axis=1)

        return new_window

    def get_output_shape_for(self, input_shapes):
        """
        Returns new window shape = last window shape
        """
        new_state, last_window_shape = input_shapes

        return last_window_shape
