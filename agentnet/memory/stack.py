"""
A simple stack augmentation for RNN http://arxiv.org/abs/1503.01007
"""
from lasagne.layers.base import MergeLayer

import theano.tensor as T

from ..utils import insert_dim


class StackAugmentation(MergeLayer):
    """
    A special kind of memory augmentation that implements
    end-to-end diferentiable stack in accordance to this paper: http://arxiv.org/abs/1503.01007

    :param observation_input: an item that can be pushed into the stack (e.g. RNN state)
    :type observation_input: lasagne.layers.Layer
    :param prev_state_input: revious stack state of shape [batch,stack depth, stack item size]
    :type prev_state_input: lasagne.layers.Layer (usually InputLayer)
    :param controls_layer: a layer with 3 channels: PUSH_OP, POP_OP and NO_OP accordingly (must sum to 1)
    :type controls_layer: lasagne.layers.layer (usually DenseLayer with softmax nonlinearity)

    A simple snippet that runs that augmentation from the Stack RNN example
    stack_width = 3
    stack_depth = 50
    # previous stack goes here
    prev_stack_layer = InputLayer((None,stack_depth,stack_width))
    # Stack controls - push, pop and no-op
    stack_controls_layer = DenseLayer(<rnn>,3, nonlinearity=lasagne.nonlinearities.softmax,)
    # stack input
    stack_input_layer = DenseLayer(<rnn>,stack_width)
    #new stack state
    next_stack = StackAugmentation(stack_input_layer,prev_stack_layer,stack_controls_layer)

    """
    def __init__(self,
                 observation_input,
                 prev_state_input,
                 controls_layer,
                 **kwargs):
        # default name
        if "name" not in kwargs:
            kwargs["name"] = "YetAnother" + self.__class__.__name__

        super(StackAugmentation, self).__init__([observation_input, prev_state_input, controls_layer], **kwargs)

    def get_output_for(self, inputs, **kwargs):
        """
            Updates stack given input, stack controls and output in the inputs array
        """

        # unpack inputs
        input_val, prev_stack, controls = inputs
        assert input_val.ndim == 2

        # cast shapes
        controls = controls.reshape([-1, 3, 1, 1])
        input_val = insert_dim(input_val, 1)
        zeros_at_the_top = insert_dim(T.zeros_like(prev_stack[:, 0]), 1)

        # unpack controls
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]

        # a version of stack that is pushed down (push)
        stack_down = T.concatenate([prev_stack[:, 1:], zeros_at_the_top], axis=1)

        # a version of stack that is moved up (pop)
        stack_up = T.concatenate([input_val, prev_stack[:, :-1]], axis=1)

        # new stack
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down

        return new_stack

    def get_output_shape_for(self, input_shapes):
        """
        Returns new stack shape = last stack shape
        """
        observation_shape, last_memory_state_shape, controls_shape = input_shapes

        return last_memory_state_shape
