"""
A simple stack augmentation for RNN http://arxiv.org/abs/1503.01007
"""
from lasagne.layers.base import MergeLayer

import theano.tensor as T

from ..utils import insert_dim


class StackAugmentation(MergeLayer):
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
