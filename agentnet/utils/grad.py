"""
Symbolic gradient-related operations such as "consider constant", "reverse gradient", etc
"""

import theano

# Specify default consider_constant op.
# Changing it here will change behaviour throughout entire library (except examples)
from theano.gradient import disconnected_grad

consider_constant = disconnected_grad


class MultiplyGradient(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda=1):
        """this operation multiplies the gradient by hp_lambda when computing grads
        
        Code by Daniel Renshaw 
         - http://stackoverflow.com/users/127480/daniel-renshaw
     
        All thanks to him.
        """
        super(MultiplyGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [self.hp_lambda * output_gradients[0]]


reverse_gradient = MultiplyGradient(-1)
