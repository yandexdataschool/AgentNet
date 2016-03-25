__doc__="""symbolic gradient-related operations such as "consider constant", "reverse gradient", etc"""

from theano import tensor as T
import theano
from theano.tensor.opt import register_canonicalize
import numpy as np



#consider constant op by benanne
#https://gist.github.com/benanne/9212037
class ConsiderConstant(theano.compile.ViewOp):
    """treats input as constant when computing grads"""
    def grad(self, args, g_outs):
        return [T.zeros(g_out.shape,dtype=theano.config.floatX) for g_out in g_outs]

consider_constant = ConsiderConstant()
register_canonicalize(theano.gof.OpRemove(consider_constant), name='remove_consider_constant_op')


#gradient reversal layer by Daniel Renshaw 
#http://stackoverflow.com/users/127480/daniel-renshaw
#thanks to him, but idk if it works :P
class MultiplyGradient(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda=1):
        """this operation multiplies the gradient by hp_lambda when computing grads"""
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