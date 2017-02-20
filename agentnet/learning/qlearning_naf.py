"""
Implements layers required to train qlearning with normalized advantage functions.
All the math taken from the original article: http://arxiv.org/abs/1603.00748
Loss function is exactly same as deterministic policy gradient (agentnet.learning.dpg)
Usage example: https://github.com/yandexdataschool/AgentNet/blob/master/examples/Continuous%20LunarLander%20%20using%20normalized%20advantage%20functions.ipynb
"""
import numpy as np
import theano.tensor as T
from lasagne.layers import InputLayer,DenseLayer,GaussianNoiseLayer,ElemwiseSumLayer,NonlinearityLayer

from collections import OrderedDict
from lasagne.layers import Layer, DenseLayer
from ..utils.logging import warn

def diag_to_tril_size(diag):
    return int(diag*(diag+1)/2)

def tril_size_to_diag(tril_size):
    return int(-0.5 + (0.25 + 2*tril_size)**.5)

class LowerTriangularLayer(Layer):
    def __init__(self, incoming, matrix_diag=None, **kwargs):
        """
        Fills out a lower-triangular matrix from either the incoming layer or it's transformation.

        :param incoming: layer used to fill lower-triangular matrix (if matrix_diag=None)
            or transformed into elements that fill such matrix (if matrix_diag != None).
            In the latter case, transformation is done via a single DenseLayer with n_outputs = n_elements to be filled
        :param matrix_diag: if not given, infers matrix size from incoming.output_shape [number of units must correspond to ]
            Other

        """
        name = kwargs.get("name", "")

        if matrix_diag is None:
            # infer matrix shape
            assert len(incoming.output_shape) == 2
            n_units = incoming.output_shape[-1]

            matrix_diag = tril_size_to_diag(n_units)
            if matrix_diag != int(matrix_diag):
                raise ValueError("""%s incoming layer must have num_units appropriate to fill a triangular matrix.
                                 If you wish to automatically infer the appropriate amount of linear units,
                                 use matrix_diag=<whatever>. """ % (self.__class__.__name__))
        else:
            if len(incoming.output_shape) == 2 and tril_size_to_diag(incoming.output_shape[1]) == matrix_diag:
                warn("""%s will learn transformation from it's input of size %s to new vector of same size %s.
                        If you instead want to use your input layer outputs explicitly as lower triangular matrix elements,
                        use matrix_diag = None param.""" % (
                self.__class__.__name__, incoming.output_shape, incoming.output_shape))
            incoming = DenseLayer(incoming,
                                  num_units=diag_to_tril_size(matrix_diag),
                                  name=name + ".preprocess_input",
                                  nonlinearity=None)

        self.matrix_diag = int(matrix_diag)
        super(LowerTriangularLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **flags):

        matrix_diag = self.matrix_diag

        matrix = T.zeros([input.shape[0], matrix_diag, matrix_diag])

        tril_i, tril_j = np.tril_indices(matrix_diag)
        batch_range = T.repeat(T.arange(input.shape[0])[:, None], len(tril_i), axis=1)

        # repeat_batch = lambda v: T.repeat(v[:,None],input.shape[0],axis=1)
        tril_indices = (batch_range, tril_i, tril_j)

        tril_matrix = T.set_subtensor(matrix[tril_indices], input)

        return tril_matrix

    def get_output_shape_for(self, input_shape,**flags):
        matrix_diag = self.matrix_diag
        return (input_shape[0], matrix_diag, matrix_diag)


from lasagne.layers import MergeLayer


class NAFLayer(MergeLayer):
    def __init__(self, action_layer, mean_layer, L_layer, **kwargs):
        """
        Computes the advantage term(A) for the Qvalues in the
        continuous Qlearning with normalized advantage functions

        for state s, action u (action layer)
        Q(s,u) = V(s) + A(s,u)
        A(s,u) = -0.5 * (u-mu) * L * L.T * (u-mu).T

        mu = optimal action (mean layer)
        L = lower triangular matrix (learned param)
        * denotes matrix multiplication, T denotes transposition

        :param action_layer: actions taken by agent (u term)
        :param mean_layer: optimal actions according to NAF (mu term)
        :param L_layer:

        """

        assert len(mean_layer.output_shape) == 2
        assert len(action_layer.output_shape) == 2

        super(NAFLayer, self).__init__([action_layer, mean_layer, L_layer], **kwargs)

    def get_output_for(self, inputs, **flags):
        u, mu, L = inputs

        # batch-wise matrix multiplication P = L * L.T
        P = T.batched_tensordot(L, L.swapaxes(2, 1), axes=[2, 1])

        # (u-mu) * P for each batch
        diff_times_P = T.batched_tensordot((u - mu), P, axes=[1, 2])

        # A = - 0.5 * (u-mu) * P * (u-mu).T
        A = -0.5 * T.batched_dot(diff_times_P, (u - mu))[:,None]

        #shape = (None,1)
        assert A.ndim ==2

        return A

    def get_output_shape_for(self, input_shapes,**flags):
        batch_size = input_shapes[0][0]
        return (batch_size,1)


def build_NAF_controller(input_layer = None,
                         action_dimensions=1,
                         exploration=1e-2,
                         additive_exploration=True,
                         mean_layer = None,
                         V_layer = None,
                         L_layer = None,
                         action_low=-np.inf,
                         action_high=np.inf,
                         ):
        '''
        Builds the regular NAF controller and outputs a dictionary of lasagne layers for each component.


        :param input_layer: layer which is used to predict policy parameters.
            MUST be present unless both V_layer, L_layer and mean_layer are given
        :param action_dimensions: amount of action params to predict.
        :param exploration: if a layer is given, uses it to compute the action with exploration (details: see additive_additive_exploration)
            Alternatively, if a number  or a symbolic scalar is given, uses it as sigma for gaussian exploration noise,
             thus action = mean_layer + N(0,exploration).
             To adjust
        :param additive_exploration: if True(default), adds exploration term to the mean_layer.
            if False, uses exploration param as the picked actions including exploration (ignoring mean values)

        :param mean_layer: layer which returns optimal actions (Mu).
            If not given, uses DenseLayer(input_layer)
        :param V_layer: layer which returns state value baseline (V(state))
        :param L_layer: a layer which is used to compute the advantage term
            A(u) =  -1/2 * (u - mu)^T * P_layer * (u - mu)
        :param action_low: minimum value for an action (float or np array for each action)
        :param action_high: maximum value for an action (float or np array for each action)
        :returns: a dict of 'means','actions','action_qvalues','state_values', 'advantage' to respective lasagne layers
        :rtype: collections.OrderedDict
        '''
        #TODO write tests for the damn thing
        if input_layer is None:
            assert (mean_layer is not None) and (V_layer is not None) and (L_layer is not None)

        #policy
        if mean_layer is None:
            mean_layer = DenseLayer(input_layer,num_units=action_dimensions,
                                    nonlinearity=None,name="qnaf.weights")
        assert len(mean_layer.output_shape)==2 and mean_layer.output_shape[-1] == action_dimensions

        #mean layer, clipped to action limits, used only for action picking
        mean_clipped = NonlinearityLayer(mean_layer, lambda a: a.clip(action_low, action_high))

        #action with exploration
        if not isinstance(exploration,InputLayer):
            #exploration is a number
            assert additive_exploration

            action_layer = GaussianNoiseLayer(mean_clipped,sigma=exploration)
        else:#exploration is a lasagne layer
            if additive_exploration:
                action_layer = ElemwiseSumLayer([mean_clipped,exploration])
            else:
                action_layer = exploration

        assert tuple(action_layer.output_shape)== tuple(mean_layer.output_shape)

        #state value
        if V_layer is None:
            V_layer = DenseLayer(input_layer,num_units=1,name="qnaf.state_value")

        assert len(V_layer.output_shape)==2 and V_layer.output_shape[-1]==1

        #L matrix (lower triangular
        if L_layer is None:
            L_layer = LowerTriangularLayer(input_layer,matrix_diag=action_dimensions,name="qnaf.L")

        assert len(L_layer.output_shape)==3 #shape must be [batch,action,aciton]
        assert L_layer.output_shape[1] == L_layer.output_shape[2] == action_dimensions

        advantage_layer = NAFLayer(action_layer,mean_layer,L_layer,name="qnaf.advantage")
        Q_layer = ElemwiseSumLayer([V_layer,advantage_layer])

        return OrderedDict([

            #means aka optimal actions aka Mu
            ('means', mean_layer),

            #actual actions after exploration
            ('actions', action_layer),

            # qvalue for actions from action_layer
            ('action_qvalues',Q_layer),

            # qvalue for optimal actions  aka V aka Q(Mu)
            ('state_value', V_layer),

            # advantage term (negative)
            ('advantage', advantage_layer)

        ])


#Since q-learning with NAF is effectively using a deterministic policy gradient over mu (mean, optimal action),
#we can seamlessly train it using dpg objective
from . import dpg
get_elementwise_objective = dpg.get_elementwise_objective_critic
