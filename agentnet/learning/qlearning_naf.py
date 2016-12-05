import numpy as np
import theano.tensor as T
from lasagne.layers import InputLayer,DenseLayer,GaussianNoiseLayer,ElemwiseSumLayer,ExpressionLayer,FlattenLayer
from agentnet.utils.layers import DictLayer

from warnings import warn
from collections import OrderedDict
from lasagne.layers import Layer, DenseLayer

def diag_to_tril_size(diag):
    return diag*(diag+1)/2

def tril_size_to_diag(tril_size):
    return (-0.5 + (0.25 + 2*tril_size)**.5)

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


def build_NAF_controller(self,
                         input_layer = None,
                         action_dimensions=1,
                         exploration=1e-2,
                         additive_exploration=True,
                         mean_layer = None,
                         V_layer = None,
                         L_layer = None,
                         ):
        '''
        Builds the regular NAF controller and outputs a dictionary of lasagne layers for each component.

        action space is (-inf,inf) by default!

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

        #action with exploration
        if not isinstance(exploration,InputLayer):
            #exploration is a number
            assert additive_exploration
            action_layer = GaussianNoiseLayer(mean_layer,sigma=exploration)
        else:#exploration is a lasagne layer
            if additive_exploration:
                action_layer = ElemwiseSumLayer([mean_layer,exploration])
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
        print V_layer.output_shape,advantage_layer.output_shape
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

            # advantage term
            ('advantage', advantage_layer)

        ])



from helpers import get_n_step_value_reference,get_end_indicator
from ..utils.grad import consider_constant
from lasagne.objectives import squared_error
def get_elementwise_objective(action_qvalues,
                              state_values,
                              rewards,
                              is_alive="always",
                              n_steps=None,
                              gamma_or_gammas=0.95,
                              crop_last=True,
                              force_state_values_after_end=True,
                              state_values_after_end="zeros",
                              consider_reference_constant=True,
                              return_reference=False,
                              scan_dependencies=(),
                              scan_strict=True):
    """
    Returns squared error between predicted and reference Q-values according to n-step Q-learning algorithm

        Qreference(state,action) = reward(state,action) + gamma*reward(state_1,action_1) + ... + gamma^n * max[action_n]( Q(state_n,action_n)
        loss = mean over (Qvalues - Qreference)**2

    :param action_qvalues: [batch,tick,action_id] - predicted qvalues
    :param state_values: [batch,tick] - predicted state values (aka qvalues for best actions)
    :param actions: [batch,tick] - commited actions
    :param rewards: [batch,tick] - immediate rewards for taking actions at given time ticks
    :param is_alive: [batch,tick] - whether given session is still active at given tick. Defaults to always active.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss
    :param n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
    :param gamma_or_gammas: delayed reward discounts: a single value or array[batch,tick](can broadcast dimensions).
    :param crop_last: if True, zeros-out loss at final tick, if False - computes loss VS Qvalues_after_end
    :param force_state_values_after_end: if true, sets reference Qvalues at session end to rewards[end] + qvalues_after_end
    :param state_values_after_end: [batch,1] - symbolic expression for "best next state q-values" for last tick
                            used when computing reference Q-values only.
                            Defaults at  T.zeros_like(Q-values[:,0,None,0])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
    :param consider_reference_constant: whether or not zero-out gradient flow through reference_Qvalues
            (True is highly recommended)
    :param aggregation_function: a function that takes all Qvalues for "next state Q-values" term and returns what
                                is the "best next Q-value". Normally you should not touch it. Defaults to max over actions.
                                Normally you shouldn't touch this
                                Takes input of [batch,n_actions] Q-values
    :param return_reference: if True, returns reference Qvalues.
            If False, returns squared_error(action_Qvalues, reference_Qvalues)
    :param scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
    :param scan_strict: whether to evaluate Qvalues using strict theano scan or non-strict one
    :return: mean squared error over Q-values (using formula above for loss)

    """

    assert action_qvalues.ndim  == state_values.ndim == rewards.ndim ==2
    if is_alive != 'always': assert is_alive.ndim==2


    # get reference Q-values via Q-learning algorithm
    reference_qvalues = get_n_step_value_reference(
        state_values=state_values,
        rewards=rewards,
        is_alive=is_alive,
        n_steps=n_steps,
        gamma_or_gammas=gamma_or_gammas,
        optimal_state_values_after_end=state_values_after_end,
        dependencies=scan_dependencies,
        strict=scan_strict
    )

    if consider_reference_constant:
        # do not pass gradient through reference Qvalues (since they DO depend on Qvalues by default)
        reference_qvalues = consider_constant(reference_qvalues)

    if force_state_values_after_end and is_alive != "always":
        # if asked to force reference_Q[end_tick+1,a] = 0, do it
        # note: if agent is always alive, this is meaningless
        # set future rewards at session end to rewards+qvalues_after_end
        end_ids = get_end_indicator(is_alive, force_end_at_t_max=True).nonzero()

        if state_values_after_end == "zeros":
            # "set reference Q-values at end action ids to just the immediate rewards"
            reference_qvalues = T.set_subtensor(reference_qvalues[end_ids], rewards[end_ids])
        else:
            # "set reference Q-values at end action ids to the immediate rewards + qvalues after end"
            new_reference_values = rewards[end_ids] + gamma_or_gammas * state_values_after_end
            reference_qvalues = T.set_subtensor(reference_qvalues[end_ids], new_reference_values[end_ids[0], 0])

    #If asked, make sure loss equals 0 for the last time-tick.
    if crop_last:
        reference_qvalues = T.set_subtensor(reference_qvalues[:,-1],state_values[:,-1])

    if return_reference:
        return reference_qvalues
    else:
        # tensor of elementwise squared errors
        elwise_squared_error = squared_error(action_qvalues,reference_qvalues)
        return elwise_squared_error * is_alive
