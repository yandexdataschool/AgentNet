import theano.tensor as T
import theano.tensor.shared_randomstreams as random_streams

from .base import BaseResolver


class ProbabilisticResolver(BaseResolver):
    """
    instance, that:
        - determines which action should be taken given policy
        - samples actions with probabilities given by input layer
    """

    def __init__(self, incoming, assume_normalized=False, seed=1234, output_dtype='int32',
                 name='ProbabilisticResolver'):
        """
        :param incoming: a lasagne layer that outputs action probability vectors
                WARNING! We assume that incoming probabilities are all nonnegative even if assume_normalized=False.
        :type incoming: lasagne.layers.Layer
        
        :param assume_normalized: if set to True, the incoming layer is assumed to 
            return outputs that add up to 1 (e.g. softmax output) along last axis
        :type assume_normalized: bool   
        
        :param seed: - random seed
        :type seed: int
        
        :action_dtype: type of action (usually (u)int 32 or 64)
        :type action_dtype: string or dtype
        
        :param name: layer name (using lasagne conventions)
        :type name: string
        """

        # probas float[2] - probability of random and optimal action respectively

        self.assume_normalized = assume_normalized

        self.rng = random_streams.RandomStreams(seed)

        super(ProbabilisticResolver, self).__init__(incoming, name=name,output_dtype=output_dtype)

    def get_output_for(self, policy, greedy=False, **kwargs):
        """
        picks the action with probabilities from policy
        :param policy: probabilities for all actions (e.g. a2c actor policy or standartized Q-values)
        :type policy: tensor of float[batch_id, action_id]
        
        :returns: actions ids of actions picked  
        :rtype: vector of int[batch_id]
        """
        if greedy:
            # greedy branch
            chosen_action_ids = T.argmax(policy, axis=-1).astype(self.output_dtype)

        else:
            # probabilistic branch
            batch_size, n_actions = policy.shape

            if self.assume_normalized:
                probas = policy
            else:
                probas = policy / T.sum(policy, axis=1, keepdims=True)

            # p1, p1+p2, p1+p2+p3, ... 1
            cum_probas = T.cumsum(probas, axis=1)

            batch_randomness = self.rng.uniform(low=0., high=1., size=[batch_size, 1])

            # idea: to compute the chosen action we count how many cumulative probabilities are
            # less than the random number [0,1].
            # we deliberately exclude the LAST cumulative probability because it has to be equal to 1
            # by definition (never being less than random[0,1]), but it can be less due to
            # inaccurate float32 computation, causing algorithm to pick action id = (n_actions)+1
            # which results in IndexError
            chosen_action_ids = T.sum((batch_randomness > cum_probas[:, :-1]), axis=1, dtype=self.output_dtype)

        return chosen_action_ids
