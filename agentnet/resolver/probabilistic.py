import theano.tensor as T
import theano.tensor.shared_randomstreams as random_streams

from .base import BaseResolver


class ProbabilisticResolver(BaseResolver):
    """
    instance, that:
        - determines which action should be taken given policy
        - samples actions with probabilities given by input layer
    """

    def __init__(self, incoming, assume_normalized=False, seed=1234, action_dtype='int32',
                 name='ProbabilisticResolver'):
        """
            incoming - a lasagne layer that outputs policy vectors
            assume_normalized - if set to True, the incoming layer is assumed to return outputs
                that add up to 1 (e.g. softmax output)
            seed constant: - random seed
        """

        # probas float[2] - probability of random and optimal action respectively

        self.assume_normalized = assume_normalized
        self.action_dtype = action_dtype

        self.rng = random_streams.RandomStreams(seed)

        super(ProbabilisticResolver, self).__init__(incoming, name=name)

    def get_output_for(self, policy, greedy=False, **kwargs):
        """
        picks the action with probabilities from policy
        arguments:
            policy float[batch_id, action_id]: policy values for all actions (e.g. Q-values of action probabilities)
        returns:
            actions int[batch_id]: ids of actions picked  
        """
        if greedy:
            # greedy branch
            chosen_action_ids = T.argmax(policy, axis=-1).astype(self.action_dtype)

        else:
            # probabilistic branch
            batch_size, n_actions = policy.shape

            if self.assume_normalized:
                probas = policy
            else:
                # TODO (arogozhnikov) problems with negative values?
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
            chosen_action_ids = T.sum((batch_randomness > cum_probas[:, :-1]), axis=1, dtype=self.action_dtype)

        return chosen_action_ids
