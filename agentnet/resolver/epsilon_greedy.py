import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams as random_streams

from .base import BaseResolver


class EpsilonGreedyResolver(BaseResolver):
    """
    Epsilon-greedy resolver:
        - determines which action should be taken given agent's policy,
        - takes maximum policy action with probability 1 - epsilon
        - takes random action with probability epsilon
    """

    def __init__(self, incoming, epsilon=None, seed=1234, name='EpsilonGreedyResolver',**kwargs):
        """
            epsilon float scalar: probability of random choice instead of optimal one
            seed constant: - random seed
        """
        if epsilon is None:
            epsilon = theano.shared(np.float32(0.1), "e-greedy.epsilon")

        self.epsilon = epsilon

        # probas float[2] - probability of random and optimal action respectively

        self.probas = T.stack([epsilon, 1 - epsilon])

        self.rng = random_streams.RandomStreams(seed)

        super(EpsilonGreedyResolver, self).__init__(incoming, name=name,**kwargs)

    def get_output_for(self, policy, **kwargs):
        """
        picks the action based on policy
        arguments:
            policy float[batch_id, action_id]: policy for all actions
        returns:
            actions int[batch_id]: ids of actions picked  
        """
        batch_size, n_actions = policy.shape

        # is_optimal_action  bool[batch_i] - 1 if agent takes optimal action at this time, 0 if random
        is_optimal_action = self.rng.choice(size=(batch_size,), a=2, p=self.probas, dtype='uint8')

        # best_action int[batch_i] - id of best_action
        best_action_ids = T.argmax(policy, axis=1).astype('int32')

        # random_action int[batch_i] - id of random action
        random_action_ids = self.rng.choice(size=(batch_size,), a=n_actions, dtype='int32')

        # chosen_action_ids int[batch_i] - action ids picked according to epsilon-greedy strategy
        chosen_action_ids = T.switch(
            is_optimal_action,
            best_action_ids,
            random_action_ids
        )

        return chosen_action_ids
