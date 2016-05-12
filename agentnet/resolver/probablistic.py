import theano.tensor as T

import lasagne
from .base import BaseResolver

from agentnet.utils.shared import create_shared

class ProbablisticResolver(BaseResolver):
    """
    instance, that:
        - determines which action should be taken given policy
        - samples actions with proibabilities given by input layer
    """
    
    def __init__(self,incoming,assume_normalized = False,seed = 1234,action_dtype='int32',
                 **kwargs):
        """
            incoming - a lasagne layer that outputs policy vectors
            assume_normalized - if set to True, the incoming layer is assumed to return outputs
                that add up to 1 (e.g. softmax)
            seed constant: - random seed
        """
        
        
        #probas float[2] - probability of random and optimal action respectively

        self.assume_normalized = assume_normalized
        self.action_dtype = action_dtype
        
        self.rng = T.shared_randomstreams.RandomStreams(seed)
                
        super(ProbablisticResolver, self).__init__(incoming, **kwargs)
        
        

        
    
    def get_output_for(self,policy,greedy=False,**kwargs):
        """
        picks the action with probabilities from policy
        arguments:
            policy float[batch_id, action_id]: policy values for all actions (e.g. Qvalues of action probabilities)
        returns:
            actions int[batch_id]: ids of actions picked  
        """
        
        ##probablistic branch
        if not greedy:
            batch_size,n_actions = policy.shape

            if self.assume_normalized:
                probas = policy
            else:
                probas = policy / T.sum(policy,axis=1,keepdims=True)


            #p1, p1+p2, p1+p2+p3, ... 1
            cum_probas = T.cumsum(probas,axis=1)


            batch_randomness = self.rng.uniform(low=0.,high=1., size = [probas.shape[0],1])


            #idea: to compute the chosen action we count how many cumulative probabilities are 
            #less than  the random number [0,1].
            #we deliberately exclude the LAST cumulative probability because it has to equal 1
            # by definition (never being less than random[0,1]), but it can be less due to
            #inaccurate float32 computation, causing algorithm to pick action id = (n_actions)+1
            #which results in IndexError
            chosen_action_ids = T.sum((batch_randomness > cum_probas[:,:-1]), axis=1, dtype=self.action_dtype)
            
        
        else: #greedy branch
        
            chosen_action_ids = T.argmax(policy,axis=-1).astype(self.action_dtype)

        return chosen_action_ids
    