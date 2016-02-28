import theano.tensor as T

import lasagne
class BaseResolver(lasagne.layers.Layer):
    """
    special Lasagne Layer instance, that:
        - determines Q-values for all actions given current agent state and current observation,
    """
    def __init__(self,incoming,*args,**kwargs):
        super(BaseResolver, self).__init__(incoming, **kwargs)
    def reset():
        """
        performs this action each time a new session [batch] is loaded
        """
        pass

    
    def get_output_for(self,Qvalues,**kwargs):
        """
        picks the action based on Qvalues
        arguments:
            Qvalues float[batch_id, action_id]: Qvalues for all actions
        returns:
            actions int[batch_id]: ids of actions picked  
        """
        
        return T.argmax(Qvalues,axis=1)
    def get_output_shape_for(self,input_shape):
        """
        returns output shape [batch_id, 1]
        """
        batch_size = input_shape[0]
        return [batch_size,1]
               
        




class EpsilonGreedyResolver(BaseResolver):
    """
    instance, that:
        - determines Q-values for all actions given current agent state and current observation,
    """
    
    def __init__(self,incoming,epsilon,seed = 1234,**kwargs):
        """
            epsilon float scalar: probability of random choice instead of optimal one
            seed constant: - random seed
        """
        self.epsilon = epsilon
        
        
        #probas float[2] - probability of random and optimal action respectively

        self.probas = T.stack([epsilon,1-epsilon]) 
        
        
        self.rng = T.shared_randomstreams.RandomStreams(seed)
        
        super(EpsilonGreedyResolver, self).__init__(incoming, **kwargs)

        
    
    def get_output_for(self,Qvalues,**kwargs):
        """
        picks the action based on Qvalues
        arguments:
            Qvalues float[batch_id, action_id]: Qvalues for all actions
        returns:
            actions int[batch_id]: ids of actions picked  
        """
        batch_size,n_actions = Qvalues.shape
        
        
        #is_optimal_action  bool[batch_i] - 1 if agent takes optimal action at this time, 0 if random
        is_optimal_action = self.rng.choice(size=(batch_size,),a=2,p=self.probas,dtype='uint8')

        #best_action int[batch_i] - id of best_action
        best_action_ids = T.argmax(Qvalues,axis=1).astype('int32')

        #random_action int[batch_i] - id of random action
        random_action_ids = self.rng.choice(size = (batch_size,), a = n_actions,dtype='int32')

        #chosen_action_ids int[batch_i] - action ids picked according to epsilon-greedy strategy
        chosen_action_ids = T.switch(
            is_optimal_action,
                best_action_ids,
                random_action_ids
            )
        
        return chosen_action_ids
    