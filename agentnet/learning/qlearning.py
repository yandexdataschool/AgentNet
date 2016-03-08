
__doc__="""basic function for Q-learning reference values"""

import theano.tensor as T
import theano
import numpy as np

from ..utils import create_shared
default_gamma = create_shared('qlearning_gamma_default',np.float32(0.99), theano.config.floatX)
    
    
def get_reference(Qvalues,actions,rewards,
                  gamma_or_gammas = default_gamma,
                  qvalues_after_end = "zeros",
                  aggregation_function = lambda qv:T.max(qv,axis=1)
                            ):
    """
    Returns reference Qvalues according to Q-learning algorithm
    
    parameters:
    Qvalues [batch,tick,action_id] - predicted qvalues
    actions [batch,tick] - commited actions
    gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
    qvalues_after_end - symbolic expression for "next state q-values" for last tick used for reference only. 
                        Defaults at  T.zeros_like(Qvalues[:,0,None,:])
                        If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
                        
    Aggregation function - a function that takes all Qvalues for "next state qvalues" term and returns what is the "best next Qvalue". 
                            Normally you should not touch it.
        
    Returns:
    Qreference - reference qvalues at [batch,tick] using formula
    
        Q reference [batch,action_at_tick] = rewards[t] + gamma_or_gammas* Qs(t+1, best_action_at(t+1))
    
            where  Qs(t+1, best_action_at(t+1)) is computed as aggregation_function(next_Qvalues)
    
    """
    if qvalues_after_end == "zeros":
        qvalues_after_end = T.zeros_like(Qvalues[:,0,None,:])
        

    # Qvalues for "next" states (padded with zeros at the end): float[batch,tick,action]
    next_Qvalues_predicted = T.concatenate(
        [
            Qvalues[:,1:],
            qvalues_after_end,
        ],
        axis=1
    )

    #"optimal next reward" after commiting action : float[batch,tick]
    next_Qvalues = next_Qvalues_predicted.reshape([-1,next_Qvalues_predicted.shape[-1]])
    optimal_next_Qvalue = aggregation_function(next_Qvalues).reshape(next_Qvalues_predicted.shape[:-1])


    #full Qvalue formula (taking chosen_action and behaving optimally later on)
    reference_Qvalues = rewards + gamma_or_gammas*optimal_next_Qvalue
          

    return reference_Qvalues
    
