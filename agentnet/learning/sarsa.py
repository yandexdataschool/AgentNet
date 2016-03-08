
__doc__="""basic function for Q-learning reference values"""

import theano.tensor as T
import theano
import numpy as np

from ..utils import create_shared
default_gamma = create_shared('sarsa_gamma_default',np.float32(0.99), theano.config.floatX)
    
from ..utils.mdp import get_action_Qvalues


def get_reference(Qvalues,actions,rewards,
                  gamma_or_gammas = default_gamma,
                  future_rewards_after_end = "zeros"
                 ):
    """
    Returns reference Qvalues according to State-Action-Reward-State-Action (SARSA) algorithm
    
    parameters:
    Qvalues [batch,tick,action_id] - predicted qvalues
    actions [batch,tick] - commited actions
    gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
    future_rewards_after_end - symbolic expression for "future rewards" term for last tick used for reference only. 
                        Defaults at  T.zeros_like(rewards[:,0,None])
                        If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
                        
        
    Returns:
    Qreference - reference qvalues at [batch,tick] using formula
    
        Q reference [batch,action_at_tick] = rewards[t] + gamma_or_gammas* Qs(t+1,action[t+1])
        Where action[t+1] is simply action that agent took at next time tick [padded with future_rewards_after_end]
    
    
    """
    if future_rewards_after_end == "zeros":
        future_rewards_after_end = T.zeros_like(rewards[:,0,None])
        

    # Qvalues for "next" states (missing last tick): float[batch,tick-1,action]
    next_Qvalues_predicted = Qvalues[:,1:]
    #actions commited at next ticks (missing last tick): int[batch,tick-1]
    next_actions = actions[:,1:]
    
    
    future_rewards_estimate = get_action_Qvalues(next_Qvalues_predicted,next_actions)
    
    #adding the last tick
    future_rewards_estimate = T.concatenate(
        [
            future_rewards_estimate,
            future_rewards_after_end,
        ],
        axis=1
    )
    
    
    #full Qvalue formula (SARSA algorithm)
    reference_Qvalues = rewards + gamma_or_gammas*future_rewards_estimate
          

    return reference_Qvalues
    
