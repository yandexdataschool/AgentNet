
__doc__="""basic algorithms for Q-learning reference values"""

import theano.tensor as T
import theano
import numpy as np

from ..utils import create_shared
default_gamma = create_shared('gamma_default',np.float32(0.99), theano.config.floatX)
    
    
def get_reference(Qvalues,actions,rewards,
                  gamma_or_gammas = default_gamma,
                  aggregation_function = lambda qv:T.max(qv,axis=1)
                            ):
    """
    parameters:
    Qvalues [batch,tick,action_id] - predicted qvalues
    actions [batch,tick] - commited actions
    is_alive[batch,tick] - 1 if session is still active by this tick, 0 otherwise
    gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
        
    Returns:
    computes three vectors:
    action IDs (1d,single integer) vector of all actions that were commited during all sequences
    in the batch, concatenated in one vector... that is, excluding the ones that happened
    after the end_code action was sent.
    Qpredicted - qvalues for action_IDs ONLY at each time 
    before sequence end action was committed for each sentence in the batch (concatenated) 
    but for the last time-slot.
    Qreference - sum over immediate rewards and gamma*predicted qvalues for
    next round after first vector predictions
    if naive == True, all actions are considered optimal in terms of Qvalue
    """
        
        

    # Qvalues for "next" states (padded with zeros at the end): float[batch,tick,action]
    next_Qvalues_predicted = T.concatenate(
        [
            Qvalues[:,1:],
            T.zeros_like(Qvalues[:,0,None,:])
        ],
        axis=1
    )

    #"optimal next reward" after commiting action : float[batch,tick]
    next_Qvalues = next_Qvalues_predicted.reshape([-1,next_Qvalues_predicted.shape[-1]])
    optimal_next_Qvalue = aggregation_function(next_Qvalues).reshape(next_Qvalues_predicted.shape[:-1])


    #full Qvalue formula (taking chosen_action and behaving optimally later on)
    reference_Qvalues = rewards + gamma_or_gammas*optimal_next_Qvalue
          

    return reference_Qvalues
    
def get_reference_naive(Qvalues,actions,rewards,is_alive="always",
                        gamma_or_gammas = default_gamma,
                        dependencies=[],strict = True):
    """ computes Qvalues assuming all actions are optimal
    params:
        rewards: immediate rewards floatx[batch_size,time]
        is_alive: whether the session is still active int/bool[batch_size,time]
        gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]
        dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        strict: whether to evaluate Qvalues using strict theano scan or non-strict one
    returns:
        Q reference [batch,action_at_tick] = _rewards[t] + _gamma_or_gammas* Qs(t+1, best_action_at(t+1))
        
        using our assumption on optimality:
                    best_action_at(t) = action(t)
                    
        this essentially becomes:
            Qs(t,picked_action) = _rewards[t] + _gamma_or_gammas* Qs(t+1,next_picked_action)
                and does not require actions themselves to compute (given rewards)

    """
    
    if is_alive == "always":
        is_alive = T.ones_like(rewards)
                            
    #recurrent computation of Qvalues reference (backwards through time)
                            
    outputs_info = [T.zeros_like(rewards[:,0]),]
    non_seqs = [gamma_or_gammas]+dependencies

    sequences = [rewards.T,is_alive.T] #transpose to iterate over time, not over batch

    def backward_qvalue_step(rewards,is_alive, next_Qs,*args):
        this_Qs = T.switch(is_alive,
                           rewards + gamma_or_gammas * next_Qs, #assumes optimal next action
                           0.
                          )
        return this_Qs

    reference_Qvalues = theano.scan(backward_qvalue_step,
                                    sequences=sequences,
                                    non_sequences=non_seqs,
                                    outputs_info=outputs_info,
                                    go_backwards=True,
                                    strict = strict
                                   )[0] #shape: [time_seq_inverted,batch]
        
    reference_Qvalues = reference_Qvalues.T[:,::-1] #[batch,time_seq]
        

    return reference_Qvalues

    
    
