
__doc__="""basic algorithms for Q-learning reference values assuming all agent actions are optimal. This, in principle, equals [session_length]-step Q-learning"""

import theano.tensor as T
import theano
import numpy as np

from ..utils import create_shared
default_gamma = create_shared('naive_qlearning_gamma_default',np.float32(0.99), theano.config.floatX)
    
def get_reference(Qvalues,actions,rewards,is_alive="always",
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

    
    
