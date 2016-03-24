
__doc__="""basic algorithms for Q-learning reference values assuming all agent actions are optimal. This, in principle, equals [session_length]-step Q-learning"""




import theano.tensor as T
import theano
import numpy as np

from ..utils import create_shared

default_gamma = create_shared('a3c_gamma_default',np.float32(0.99), theano.config.floatX)
    
def get_state_value_reference(state_values,rewards,
             is_alive="always",
             max_n = None,
             gamma_or_gammas = default_gamma,
             state_values_after_end = "zeros",     
             dependencies=[],strict = True):
    """
    Computes the 
    
    """
    
    if is_alive == "always":
        is_alive = T.ones_like(rewards)
        
    
    if state_values_after_end == "zeros":
        state_values_after_end = T.zeros([state_values.shape[0],1])
        

    #get "Next state_values": floatx[batch,time] at each tick, pad with state_values_after_end
    next_state_values = T.concatenate(
        [
            state_values[:,1:] * is_alive[:,1:],
            state_values_after_end,
        ],
        axis=1
    )
    
    
    
    
    #recurrent computation of reference state values (backwards through time)

    #initialize each reference with ZEROS after the end (won't be in output tensor)
    outputs_info = [T.zeros_like(rewards[:,0]),]   
    
    
    non_seqs = [gamma_or_gammas]+dependencies
    
    time_ticks = T.arange(rewards.shape[1])

    sequences = [rewards.T,is_alive.T,next_state_values.T,#transpose to iterate over time, not over batch
                 time_ticks] 

    def backward_V_step(rewards,is_alive,next_Vpred,time_i, 
                        next_Vref,
                        *args):
        
        
        
        propagated_Vref = T.switch(is_alive,
                           rewards + gamma_or_gammas * next_Vref, #assumes optimal next action
                           0.
                          )
        
        if max_n is None:
            this_Vref = propagated_Vref
        else:
            
            Vref_at_tmax = T.switch(is_alive,
                           rewards + gamma_or_gammas *next_Vpred,
                           0.
                          )
            
            this_Vref = T.switch(T.eq(time_i % max_n,0), #if Tmax
                                        Vref_at_tmax,  #use special case Qvalues
                                        propagated_Vref #else use generic ones
                                   )
                                         
                                 
        
        
        
        return this_Vref

    reference_state_values = theano.scan(backward_V_step,
                                    sequences=sequences,
                                    non_sequences=non_seqs,
                                    outputs_info=outputs_info,
                                    go_backwards=True,
                                    strict = strict
                                   )[0] #shape: [time_seq_inverted,batch]
        
    reference_state_values = reference_state_values.T[:,::-1] #[batch,time_seq]
    
    
    
    return reference_state_values


from ..utils.mdp import get_action_Qvalues
from ..utils import consider_constant

from lasagne.objectives import squared_error
    
def get_objective(policy,state_values,actions,reference_state_values,
                         is_alive = "always",min_log_proba = -1e3):
    """returns crossentropy-like objective function for Actor-Critic method
    
    L_policy = - log(policy)*(Vreference - const(V))
    L_V = (V - Vreference)^2
    
    """
    
    if is_alive == "always":
        is_alive = T.ones_like(actions,dtype=theano.config.floatX)
        
    
    action_probas =  get_action_Qvalues(policy,actions)
    
    reference_state_values = consider_constant(reference_state_values)
    
    
    total_n_actions = is_alive.sum()
    
    log_probas = T.maximum(T.log(action_probas),min_log_proba)
    
    policy_loss = - log_probas * (reference_state_values - consider_constant(state_values))
    policy_loss = T.sum(policy_loss*is_alive) / total_n_actions
    
    
    V_err_elwise = squared_error(reference_state_values,state_values)
    V_loss = T.sum(V_err_elwise*is_alive) / total_n_actions
    
    return policy_loss + V_loss
    
