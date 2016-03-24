
__doc__="""basic algorithms for Q-learning reference values assuming all agent actions are optimal. This, in principle, equals [session_length]-step Q-learning"""

import theano.tensor as T
import theano
import numpy as np

from ..utils import create_shared
default_gamma = create_shared('n_step_qlearning_gamma_default',np.float32(0.99), theano.config.floatX)
    
def get_reference(Qvalues,actions,rewards,is_alive="always",
                        qvalues_after_end = "zeros",
                        max_n = None,
                        gamma_or_gammas = default_gamma,
                        aggregation_function = lambda qv:T.max(qv,axis=1),
                        dependencies=[],strict = True):
    """ computes Qvalues using an N-step q-learning algorithm. 
    If max_n is None, computes "naive" RL reference, assuming all actions optimal.
    
    params:
        Qvalues: predicted Qvalues floatx[batch_size,time,action]. 
            If max_n is None(see next), they're unused so you can provide arbitrary(e.g. zero) tensor of that shape.
            
        rewards: immediate rewards floatx[batch_size,time]
        
        is_alive: whether the session is still active int/bool[batch_size,time]
        
        qvalues_after_end - symbolic expression for "next state q-values" for last tick used for reference only. 
                        Defaults at  T.zeros_like(Qvalues[:,0,None,:])
                        If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
        
        max_n: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If max_n equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
            
        
        gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]
        
        aggregation_function - a function that takes all Qvalues for "next state qvalues" term and 
                    returns what is the "best next Qvalue" at the END of n-step cycle. 
                    Normally you should not touch it.

        dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        strict: whether to evaluate Qvalues using strict theano scan or non-strict one
    returns:
        Q reference [batch,action_at_tick] according to N-step Q_learning
        mentioned here http://arxiv.org/pdf/1602.01783.pdf as Algorithm 3

    """
    
    if is_alive == "always":
        is_alive = T.ones_like(rewards)
        
    if qvalues_after_end == "zeros":
        qvalues_after_end = T.zeros_like(Qvalues[:,0,None,:])
        

    # Qvalues for "next" states (padded with zeros at the session end): float[batch,tick,action]
    next_Qvalues_predicted = T.concatenate(
        [
            Qvalues[:,1:] * is_alive[:,1:,None],
            qvalues_after_end,
        ],
        axis=1
    )
    
    
    
    
    #recurrent computation of Qvalues reference (backwards through time)
                            
    outputs_info = [T.zeros_like(rewards[:,0]),]   #start each reference with ZEROS after the end
    
    
    non_seqs = [gamma_or_gammas]+dependencies
    
    time_ticks = T.arange(rewards.shape[1])

    sequences = [rewards.T,is_alive.T,
                 next_Qvalues_predicted.dimshuffle(1,0,2),#transpose to iterate over time, not over batch
                 time_ticks] 

    def backward_qvalue_step(rewards,is_alive,next_Qpredicted,time_i, 
                             next_Qref,*args):
        
        
        
        propagated_Qvalues = T.switch(is_alive,
                           rewards + gamma_or_gammas * next_Qref, #assumes optimal next action
                           0.
                          )
        
        if max_n is None:
            this_Qref =  propagated_Qvalues
        else:
            
            qvalues_at_tmax = T.switch(is_alive,
                           rewards + gamma_or_gammas * aggregation_function(next_Qpredicted),
                           0.
                          )
            
            this_Qref = T.switch(T.eq(time_i % max_n,0), #if Tmax
                                        qvalues_at_tmax,  #use special case Qvalues
                                        propagated_Qvalues #else use generic ones
                                   )
                                         
                                 
        
        
        
        return this_Qref

    reference_Qvalues = theano.scan(backward_qvalue_step,
                                    sequences=sequences,
                                    non_sequences=non_seqs,
                                    outputs_info=outputs_info,
                                    go_backwards=True,
                                    strict = strict
                                   )[0] #shape: [time_seq_inverted,batch]
        
    reference_Qvalues = reference_Qvalues.T[:,::-1] #[batch,time_seq]
        

    return reference_Qvalues

    
    
