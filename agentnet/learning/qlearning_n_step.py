
__doc__="""basic algorithms for Q-learning reference values assuming all agent actions are optimal. This, in principle, equals [session_length]-step Q-learning"""

import theano.tensor as T
import theano
import numpy as np

from ..utils import create_shared
default_gamma = create_shared('n_step_qlearning_gamma_default',np.float32(0.99), theano.config.floatX)
    
def get_reference_Qvalues(Qvalues,actions,rewards,is_alive="always",
                        qvalues_after_end = "zeros",
                        n_steps = None,
                        gamma_or_gammas = default_gamma,
                        aggregation_function = lambda qv:T.max(qv,axis=-1),
                        dependencies=[],strict = True):
    """ computes Qvalues using an N-step q-learning algorithm. 
    If n_steps is None, computes "naive" RL reference, assuming all actions optimal.
    
    params:
        Qvalues: predicted Qvalues floatx[batch_size,time,action]. 
            If n_steps is None(see next), they're unused so you can provide arbitrary(e.g. zero) tensor of that shape.
            
        rewards: immediate rewards floatx[batch_size,time]
        
        is_alive: whether the session is still active int/bool[batch_size,time]
        
        qvalues_after_end - symbolic expression for "next state q-values" for last tick used for reference only. 
                        Defaults at  T.zeros_like(Qvalues[:,0,None,:])
                        If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )
        
        n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
            
        
        gamma_or_gammas: delayed reward discount number, scalar or vector[batch_size]
        
        aggregation_function - a function that takes all Qvalues for "next state qvalues" term and 
                    returns what is the "best next Qvalue" at the END of n-step cycle. 
                    Normally you should not touch it.
                    Takes input of [batch,n_actions] Qvalues

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
        
        if n_steps is None:
            this_Qref =  propagated_Qvalues
        else:
            
            qvalues_at_tmax = T.switch(is_alive,
                           rewards + gamma_or_gammas * aggregation_function(next_Qpredicted),
                           0.
                          )
            
            this_Qref = T.switch(T.eq(time_i % n_steps,0), #if Tmax
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

    
    
    
    
import lasagne
from ..utils.mdp import get_end_indicator, get_action_Qvalues
from ..utils import consider_constant

    
def get_elementwise_objective(Qvalues,actions,rewards,
                              is_alive = "always",
                              n_steps = None,
                              gamma_or_gammas = 0.95,
                              force_qvalues_after_end = True,
                              qvalues_after_end = "zeros",
                              consider_reference_constant = True,
                              aggregation_function = lambda qv:T.max(qv,axis=-1),
                              scan_dependencies = [], scan_strict = True):
    """
    Returns squared error between predicted and reference Qvalues according to Q-learning algorithm
    
        Qreference(state,action) = reward(state,action) + gamma* max[next_action]( Q(next_state,next_action)  
        loss = mean over (Qvalues - Qreference)**2
        
    parameters:
    
        Qvalues [batch,tick,action_id] - predicted qvalues
        actions [batch,tick] - commited actions
        rewards [batch,tick] - immediate rewards for taking actions at given time ticks
        
        is_alive [batch,tick] - whether given session is still active at given tick. Defaults to always active.
                            Default value of is_alive implies a simplified computation algorithm for Qlearning loss
        
        n_steps: if an integer is given, the references are computed in loops of 3 states.
            Defaults to None: propagating rewards throughout the whole session.
            If n_steps equals 1, this works exactly as Q-learning (though less efficient one)
            If you provide symbolic integer here AND strict = True, make sure you added the variable to dependencies.
        
        gamma_or_gammas - a single value or array[batch,tick](can broadcast dimensions) of delayed reward discounts 
        
        force_qvalues_after_end - if true, sets reference Qvalues at session end to rewards[end] + qvalues_after_end
        
        qvalues_after_end [batch,1,n_actions] - symbolic expression for "next state q-values" for last tick used for reference only. 
                            Defaults at  T.zeros_like(Qvalues[:,0,None,:])
                            If you wish to simply ignore the last tick, use defaults and crop output's last tick ( qref[:,:-1] )

        consider_reference_constant - whether or not zero-out gradient flow through reference_Qvalues (True highly recommended)

        aggregation_function - a function that takes all Qvalues for "next state qvalues" term and returns what 
                                is the "best next Qvalue". Normally you should not touch it. Defaults to max over actions.
                                Normaly you shouldn't touch this
                                Takes input of [batch,n_actions] Qvalues

        scan_dependencies: everything you need to evaluate first 3 parameters (only if strict==True)
        scan_strict: whether to evaluate Qvalues using strict theano scan or non-strict one
    Returns:
                
        mean squared error over Qvalues (using formua above for loss)

    """
    
    #get reference Qvalues via Q-learning algorithm
    reference_Qvalues = get_reference_Qvalues(Qvalues,actions,rewards, is_alive,
                                              n_steps = n_steps,
                                              qvalues_after_end = qvalues_after_end,
                                              gamma_or_gammas = gamma_or_gammas,
                                              aggregation_function = aggregation_function,
                                              dependencies = scan_dependencies,
                                              strict = scan_strict
                                             )
    
    if consider_reference_constant:
        #do not pass gradient through reference Qvalues (since they DO depend on Qvalues by default)
        reference_Qvalues = consider_constant(reference_Qvalues)
    
    #get predicted qvalues for commited actions (to compare with reference Qvalues)
    action_Qvalues = get_action_Qvalues(Qvalues,actions)
    
    
    
    #if agent is always alive, return the simplified loss
    if is_alive == "always":
        
        #tensor of elementwise squared errors
        elwise_squared_error = lasagne.objectives.squared_error(reference_Qvalues,action_Qvalues)

        
        
    else: #we are given an is_alive matrix : uint8[batch,tick] 

        #if asked to force reference_Q[end_tick+1,a] = 0, do it
        #note: if agent is always alive, this is meaningless
        
        if force_qvalues_after_end:
            #set future rewards at session end to rewards+qvalues_after_end
            end_ids = get_end_indicator(is_alive,force_end_at_t_max = True).nonzero()

            if qvalues_after_end == "zeros":
                # "set reference Qvalues at end action ids to just the immediate rewards"
                reference_Qvalues = T.set_subtensor(reference_Qvalues[end_ids],
                                                    rewards[end_ids]
                                                    )
            else:
                last_optimal_rewards = aggregation_function(qvalues_after_end)
            
                # "set reference Qvalues at end action ids to the immediate rewards + qvalues after end"
                reference_Qvalues = T.set_subtensor(reference_Qvalues[end_ids],
                                                    rewards[end_ids] + gamma_or_gammas*last_optimal_rewards[end_ids[0],0]
                                                    )
        
    
        #tensor of elementwise squared errors
        elwise_squared_error = lasagne.objectives.squared_error(reference_Qvalues,action_Qvalues)

        #zero-out loss after session ended
        elwise_squared_error = elwise_squared_error * is_alive

    
    return elwise_squared_error
    