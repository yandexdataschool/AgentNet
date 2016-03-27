___doc___ = """A few functions used for various mdp-related stuff that are not large enoguh to get their own modules"""

from theano import tensor as T

def get_action_Qvalues(Qvalues,actions):
    """auxilary function to select Qvalues corresponding to actions taken
        Returns Qvalues predicted that resulted in actions: float[batch,tick]"""

    batch_i = T.arange(Qvalues.shape[0])[:,None]
    time_i = T.arange(Qvalues.shape[1])[None,:]
    action_Qvalues_predicted= Qvalues[batch_i,time_i, actions]
    return action_Qvalues_predicted

def get_end_indicator( is_alive,force_end_at_t_max = False):
    """ auxilary function to transform session alive indicator into end action indicator
    If force_end_at_t_max is True, all sessions that didn't end by the end of recorded sessions
    are ended at the last recorded tick."""
    #session-ending action indicator: uint8[batch,tick]
    is_end = T.eq(is_alive[:,:-1] - is_alive[:,1:],1)
    
    if force_end_at_t_max:
        session_ended_before = T.sum(is_end,axis=1,keepdims=True)
        is_end_at_tmax = 1 - T.gt(session_ended_before, 0 )
    else:
        is_end_at_tmax = T.zeros((is_end.shape[0],1),dtype=is_end.dtype) 
    
    is_end = T.concatenate(
        [is_end,
         is_end_at_tmax],
        axis=1
    )
    return is_end


def ravel_alive(is_alive,*args):
    """takes all is_alive ticks from all sessions and merges them into 1 dimension"""
    alive_selector = is_alive.nonzero()
    return [arg[alive_selector] for arg in args]

