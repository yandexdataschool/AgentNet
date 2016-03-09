__doc__ = """this module contains several auxilary functions used to print or plot agent's actions and state throughout session"""

import numpy as np
import matplotlib.pyplot as plt


def select_action_qvalues(qvalues_seq,action_seq):
    batch_i = np.arange(qvalues_seq.shape[0])[:,None]
    time_i = np.arange(qvalues_seq.shape[1])[None,:]

    return qvalues_seq[batch_i,time_i, action_seq]


def print_sessions(qvalues_seq,action_seq,reward_seq, action_names = None,
                   is_alive_seq = None, reference_qvalues_seq= None,
                  pattern = " {action}(qv = {qpred}) -> {reward}{qref} |",
                   
                  plot_qvalues = True, hidden_seq = None, legend = True,
                       qv_line_width = lambda action_i:1
                  ):
    """prints the sequence of agent actions along with specified predicted Qvalues
    
    parameters:
        qvalues_seq - qvalues for every [batch,tick,action]
        action_seq - actions taken on every [batch,tick]
        reward_seq - rewards given for action_seq on [batch,tick]
        action_names - names of all [actions]. Defaults to "action #i"
        is_alive_seq - whether or not session is terminated by [batch,tick]. Defaults to always alive
        reference_qvalues_seq - qvalues reference  for CHOSEN actions for each [batch,tick]
        pattern - how to print a single action cycle. Can use action, qpred, reward and qref variables.
        """

    if action_names is None:
        action_names = map("action #{}".format,range(np.max(action_seq)))
    if is_alive_seq is None:
        is_alive_seq = np.ones_like(action_seq)
    if reference_qvalues_seq is None:
        #dummy values
        reference_qvalues_seq = qvalues_seq
        print_reference = False
    else:
        print_reference = True
        
    #if we are on;y given one session, reshape everithing as a 1-session batch
    if len(action_seq.shape) ==1:
        qvalues_seq,action_seq,reward_seq,is_alive_seq,reference_qvalues_seq =\
            map(lambda v: v[None,:], [qvalues_seq,action_seq,reward_seq,is_alive_seq,reference_qvalues_seq])

    #if all qvalues are given for [batch,tick,action], select only qvalues for taken actions
    assert len(qvalues_seq.shape)==3
    if len(reference_qvalues_seq.shape) ==3:
        reference_qvalues_seq = select_action_qvalues(reference_qvalues_seq,action_seq)
    

        
    #loop over sessions
    for s_i in range(qvalues_seq.shape[0]):
        
                
        time_range = np.arange(qvalues_seq.shape[1])
        session_tuples = zip(qvalues_seq[s_i,time_range, action_seq[s_i]],
                             action_seq[s_i],reward_seq[s_i],
                             reference_qvalues_seq[s_i],is_alive_seq[s_i])
        
        
        #print session log
        print "session #",s_i
        
        for t_i, (qpred, a, r, qref,is_al) in enumerate(session_tuples):
            
            if not is_al:
                print '\n'
                break
            
            if print_reference:
                qref = "(ref = {})".format(qref)
            else:
                qref = ""
            
            action_name = action_names[a]
            
            print pattern.format(action=action_name, qpred=qpred, reward=r, qref=qref),
            
        else:
            print "reached max session length"
        
        #plot qvalues, actions, etc
        if plot_qvalues :
            plt.figure(figsize=[16,8])
            
            
            
            session_len = t_i
            
            #plot limits
            plt.xlim(0,max(session_len*1.1,2))
            plt.xticks(np.arange(session_len))
            plt.grid()

            
            
            q_values = qvalues_seq[s_i].T
            for a in range(q_values.shape[0]):
                plt.plot(q_values[a],label=action_names[a],linewidth =  qv_line_width(a))

            if hidden_seq is not None:
                hidden_activity =  hidden_seq[s_i].T

                for i, hh in enumerate(hidden_activity):
                    plt.plot(hh,'--',label='hidden #'+str(i))
                    
                    
            session_actions = action_seq[s_i,:session_len]
            action_range = np.arange(len(session_actions))
            
            plt.scatter(action_range, q_values[session_actions,action_range])

            if legend:
                plt.legend()

            #session end line
            plt.plot(np.repeat(session_len-1,2),plt.ylim(),c='blue')

            plt.show()
