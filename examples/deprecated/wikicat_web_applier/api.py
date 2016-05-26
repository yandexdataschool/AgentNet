import nnet
import pickle
import os


def next_name(folder = "./states/"):
    i = 0
    files = set(os.listdir(folder))
    while True:
        fname= "state"+str(i)
        if fname not in files:
            return os.path.join(folder,fname)
        i+=1


def new_session():
    
    observation = nnet.fake_observation
    state = nnet.zero_memory_state
    
    reaction = nnet.get_agent_reaction(state,observation)
    
    
    state_1,policy_1,action_1 = reaction
    
    fname = next_name()
    with open(fname,'w') as fout:
        pickle.dump(reaction,fout)
    return fname, dict(zip(nnet.feature_names,policy_1)), nnet.feature_names[action_1]
    
def load_state(state_fname):
    with open(state_fname,'r') as fin:
        state, policy, action = pickle.load(fin)
        return state_fname, dict(zip(nnet.feature_names,policy)), nnet.feature_names[action]
    
def get_next_state(last_state_fname,response ):
    with open(last_state_fname,'r') as fin:
        prev_state, prev_policy, prev_action = pickle.load(fin)
        
    observation = nnet.response_to_observation(response,prev_action)
    reaction = nnet.get_agent_reaction(prev_state,observation)
    
    fname = next_name()
    with open(fname,'w') as fout:
        pickle.dump(reaction,fout)

    _,policy, action = reaction
    return fname, dict(zip(nnet.feature_names,policy)), nnet.feature_names[action]
