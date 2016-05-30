import numpy as np
import os
import sys
import lasagne
import theano
import theano.tensor as T
floatX = theano.config.floatX


#Create an environment with all default parameters
from agentnet.experiments import wikicat as experiment
env = experiment.WikicatEnvironment()
attrs,cats,feature_names = env.get_dataset()
env.load_data_batch(attrs,cats)

#setup agent

from agentnet.resolver import EpsilonGreedyResolver
from agentnet.memory import GRUMemoryLayer
from agentnet.agent import Agent

n_hid_1=1024 #first GRU memory
n_hid_2=1024 #second GRU memory


_observation_layer = lasagne.layers.InputLayer([None,env.observation_size],name="obs_input")

_prev_gru1_layer = lasagne.layers.InputLayer([None,n_hid_1],name="prev_gru1_state_input")
_prev_gru2_layer = lasagne.layers.InputLayer([None,n_hid_2],name="prev_gru2_state_input")

#memory
gru1 = GRUMemoryLayer(n_hid_1,
                     _observation_layer,
                     _prev_gru1_layer,
                     name="gru1")

gru2 = GRUMemoryLayer(n_hid_2,
                     gru1,        #note that it takes CURRENT gru1 output as input.
                                  #replacing that with _prev_gru1_state would imply taking previous one.
                     _prev_gru2_layer,
                     name="gru2")

concatenated_memory = lasagne.layers.concat([gru1,gru2])

#q_eval
n_actions = len(feature_names)
q_eval = lasagne.layers.DenseLayer(concatenated_memory, #taking both memories. 
                                                        #Replacing with gru1 or gru2 would mean taking one
                                   num_units = n_actions,
                                   nonlinearity=lasagne.nonlinearities.linear,name="QEvaluator")
#resolver
epsilon = theano.shared(np.float32(0.0),"e-greedy.epsilon")

resolver = EpsilonGreedyResolver(q_eval,epsilon=epsilon,name="resolver")




from collections import OrderedDict
#all together
agent = Agent(_observation_layer,
              OrderedDict([
                    (gru1,_prev_gru1_layer),
                    (gru2,_prev_gru2_layer)
              ]),
              [q_eval,concatenated_memory],resolver)





##load weights from snapshot

snapshot_path ="./demo_stand.qlearning_3_step.epoch60000.pcl"
snapshot_url = "https://www.dropbox.com/s/vz4hz5tpm0u2zkw/demo_stand.qlearning_3_step.epoch60000.pcl?dl=1"

from agentnet.utils import load
if not os.path.isfile(snapshot_path):
    print "loading snapshot..."
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    urlretrieve(snapshot_url,snapshot_path)



load(resolver,snapshot_path)

epsilon.set_value(0)



##one-turn response function
prev_memory_state = T.vector("prev_mem_state_input",dtype=floatX)
observation = T.vector("observation_input",dtype=floatX)



prev_memory_tensor = prev_memory_state.reshape([1,-1])
prev_gru1 = prev_memory_tensor[:,:n_hid_1]
prev_gru2 = prev_memory_tensor[:,n_hid_1:]

observation_tensor = observation.reshape([1,-1])


action_channels,_, outputs, = agent.get_agent_reaction(
    {
        gru1:prev_gru1,
        gru2:prev_gru2
    },observation_tensor,)

action_tensor = action_channels[0]
Qvalues_tensor =  outputs[0]
new_state_tensor = outputs[1] #concatenated memory


new_state, Qvalues, action = new_state_tensor[0],Qvalues_tensor[0],action_tensor[0]


get_agent_reaction = theano.function([prev_memory_state,observation], [new_state,Qvalues,action])



##auxilary transformer function
def response_to_observation(response,prev_action,is_alive=True):
    """creates a wikicat-format observation from"""
    
    observation = np.zeros(env.observation_size,dtype=floatX)
    
    observation[0] = response
    
    observation[1] = is_alive
    
    observation[2+prev_action] = 1
    return observation




#special states
zero_memory_state = np.zeros(concatenated_memory.output_shape[1],dtype='float32')
fake_observation = np.zeros(env.observation_size,dtype='float32')



