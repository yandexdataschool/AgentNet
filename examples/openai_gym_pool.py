__doc__="""
a thin wrapper for openAI gym environments that maintains a set of parallel games and has a method to generate interaction sessions
given agent one-step applier function
"""

import gym
import numpy as np

#A whole lot of space invaders
class GamePool:
    def __init__(self,game_title,n_games):
        """
        A pool that stores several
           - game states (gym environment)
           - prev_observations - last agent observations
           - prev memory states - last agent hidden states
           
       """
        
        
        self.ataries = [gym.make(game_title) for i in range(n_games)]

        self.prev_observations = [atari.reset() for atari in self.ataries]
    
        self.prev_memory_states = 'zeros'
    
    def interact(self,agent_step,n_steps = 100,verbose=False):
        """generate interaction sessions with ataries (openAI gym atari environments)
        Sessions will have length n_steps. 
        Each time one of games is finished, it is immediately getting reset
        
        
        params:
            agent_step: a function(observations,memory_states) -> actions,new memory states for agent update
            n_steps: length of an interaction
            verbose: if True, prints small debug message whenever a game gets reloaded after end.
        returns:
            observation_log,action_log,reward_log,[memory_logs],is_alive_log,info_log
            a bunch of tensors [batch, tick, size...]
            
            the only exception is info_log, which is a list of infos for [time][batch]
                
            
        
        """
        history_log = []

        for i in range(n_steps):

            actions,new_memory_states = agent_step(self.prev_observations,self.prev_memory_states)

            new_observations, cur_rewards, is_done, infos = \
                zip(*map(
                        lambda atari, action: atari.step(action), 
                        self.ataries,
                        actions)
                   )

            new_observations = np.array(new_observations)

            for i in range(len(self.ataries)):
                if is_done[i]:
                    new_observations[i] = self.ataries[i].reset()

                    for m_i in range(len(new_memory_states)):
                        new_memory_states[m_i][i] = 0

                    if verbose:
                        print("atari %i reloaded"%i)


            #append observation -> action -> reward tuple
            history_log.append((self.prev_observations,actions,cur_rewards,new_memory_states,is_done,infos))

            self.prev_observations = new_observations
            self.prev_memory_states = new_memory_states

            
            
        #cast to numpy arrays
        observation_log,action_log,reward_log,memories_log,is_done_log,info_log = zip(*history_log)
        
        #tensor dimensions    
        # [batch_i, time_i, observation_size...]
        observation_log = np.array(observation_log).swapaxes(0,1)
        
        # [batch, time, units] for each memory tensor
        memories_log = map(lambda mem: np.array(mem).swapaxes(0,1),zip(*memories_log))
                                                                                    
        # [batch_i,time_i]
        action_log = np.array(action_log).swapaxes(0,1)
    
        # [batch_i, time_i]
        reward_log = np.array(reward_log).swapaxes(0,1)
        
        # [batch_i, time_i]
        is_alive_log = 1- np.array(is_done_log,dtype = 'int8').swapaxes(0,1)
    
        
        
        
        return observation_log,action_log,reward_log,memories_log,is_alive_log,info_log