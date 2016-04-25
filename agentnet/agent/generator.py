import lasagne                 
from lasagne.utils import unroll_scan
from theano import tensor as T
from ..utils import insert_dim

from base import BaseAgent
from ..environment.session_batch import SessionBatchEnvironment
from ..environment.feedback import  FeedbackEnvironment


from ..utils.format import supported_sequences,check_list




class Generator(BaseAgent):
    def __init__(self,
                 observation_layers,
                 memory_dict,
                 policy,
                 resolver,
                ):
        """
        A sequence generator that recurrently spawns sequence elements and observes them as his input next turn,
            memory_dict - OrderedDict{ memory_output: memory_input}, where
                memory_output: lasagne layer
                    - generates first (a-priori) agent state
                    - determines new agent state given previous agent state and an observation
                    
                memory_input: lasagne.layers.InputLayer that is used as "previous state" input for memory_output
            policy - lasagne.Layer child instance that
                - determines Q-values or probabilities for all actions given current agent state and current observation,
                - can .get_output_for(hidden_state)
            resolver - resolver.BaseResolver child instance that
                - determines agent's action given Q-values for all actions
        """        
        
        self.single_resolver = type(resolver) not in supported_sequences
        self.single_policy = type(policy) not in supported_sequences
        self.single_observation = type(observation_layers) not in supported_sequences

        
        super(Generator, self).__init__(observation_layers,resolver,memory_dict,policy)
        

    def get_sessions(self, 
                     session_length = 10,
                     batch_size = None,
                     recorded_sequences = None,
                     initial_state_variables = 'zeros',
                     initial_actions = 'zeros',
                     additional_output_layers = [],
                     **flags
                     ):
        """returns history of agent-generated sequences for given number of turns:
        parameters:
            session_length - how many turns of interaction shall there be for each batch
            batch_size - [required parameter] amount of independed sessions [number or symbolic].Irrelevant if you manually set all initial_*.
            
            recorded_sequence - if None, generator is actually generating output.
                if a tensor[batch_i,time_tick,...] is passed instead, the generator observes this sequence 
                    instead of it's own output
            
            initial_<something> - initial values for all variables at 0-th time step
            Unless you are doing something nasty, initial policy and actions will not matter at all
            'zeros' default means filling variable with zeros
            Initial values are NOT included in history sequences
            additional_output_layers - any layers of a network which outputs need to be added to the outputs
            flags: optional flags to be sent to NN when calling get_output (e.g. deterministic = True)


        returns:
            hidden_seq,policy_seq,action_seq, [additional_output_0, additional_output_1]
            for hidden state, agent policy and chosen actions respectively
            each of them having dimensions of [batch_i,seq_i,...]
            
            
        """
        
        if recorded_sequences is None:
            environment = FeedbackEnvironment()
        else:
            recorded_sequences = check_list(recorded_sequences)
            environment = SessionBatchEnvironment(recorded_sequences)
        
        
        groups = super(Generator, self).get_sessions(environment=environment,
                                                  session_length=session_length,
                                                  batch_size = batch_size,
                                                  initial_state_variables=initial_state_variables,
                                                  initial_observations = initial_actions, #feeding initial_action as observation
                                                  **flags
                                                  )
        
        env_states,observations,agent_states,actions,policy = groups
        

        
        
        if type(environment.state_size) not in supported_sequences:
            env_states = env_states[0]
        if self.single_observation:
            observations = observations[0]
        if self.single_resolver:
            actions = actions[0]
        if self.single_policy:
            policy = policy[0]
            
        return env_states,observations,agent_states,actions,policy
        
        
        
        
        if initial_hidden == 'zeros':
            memory_state_shape = lasagne.layers.get_output_shape(self.memory)[1:]
            initial_hidden = T.zeros((batch_size,)+tuple(memory_state_shape))
        
        time_ticks = T.arange(session_length)

        
        
        
        #recurrent step functions
        def step_active(time_tick,last_hidden,last_policy,last_action,
                 *args):
            """a recurrent step function where generator actually generates sequence"""

            hidden,policy,action,additional_outputs = self.get_agent_reaction(last_hidden,last_action,
                                                                               additional_output_layers,**flags)
            return [hidden,policy,action]+additional_outputs
        
        def step_passive(time_tick,current_observation,last_hidden,last_policy,last_action,
                 *args):
            """a recurrent step function where generator observes recorded sequence of actions and generates
            possible next steps for recorded sequence prefices. Used for passive training (like language model)"""
            hidden,policy,action,additional_outputs = self.get_agent_reaction(last_hidden,current_observation,
                                                                               additional_output_layers,**flags)
            return [hidden,policy,action]+additional_outputs


        ##main recurrence loop
        
        #state 0 values
        additional_init = [None for i in additional_output_layers]
        outputs_info = [initial_hidden,None,initial_actions] + additional_init
        
        #time ticks and [optional] transposed recorded sequence [tick,batch,...]
        sequences = [time_ticks]
        if recorded_sequence is not None:
            sequences.append(recorded_sequence.swapaxes(1,0))
        
        
        step = step_active if recorded_sequence is None else step_passive
        
        history = unroll_scan(step,
            sequences = sequences,
            outputs_info = outputs_info,
            non_sequences = [],
            n_steps = session_length
        )
        
        
        self.history = history
        
        #from [time,batch,...] to [batch,time,...]
        history = [ (var.swapaxes(1,0) if var.ndim >1 else var) for var in history]
        
        #what's inside:
        hidden_seq,policy_seq,action_seq = history[:3]
        
        additional_output_sequences = tuple(history[3:])
        
        
        
        return (hidden_seq,policy_seq,action_seq) + additional_output_sequences
                 