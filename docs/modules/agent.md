agent and recurrence
============

AgentNet core abstraction is Recurrence - a lasagne container-layer that can hold
  arbitrary graph and roll it for specified number of steps.

An Agent (MDP Agent) is user-friendly interface that implements Markov Decision Process
  kind of interaction via recurrence (environment->agent->environment->agent->...).

.. automodule:: agentnet.agent

.. currentmodule:: agentnet.agent

.. class:: Agent
   Alias for MDPAgent

.. autoclass:: MDPAgent
   :members:
   :member-order: bysource

.. autoclass:: Recurrence
   :members:
   :member-order: bysource