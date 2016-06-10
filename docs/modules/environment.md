environment
============

blah

.. automodule:: agentnet.environment

.. currentmodule:: agentnet.environment

BaseEnvironment
----------------
detailed walkthrough how to create it

.. autoclass:: BaseEnvironment
   :members:
   :member-order: bysource

Experience Replay
-----------------

what and why


.. autoclass:: SessionPoolEnvironment
  :members: load_sessions, append_sessions, session_pool, get_session_updates, select_session_batch, sample_session_batch
  :member-order: bysource

.. autofunction:: SessionBatchEnvironment