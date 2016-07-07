Environment
============

.. automodule:: agentnet.environment

.. currentmodule:: agentnet.environment

BaseEnvironment
----------------

.. autoclass:: BaseEnvironment
   :members:
   :member-order: bysource

Experience Replay
------------------

.. autoclass:: SessionPoolEnvironment
  :members: load_sessions, append_sessions, get_session_updates, select_session_batch, sample_session_batch
  :member-order: bysource

.. autofunction:: SessionBatchEnvironment