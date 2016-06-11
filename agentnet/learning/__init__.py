"""
This module contains implementations of various reinforcement learning algorithms.

The core API of each learning algorithm is .get_elementwise_objective that returns the
per-tick loss that you can minimize over NN weights using e.g. lasagne.updates.your_favorite_method.
"""

from __future__ import division, print_function, absolute_import
