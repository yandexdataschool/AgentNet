from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from collections import defaultdict


class Metrics(defaultdict):
    """
      An auxiliary class used to store and plot several [time-related] metrics on a single plot

      Example usage:
      >>> mm = Metrics()
      >>> mm["y"][0] = 1
      >>> mm["y"][1] = 2
      >>> mm["z"][0] = 2
      >>> mm["z"][1] = 1.5
      >>> mm["z"][2.5] = 1.4
      >>> plt.figure(figsize=[10, 10])
      >>> mm.plot()
      >>> plt.show()
    """
    def __init__(self):

        defaultdict.__init__(self, dict)

    def plot(self,
             title="metrics",
             legend_loc='best',
             show=True,):
        for metric_name, metric_dict in list(self.items()):
            plt.plot(*list(zip(*sorted(metric_dict.items()))), label=metric_name)

        plt.title(title)
        plt.grid()
        plt.legend(loc=legend_loc)
        if show:
            plt.show()