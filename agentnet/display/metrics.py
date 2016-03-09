
__doc__="""An auxilary class used to store and plot several [time-related] metrics on one plot"""

#Example usage:
#mm = Metrics()
#mm["y"][0] = 1
#mm["y"][1] = 2
#mm["z"][0] = 2
#mm["z"][1] = 1.5
#mm["z"][2.5] = 1.4
#mm.plot()

import matplotlib.pyplot as plt
from collections import defaultdict

class Metrics(defaultdict):
    def __init__(self):
        defaultdict.__init__(self,dict)
    def plot(self,title="metrics",figsize=[10,10],legend_loc = 'best',show_afterwards = True):
        
        plt.figure(figsize = figsize)
        
        for metric_name, metric_dict in self.items():            
            plt.plot(*zip(*sorted(metric_dict.items())),label = metric_name)

        plt.title(title)
        plt.grid()
        plt.legend(loc=legend_loc)
        if show_afterwards:
            plt.show()
