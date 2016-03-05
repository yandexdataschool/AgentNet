__doc__="All the user-specific configuration is stored here"
snapshot_path = "/home/jheuristic/yozhik/agentnet_snapshots/"
import os
try:
    os.system("mkdir {}".format(snapshot_path))
except:
    pass

print "AgentNet examples will store and seek thier snapshots at",snapshot_path

