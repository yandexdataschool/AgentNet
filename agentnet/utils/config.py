"""Agentnet configuration object, similar to theano.config. Hopefully it remains small"""


###Warnings verbosity:
class config:

    verbose=2
    # 0 = shut up
    # 1 = essential only
    # 2 = tell me everything

    #verbosity functions for fun
    @staticmethod
    def shut_up():
        """sets agentnet.verbose to 0"""
        config.verbose = 0
