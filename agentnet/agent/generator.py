from warnings import warn




class Generator(object):
    def __init__(self,*args, **kwargs):

        warn("Generator has been replaced with a more general and powerful Recurrence layer. See tutorial on stack RNN on how to use it")
        raise NotImplementedError