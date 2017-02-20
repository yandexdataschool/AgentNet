__doc__= """Here lie parts of architecture that are no longer supported and will be removed in one of the following versons"""


from warnings import warn as default_warn
from .config import config

def warn(message="Hello, world!",verbosity_level=1,**kwargs):
    """issues a warning if verbosity_level is above current verbosity level
    :param message: what to warn about
    :verbosity_level: an integer, lower means more important warning
    :param kwargs: whatever else you want to send to warnings.warn function
    """
    if config.verbose >= verbosity_level:
        default_warn("[Verbose>=%s] %s"%(verbosity_level,message),**kwargs)

    #TODO make unsuppressable somehow


class deprecated:
    def __init__(self,new_name=None,removed_after="next major patch"):
        self.new_name = new_name
        self.removed_after = removed_after
    def __call__(self,func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emmitted
        when the function is used."""
        def newFunc(*args, **kwargs):
            warning = "%s is deprecated and will be removed in %s. " % (func.__name__,self.removed_after)
            if self.new_name is not None:
                warning += "This functionality has been replaced with %s." % self.new_name
            warn(warning,category=DeprecationWarning)
            return func(*args, **kwargs)
        newFunc.__name__ = func.__name__
        newFunc.__doc__ = func.__doc__
        newFunc.__dict__.update(func.__dict__)
        return newFunc
