__doc__="""a few auxilary methods that work with supported collection formats"""

import numpy as np
from collections import OrderedDict
from warnings import warn


supported_sequences = (tuple,list)

def check_list(variables):
    """ensure that variables is a sequence of a supported type"""
    if type(variables) not in supported_sequences:
        if hasattr(variables,'__iter__'):
            #other sequence type
            warn(str(variables)+ " will be treated as a single input/output tensor, and not a collection of such."\
                 "If you want otherwise, please cast it to list/tuple")
        #non-sequence
        variables = [variables]
    return variables

def check_ordict(variables):
    """ensure that variables is an OrderedDict"""
    try:
        return OrderedDict(variables)
    except:
        raise ValueError, "Could not convert "+variables+"to an ordered dictionary"

def unpack_list(a, *lengths):
    """
    Returns slices of the input list a.
    
    unpack_list(a,2,3,5) -> a[:2], a[2:2+3], a[2+3:2+3+5]
    """
    borders = np.concatenate([[0],np.cumsum(lengths)])
    
    groups = []
    for low,high in zip(borders[:-1],borders[1:]):
        groups.append(a[low:high])
    
    return groups