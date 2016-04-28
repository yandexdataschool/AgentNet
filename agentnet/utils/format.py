__doc__="""a few auxilary methods that work with supported collection formats"""

import numpy as np
from collections import OrderedDict
from warnings import warn


supported_sequences = (tuple,list)

def check_list(variables,target_length=None):
    """ensure that variables is a sequence of a supported type"""
    if type(variables) in supported_sequences:
        return variables
    else:
        #if it is a numpy or theano array, excluding numpy array of objects
        if hasattr(variables,'shape'):
            if hasattr(variables,'dtype'):
                if variables.dtype != np.object:
                    return [variables]
                
                
        #elif it is a different kind of sequence
        if hasattr(variables,'__iter__'):
            #try casting to tuple. If cannot, treat that it will be treated as an atomic object
            try:
                if target_length is not None and len(variables) != target_length:
                    raise "shapes do not match"

                casted_variables = tuple(variables)
                
                warn(str(variables)+ " of type "+type(variables)+ " will be treated as a sequence of "+\
                     len(casted_variables) +"elements, not a single element. If you want otherwise, please"\
                     " pass it as a single-element list/tuple")
                return casted_variables
            except:
                warn(str(variables)+ " of type "+type(variables)+ " will be treated as a single input/output tensor,"\
                    "and not a collection of such. If you want otherwise, please cast it to list/tuple")
                
        return [variables] 
    
    
def check_ordict(variables):
    """ensure that variables is an OrderedDict"""
    assert hasattr(variables,"items") 
    try:
        return OrderedDict(variables.items())
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