__doc__="""a few auxilary methods that work with supported collection formats"""

import numpy as np
from collections import OrderedDict
from warnings import warn


supported_sequences = (tuple,list)

def check_sequence(variables):
    """ensure that variables is one of supported_sequences or converts to one.
    If naive conversion fails, throws an error"""
    if any( isinstance(variables,seq) for seq in supported_sequences):
        return variables
    else:
        #if it is a numpy or theano array, excluding numpy array of objects, return a list with single element
        # yes, i know it's messy. Better options are welcome for pull requests :)
        if hasattr(variables,'shape'):
            if hasattr(variables,'dtype'):
                if variables.dtype != np.object:
                    return [variables]
                
                
        #elif it is a different kind of sequence
        if hasattr(variables,'__iter__'):
            #try casting to tuple. If cannot, treat that it will be treated as an atomic object
            try:
                casted_variables = list(variables)
                
                warn(str(variables)+ " of type "+str(type(variables))+ " will be treated as a sequence of "+\
                     len(casted_variables) +"elements, not a single element. If you want otherwise, please"\
                     " pass it as a single-element list/tuple")
                return casted_variables
            except:
                warn(str(variables)+ " of type "+str(type(variables))+ " will be treated as a single "\
                     "input/output tensor, and not a collection of such. If you want otherwise, please cast it"\
                     "to list/tuple")
                
        #otherwise it's a one-element list
        return [variables] 

def check_list(variables):
    """ensure that variables is a list or converts to one.
    If naive conversion fails, throws an error"""
    return list(check_sequence(variables))
                
def check_tuple(variables):
    """ensure that variables is a list or converts to one.
    If naive conversion fails, throws an error"""
    return tuple(check_sequence(variables))
    
def check_ordict(variables):
    """ensure that variables is an OrderedDict"""
    assert isinstance(variables,dict) 
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