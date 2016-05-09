

__doc__ = """Here you can find a number of auxilary lasagne layers that are used throughout AgentNet and examples"""


from lasagne.layers import Layer, MergeLayer, ExpressionLayer
import theano

        

def get_layer_dtype(layer, default=None):
    """ takes layer's output_dtype property if it is defined, 
    otherwise defaults to default or (if it's not given) theano.config.floatX"""
    return layer.output_dtype if hasattr(layer,"output_dtype") else default or theano.config.floatX


class TupleLayer(MergeLayer):
    """An abstract base class for Lasagne layer that returns several outputs.
    Has to implement get_output_shape so that it contains a list/tuple of output shapes(tuples),
    one for each output.
    In other words, if you return 2 elements of shape (None, 25) and (None,15,5,7),
    self.get_output_shape must return [(None,25),(None,15,5,7)]
    """
    
    def get_output_shape_for(self,input_shape):
        """
        One must implement this method when inheriting from TupleLayer.
        TupleLayer's shape must be a sequence of shapes for each output (depth-2 list or tuple)
        """
        raise NotImplementedError

    
    @property
    def element_names(self):
        name = (self.name or "tuple layer")
        return [name+"[{}]".format(i) 
                for i in range(len(self.output_shape))]
    
    @property
    def output_dtype(self):
        """ dtypes of tuple outputs"""
        return [theano.config.floatX for i in range(len(self.output_shape))]

    
    def __len__(self):
        """an amount of output layers in a tuple"""
        return len(self.output_shape)
    
    def __getitem__(self, ind):
        """ returns a lasagne layer that yields i-th output of the tuple.
        parameters:
            ind - an integer index or a slice.
        returns:
            a particular layer of the tuple or a list of such layers if slice is given 
        """
        assert type(ind) in (int, slice)
        
        if type(ind) is slice:            
            return list(self)[ind]
        
        assert type(ind) in (int,long)
        
        item_layer = ExpressionLayer(self,lambda values:values[ind], 
                               output_shape = lambda shapes:shapes[ind],
                               name = self.element_names[ind])
        
        item_layer.output_dtype = self.output_dtype[ind]
        
        return item_layer
    
    def __iter__(self):
        """ iterate over tuple output layers"""
        return (self[i] for i in range(len(self.output_shape)))
    
        


        
# <- implement rev_grad here?
        
        