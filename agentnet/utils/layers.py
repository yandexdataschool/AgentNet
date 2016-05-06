

__doc__ = """Here you can find a number of auxilary lasagne layers that are used throughout AgentNet and examples"""


from lasagne.layers import Layer, MergeLayer, ExpressionLayer


        
        


class TupleLayer(MergeLayer):
    """An abstract base class for Lasagne layer that returns several outputs.
    Has to implement get_output_shape so that it contains a list/tuple of output shapes(tuples),
    one for each output.
    In other words, if you return 2 elements of shape (None, 25) and (None,15,5,7),
    self.get_output_shape must return [(None,25),(None,15,5,7)]
    """
    
    @property
    def element_names(self):
        name = (self.name or "tuple layer")
        return [name+"[{}]".format(i) 
                for i in range(len(self.output_shape))]
    
    def __getitem__(self, ind):
        assert type(ind) in (int, slice)
        
        if type(ind) is slice:            
            return list(self)[ind]
        
        return ExpressionLayer(self,lambda values:values[ind], 
                               output_shape = lambda shapes:shapes[ind],
                               name = self.element_names[ind])
    
    def __iter__(self):
        return (self[i] for i in range(len(self.output_shape)))
    

    
    
    def get_output_shape_for(self,input_shape):
        """
        One must implement this method when inheriting from TupleLayer.
        TupleLayer's shape must be a sequence of shapes for each output (depth-2 list or tuple)
        """
        raise NotImplementedError
        
        
        
# <- implement rev_grad here?
        
        