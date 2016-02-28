__doc__="""several more-or-less successfull layer designs used in the lasagne NN architecture"""


import numpy as np
import theano
import theano.tensor as T
import lasagne

def ChanelwiseSoftmaxLayer( _network,
                            neurons_per_channel = 50,
                            name_prefix='channelwise_softmax_dense.',
                           channels = ['x','y','z']
    ):
    _dense_outputs_list = [
    lasagne.layers.DenseLayer(_network, name = name_prefix+channel_name,
                                 num_units=neurons_per_channel,
                                 W=lasagne.init.GlorotUniform(),                     
                                 nonlinearity=lasagne.nonlinearities.softmax)
    for channel_name in channels]



    return lasagne.layers.ConcatLayer(_dense_outputs_list)


from auxilary import _normalize
def ChanelwiseNormalizedLayer( _network,
                            neurons_per_channel = 50,
                            name_prefix='channelwise_normalized_dense.',
                              channels = ['x','y','z']
    ):
    _dense_outputs_list = []
    
    for channel_name in channels:
        _layer  = lasagne.layers.DenseLayer(_network, name = name_prefix+channel_name,
                                     num_units=neurons_per_channel,
                                     W=lasagne.init.GlorotUniform(),                     
                                     nonlinearity=lasagne.nonlinearities.sigmoid)
        _layer = lasagne.layers.ExpressionLayer(_layer,_normalize)
        _dense_outputs_list.append(_layer)


    return lasagne.layers.ConcatLayer(_dense_outputs_list)





class LocalInhibitionLayer1D(lasagne.layers.Layer):
    """
    This code is adapted from pylearn2.
    """

    def __init__(self, incoming,n=3, k=0.,d=1., alpha=1,**kwargs):
        """
        Local Response Normalization for over specified axis.
        Aggregation is purely across LAST axis

        :parameters:
            - incoming: input layer or shape (tensor4)
            - alpha: coefficient for x in the exponent
            - k: laplacian smoothing term for exp(x) (added to exp(x_i), n*k added to denominator) 
            - d: denominator additive constant
            - n: number of adjacent channels to normalize over (including channel itself)
        """
        super(LocalInhibitionLayer1D, self).__init__(incoming,
                                                                **kwargs)
        self.alpha = alpha
        self.k = k
        self.n = n
        self.d = d
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, x, **kwargs):
        input_shape = self.input_shape
        
        if any(s is None for s in input_shape):
            input_shape = x.shape
        
        half_n = self.n // 2
        
        exp_x = T.exp(x*self.alpha) + self.k
        
        b, ch, r, c = input_shape
        
        #pad x
        extra_channels = T.alloc(self.k, b, ch , r, c+ 2*half_n)
        
        exp_x_padded = T.set_subtensor(extra_channels[:,:, :, half_n:half_n+c],
                                    exp_x)
        denominator = self.k*self.n + self.d
        for i in range(self.n):
            denominator += exp_x_padded[:, :, :, i:i+c]
        
        return exp_x/denominator
    
    
def ChannelwiseLocalInhibitionLayer(_nn,region_halfsize=2,
                                    denominator_addition=0.0001,
                                    denominator_power = 1,
                                    alpha = 'auto'):
    """takes BC01-like layer as an input (that means 4 dimensions: sample_i,channel_i,x,y)
    applies nonlinearity:
    $$ x_i = \frac{x_i}{ (k + ( \alpha \sum_j x_j^2 ))^\beta } $$
    to each channel along LAST axis for subregions consisting of x_i and region_halfsize neighbors
    to the left and right sides, totalling 2*region_halfsize+1
    where beta is denominator_power, k is denominator_addition
    """
    
    if denominator_addition ==0:
        print "warning: denominator additive member set to zero. May return NANs for regions with all zeroes"
        
    _nn = lasagne.layers.DimshuffleLayer(_nn,[0,3,1,2])

    region_width = region_halfsize*2+1

    if alpha == 'auto':
        alpha = (region_width - denominator_addition)/region_width
    _nn = lasagne.layers.LocalResponseNormalization2DLayer(_nn, 
                                                           n=region_width, #n = normalization region size
                                                           k = denominator_addition, #additive constant
                                                           alpha = alpha, #coef
                                                           beta=denominator_power,#denominator power
                                                           )
    _nn = lasagne.layers.DimshuffleLayer(_nn,[0,2,3,1])
    return _nn   


_tanh_zero_one = lambda x: 0.5+lasagne.nonlinearities.ScaledTanH(1,0.5)(x)
_tanh_leaky = lambda x:0.05*x+lasagne.nonlinearities.tanh(x)

def ChannelwiseDenseLayer(_nn,name_prefix = "channel",
                     units_per_channel=256, nonlinearity = _tanh_leaky):
    """applies dense layers to each channel separately,than concatenates them"""
    retina_channel_layers = []
    for i in range(_nn.output_shape[1]):
        _layer = lasagne.layers.SliceLayer(_nn,name = name_prefix+"."+str(i)+".slice",
                                           indices=i,axis = 1)
        _layer = lasagne.layers.DenseLayer(_nn,name = name_prefix+"."+str(i)+".dense",
                                          num_units= units_per_channel,
                                          nonlinearity= nonlinearity,
        )
        retina_channel_layers.append(_layer)
    _nn = lasagne.layers.ConcatLayer(retina_channel_layers)
    return _nn

#gradient reversal layer by Daniel Renshaw 
#http://stackoverflow.com/users/127480/daniel-renshaw
#thanks to him, but idk if it works :P
class ReverseGradient(theano.gof.Op):
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]
    
#todo: debugme
class ReverseGradientLayer(lasagne.layers.Layer):
    def __init__(self, incoming, hp_lambda, **kwargs):
        super(ReverseGradientLayer, self).__init__(incoming, **kwargs)
        self.op = ReverseGradient(hp_lambda)

    def get_output_for(self, input, **kwargs):
        return self.op(input)

    
