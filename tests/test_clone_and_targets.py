import lasagne
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, get_output

from agentnet.target_network import TargetNetwork


def test_targetnet():
    l_in = InputLayer([None, 10])
    l_d0 = DenseLayer(l_in, 20, name='d0')
    l_d1 = DenseLayer(l_d0, 30, name='d1')
    l_d2 = DenseLayer(l_d1, 40, name='d2')
    other_l_d2 = DenseLayer(l_d1, 41, name='d2_other')

    # test full clone
    full_clone = TargetNetwork([l_d2, other_l_d2], name='target_nn')

    layer_names = [l.name for l in lasagne.layers.get_all_layers(full_clone.output_layers)]
    param_names = [p.name for p in lasagne.layers.get_all_params(full_clone.output_layers)]

    assert len(param_names) == 8
    assert len(list(filter(lambda name: name.startswith('target_nn'), param_names))) == 8

    assert len(layer_names) == 5
    assert len(list(filter(lambda name: (name or '').startswith('target_nn'), layer_names))) == 4

    # test partial clone
    full_clone = TargetNetwork([l_d2, other_l_d2], l_d1, name='target_nn')

    layer_names = [l.name for l in lasagne.layers.get_all_layers(full_clone.output_layers)]
    param_names = [p.name for p in lasagne.layers.get_all_params(full_clone.output_layers)]

    assert len(param_names) == 8
    assert len(list(filter(lambda name: name.startswith('target_nn'), param_names))) == 4

    assert len(layer_names) == 5
    assert len(list(filter(lambda name: (name or '').startswith('target_nn'), layer_names))) == 2

    full_clone.load_weights()
    full_clone.load_weights(0.33)


from agentnet.utils.clone import clone_network
import theano.tensor as T


def test_clone():
    dummy = T.zeros([3, 10])
    l_in = InputLayer([None, 10], dummy)
    l_d0 = DenseLayer(l_in, 20, name='d0')
    l_d1 = DenseLayer(l_d0, 30, name='d1')
    l_d2 = DenseLayer(l_d1, 40, name='d2')
    l_d3 = DenseLayer(l_d2, 41, name='d3')

    original_params = lasagne.layers.get_all_params(l_d3)

    # full clone including params with added name prefix
    full_clone = clone_network(l_d3, name_prefix='cloned_net.')

    assert full_clone.name == 'cloned_net.' + l_d3.name
    assert full_clone.W.name == 'cloned_net.' + l_d3.W.name
    full_clone_params = lasagne.layers.get_all_params(full_clone)
    for par, clone_par in zip(original_params, full_clone_params):
        assert par != clone_par

    # clone with shared params
    shared_clone = clone_network(l_d3, share_params=True)
    shared_clone_params = lasagne.layers.get_all_params(shared_clone)
    for par, clone_par in zip(original_params, shared_clone_params):
        assert par == clone_par

    lasagne.layers.get_all_params(shared_clone)

    partial_clone = clone_network(l_d3, bottom_layers=l_d1)

    assert partial_clone.input_layer.input_layer == l_d1
    assert partial_clone.input_layer.input_layer.input_layer == l_d0

    # build a fake second dense layer and make clone that takes it instead of original one
    l_in_alt = InputLayer([None, 10], dummy)
    l_d1_alt = DenseLayer(l_in_alt, 30)
    substitute_clone = clone_network(l_d3, bottom_layers={l_d1: l_d1_alt})

    assert substitute_clone.input_layer.input_layer == l_d1_alt
    assert substitute_clone.input_layer.input_layer.input_layer == l_in_alt

    # make sure one can get output from any of these nets
    for nn in [full_clone, shared_clone, partial_clone, substitute_clone]:
        lasagne.layers.get_output(nn).eval()

from agentnet.utils.layers.reapply import reapply

def test_reapply():
    
    l_in1 = InputLayer([None,10],T.zeros([5,10]))
    l_d1 = DenseLayer(l_in1,20)
    l_d2 = DenseLayer(l_in1,30)
    l_cat = ConcatLayer([l_d1,l_d2])
    l_d3 = DenseLayer(l_cat,20)
    
    l_in2 = InputLayer([None,10],T.zeros([5,10]))
    
    new_l_d3 = reapply(l_d3,{l_in1:l_in2})  #reapply the whole network to a new in 
    get_output(new_l_d3).eval()

    l_in3 = InputLayer([None,30],T.zeros([5,30]))
    
    new_l_d3 = reapply(l_d3,{l_d2:l_in3})
    
    #multiple inputs
    new_l_cat = reapply(l_cat,{l_d2:ConcatLayer([l_d1,l_d2]),l_in1:l_in2})
    get_output(new_l_cat).eval()
    
    #multiple layers
    l1,l2 = reapply([l_d3,l_d2],{l_in1:l_in2})
    outs = reapply({'d3':l_d3,'d2':l_d2},{l_in1:l_in2})
    
    assert isinstance(outs,dict)
    
    outs['d3'],outs['d2']
    
    
