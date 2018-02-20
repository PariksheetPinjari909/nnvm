"""
Compile DarkNet Models
====================
**Author**: `Siju Samuel <https://github.com/siju-samuel/>`_

DarkNet symbol frontend.

"""

from __future__ import absolute_import as _abs
import math
import numpy as np
import cv2
import tvm
from nnvm.frontend.darknet_c_interface import __darknetffi__
from nnvm.frontend.darknet_c_interface import ACTIVATION
from nnvm.frontend.darknet_c_interface import LAYERTYPE

from .. import symbol as _sym

__all__ = ['from_darknet']

def _darknet_get_nnvm_op(op_name):
    """Get the nnvm operation from opname, raise error if not supported."""
    op = getattr(_sym, op_name)
    if not op:
        raise RuntimeError("Not to map op_name {} to nnvm.sym".format(op_name))
    return op

def _darknet_required_attr(attr, key):
    """Check the attribute exists and return if exists, if not return error."""
    assert isinstance(attr, dict)
    if key not in attr:
        raise AttributeError("Required attribute {} not found.".format(key))
    return attr[key]

def _darknet_raise_not_supported(attr, op='nnvm'):
    """Raise error if any operation is not supported."""
    err = "{} is not supported in {}.".format(attr, op)
    raise NotImplementedError(err)

def _darknet_warn_not_used(attr, op='nnvm'):
    """Raise warning if any operation not supported."""
    import warnings
    err = "{} is ignored in {}.".format(attr, op)
    warnings.warn(err)

def _darknet_parse_tshape(tshape):
    """Parse tshape in string."""
    return [int(x.strip()) for x in tshape.strip('()').split(',')]

def _darknet_parse_bool_str(attr, key, default='False'):
    """Parse bool string to boolean."""
    return attr.get(key, default).strip().lower() in \
                                    ['true', '1', 't', 'y', 'yes']

def _darknet_maxpooling(inputs, attrs):
    """Process the max pool 2d operation."""
    kernel = _darknet_parse_tshape(_darknet_required_attr(attrs, 'kernel'))
    if len(kernel) != 1:
        _darknet_raise_not_supported('non-2d kernel', 'pool_2d')

    op_name, new_attrs = 'max_pool2d', {}
    strides = int(attrs.get('stride', (1, 1)))
    pads = int(attrs.get('pad', (0, 0)))
    new_attrs['pool_size'] = [kernel[0], kernel[0]]
    new_attrs['strides'] = str((strides, strides))
    new_attrs['padding'] = str((pads, pads))

    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_avgpooling(inputs, attrs):
    """Process the average pool 2d operation."""
    kernel = _darknet_parse_tshape(_darknet_required_attr(attrs, 'kernel'))
    if len(kernel) != 1:
        _darknet_raise_not_supported('non-2d kernel', 'pool_2d')

    op_name, new_attrs = 'avg_pool2d', {}
    strides = int(attrs.get('stride', (1, 1)))
    pads = int(attrs.get('pad', (0, 0)))
    new_attrs['pool_size'] = [kernel[0], kernel[0]]
    new_attrs['strides'] = str((strides, strides))
    new_attrs['padding'] = str((pads, pads))

    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_batch_norm(inputs, attrs):
    """Process the batchnormalization operation."""
    op_name, new_attrs = 'darknet_batch_norm', {}
    new_attrs['axis'] = attrs.get('axis', 1)
    new_attrs['epsilon'] = attrs.get('eps', 0.000001)
    new_attrs['center'] = True
    new_attrs['scale'] = True
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_conv2d(inputs, attrs):
    """Process the convolution 2d operation."""
    kernel = _darknet_parse_tshape(_darknet_required_attr(attrs, 'kernel'))
    if len(kernel) != 1:
        _darknet_raise_not_supported('non 2d kernel', 'conv2d')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        _darknet_raise_not_supported('layout: ' + layout, 'conv2d')
    strides = int(attrs.get('stride', (1, 1)))
    pads = int(attrs.get('pad', (0, 0)))

    op_name, new_attrs = 'conv2d', {}
    new_attrs['channels'] = _darknet_required_attr(attrs, 'num_filter')
    new_attrs['kernel_size'] = [kernel[0], kernel[0]]
    new_attrs['strides'] = (strides, strides)
    new_attrs['padding'] = (pads, pads)
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    if attrs.get('use_batchNorm', False) is True:
        new_attrs['use_batchNorm'] = True
        new_attrs['eps'] = 0.000001
    if attrs.get('use_scales', False) is True:
        new_attrs['use_scales'] = True
    if attrs.get('use_bias', False) is True:
        new_attrs['use_bias'] = True
    sym = _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs)
    out_name = sym.list_output_names()[0].replace('_output', '')

    if 'activation' in attrs:
        new_attrs = {}
        new_attrs['activation'] = attrs['activation']
        new_attrs['slope'] = 0.1
        sym, _ = _darknet_activations(sym, new_attrs)
    return sym, out_name


def _darknet_conv2d_transpose(inputs, attrs):
    """Process the convolution 2d transpose operation."""
    if 'target_shape' in attrs:
        _darknet_raise_not_supported('target_shape', 'conv2d_transpose')
    kernel = _darknet_parse_tshape(_darknet_required_attr(attrs, 'kernel'))
    if len(kernel) != 2:
        _darknet_raise_not_supported('non-2d kernel', 'conv2d_transpose')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        _darknet_raise_not_supported('layout: ' + layout, 'conv2d_transpose')
    op_name, new_attrs = 'conv2d_transpose', {}
    new_attrs['channels'] = _darknet_required_attr(attrs, 'num_filter')
    new_attrs['kernel_size'] = kernel
    new_attrs['strides'] = attrs.get('stride', (1, 1))
    new_attrs['output_padding'] = attrs.get('adj', (0, 0))
    new_attrs['padding'] = attrs.get('pad', (0, 0))
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    new_attrs['use_bias'] = not _darknet_parse_bool_str(attrs, 'no_bias')
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_shortcut(inputs, attrs):
    """Process the shortcut operation."""
    op_name, new_attrs = 'shortcut', {}
    sym = _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs)
    out_name = sym.list_output_names()[0].replace('_output', '')
    if 'activation' in attrs:
        new_attrs['activation'] = attrs['activation']
        sym, _ = _darknet_activations(sym, new_attrs)
    return sym, out_name

def _darknet_dense(inputs, attrs):
    """Process the dense operation."""
    op_name, new_attrs = 'dense', {}
    new_attrs['units'] = _darknet_required_attr(attrs, 'num_hidden')

    if attrs.get('use_bias', False) is True:
        new_attrs['use_bias'] = True
    if attrs.get('use_flatten', False) is True:
        inputs[0] = _sym.flatten(inputs[0])
    sym = _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs)
    out_name = sym.list_output_names()[0].replace('_output', '')
    if 'activation' in attrs:
        new_attrs = {}
        new_attrs['activation'] = attrs['activation']
        sym, _ = _darknet_activations(sym, new_attrs)
    return sym, out_name

def _darknet_dropout(inputs, attrs):
    """Process the dropout operation, its a blank operation."""
    op_name, new_attrs = 'dropout', {}
    new_attrs['rate'] = attrs.get('p', 0.5)
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_reshape(inputs, attrs):
    """Process the reshape operation."""
    if _darknet_parse_bool_str(attrs, 'reverse'):
        _darknet_raise_not_supported('reverse', 'reshape')
    op_name, new_attrs = 'reshape', {}
    new_attrs['shape'] = _darknet_required_attr(attrs, 'shape')
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_softmax_output(inputs, attrs):
    """Process the softmax operation."""
    op_name, new_attrs = 'softmax', {}
    if _darknet_parse_bool_str(attrs, 'multi_output'):
        new_attrs['axis'] = 1

    if attrs.get('use_flatten', False) is True:
        inputs[0] = _sym.flatten(inputs[0])
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_route(inputs, attrs):
    """Process the route operation, which is equivalent to concat."""
    op_name = 'concatenate'
    new_attrs = {'axis': attrs.get('dim', 1)}
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_reorg(inputs, attrs):
    """Process the reorg operation."""
    op_name, new_attrs = 'reorg', {}
    if 'stride' in attrs:
        new_attrs = {'stride': attrs.get('stride', 1)}
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_region(inputs, attrs):
    """Process the region operation."""
    op_name, new_attrs = 'region', {}
    if 'n' in attrs:
        new_attrs['n'] = attrs.get('n', 1)
    if 'classes' in attrs:
        new_attrs['classes'] = attrs.get('classes', 1)
    if 'coords' in attrs:
        new_attrs['coords'] = attrs.get('coords', 0)
    if 'background' in attrs:
        new_attrs['background'] = attrs.get('background', 0)
    if 'softmax' in attrs:
        new_attrs['softmax'] = attrs.get('softmax', 0)
    return _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs), None

def _darknet_activations(inputs, attrs):
    """Process the activation function."""
    act = _darknet_required_attr(attrs, 'activation')
    if ACTIVATION.RELU == act:
        act_type = 'relu'
    elif ACTIVATION.TANH == act:
        act_type = 'tanh'
    elif ACTIVATION.LINEAR == act:
        return inputs, None
    elif ACTIVATION.LEAKY == act:
        act_type = 'leaky_relu'
    else:
        _darknet_raise_not_supported('act: ' + act)

    if act_type in ['relu', 'tanh']:
        op_name, new_attrs = act_type, {}
        sym = _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs)
    elif act_type in ['leaky_relu']:
        op_name, new_attrs = act_type, {}
        new_attrs['alpha'] = attrs.get('slope', 0.1)
        sym = _darknet_get_nnvm_op(op_name)(*inputs, **new_attrs)
    else:
        _darknet_raise_not_supported('act_type: ' + act_type)
    return sym, None

def _darknet_op_not_support(inputs, attrs):
    """Raise exception if the operation is not supported."""
    err = "{} is not supported in {}.".format(attrs, inputs)
    raise NotImplementedError(err)

_DARKNET_CONVERT_MAP = {
    'CONVOLUTIONAL'   : _darknet_conv2d,
    'DECONVOLUTIONAL' : _darknet_conv2d_transpose,
    'CONNECTED'       : _darknet_dense,
    'MAXPOOL'         : _darknet_maxpooling,
    'SOFTMAX'         : _darknet_softmax_output,
    'DROPOUT'         : _darknet_dropout,
    'AVGPOOL'         : _darknet_avgpooling,
    'BATCHNORM'       : _darknet_batch_norm,
    'RESHAPE'         : _darknet_reshape,
    'SHORTCUT'        : _darknet_shortcut,
    'ROUTE'           : _darknet_route,
    'REORG'           : _darknet_reorg,
    'REGION'          : _darknet_region,
    'ACTIVATION'      : _darknet_activations,
    'DETECTION'       : _darknet_op_not_support,
    'CROP'            : _darknet_op_not_support,
    'COST'            : _darknet_op_not_support,
    'NORMALIZATION'   : _darknet_op_not_support,
    'LOCAL'           : _darknet_op_not_support,
    'ACTIVE'          : _darknet_op_not_support,
    'RNN'             : _darknet_op_not_support,
    'GRU'             : _darknet_op_not_support,
    'LSTM'            : _darknet_op_not_support,
    'CRNN'            : _darknet_op_not_support,
    'NETWORK'         : _darknet_op_not_support,
    'XNOR'            : _darknet_op_not_support,
    'BLANK'           : _darknet_op_not_support,
}

def _darknet_convert_symbol(op_name, inputs, attrs):
    """Convert from darknet op to nnvm op.
    The converter must specify some conversions explicitly to
    support gluon format ops such as conv2d...

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, Connected, etc
    inputs : list of nnvm.Symbol
        List of input symbols.
    attrs : dict
        Dict of operator attributes

    Returns
    -------
    out_name : converted out name of operation
    sym : nnvm.Symbol
        Converted nnvm Symbol
    """

    if op_name in _DARKNET_CONVERT_MAP:
        sym, out_name = _DARKNET_CONVERT_MAP[op_name](inputs, attrs)
    else:
        _darknet_raise_not_supported('Operator: ' + op_name)
    if out_name is  None:
        out_name = sym.list_output_names()[0].replace('_output', '')
    return out_name, sym


def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]

def _get_darknet_layername(layer_type):
    """Get the layer name from the darknet enums."""
    return str((LAYERTYPE(layer_type))).replace('LAYERTYPE.', '')

def _get_convolution_weights(layer, opname, params, dtype):
    """Get the convolution layer weights and biases."""
    if layer.nweights == 0:
        return

    if (layer.n * layer.c * layer.size * layer.size) != layer.nweights:
        raise RuntimeError("layer weights size not matching with n c h w")

    cnt = 0
    weights = np.zeros((layer.n, layer.c, layer.size, layer.size), dtype)
    for i in range(layer.n):
        for j in range(layer.c):
            for k in range(layer.size):
                for m in range(layer.size):
                    weights[i][j][k][m] = layer.weights[cnt]
                    cnt = cnt + 1

    biases = np.zeros(layer.n, dtype)
    for i in range(0, layer.n):
        biases[i] = layer.biases[i]

    k = _get_tvm_params_name(opname, 'weight')
    params[k] = tvm.nd.array(weights)
    k = _get_tvm_params_name(opname, 'bias')
    params[k] = tvm.nd.array(biases)

    if layer.batch_normalize == 1 and layer.dontloadscales != 1:
        _get_batchnorm_weights(layer, opname, params, layer.n, dtype)

def _get_connected_weights(layer, opname, params, dtype):
    """Parse the weights and biases for fully connected or dense layer."""
    size = layer.outputs * layer.inputs
    if size == 0:
        return

    weights = np.zeros((layer.outputs, layer.inputs), dtype)
    cnt = 0
    for i in range(layer.outputs):
        for j in range(layer.inputs):
            weights[i][j] = layer.weights[cnt]
            cnt += 1

    biases = np.zeros(layer.outputs, dtype)
    for i in range(layer.outputs):
        biases[i] = layer.biases[i]

    k = _get_tvm_params_name(opname, 'weight')
    params[k] = tvm.nd.array(weights)
    k = _get_tvm_params_name(opname, 'bias')
    params[k] = tvm.nd.array(biases)

    if layer.batch_normalize == 1 and layer.dontloadscales != 1:
        _get_batchnorm_weights(layer, opname, params, layer.outputs, dtype)

def _get_batchnorm_weights(layer, opname, params, size, dtype):
    """Parse the weights for batchnorm, which includes, scales, moving mean
    and moving variances."""
    scales = np.zeros(size, dtype)
    rolling_mean = np.zeros(size, dtype)
    rolling_variance = np.zeros(size, dtype)
    for i in range(size):
        scales[i] = layer.scales[i]
        rolling_mean[i] = layer.rolling_mean[i]
        rolling_variance[i] = layer.rolling_variance[i]

    k = _get_tvm_params_name(opname, 'moving_mean')
    params[k] = tvm.nd.array(rolling_mean)
    k = _get_tvm_params_name(opname, 'moving_var')
    params[k] = tvm.nd.array(rolling_variance)
    k = _get_tvm_params_name(opname, 'scales')
    params[k] = tvm.nd.array(scales)

def _get_darknet_attrs(net, layer_num):
    """Parse attributes of each layer and return."""
    attr = {}
    use_flatten = True
    layer = net.layers[layer_num]
    op_name = _get_darknet_layername(layer.type)

    if LAYERTYPE.CONVOLUTIONAL == layer.type:
        attr.update({'layout' : 'NCHW'})
        attr.update({'pad' : str(layer.pad)})
        attr.update({'num_group' : str(layer.groups)})
        attr.update({'num_filter' : str(layer.n)})
        attr.update({'stride' : str(layer.stride)})
        attr.update({'kernel' : str(layer.size)})
        attr.update({'activation' : (layer.activation)})

        if layer.nbiases == 0:
            attr.update({'use_bias' : False})
        else:
            attr.update({'use_bias' : True})

        if layer.batch_normalize == 1 and layer.dontloadscales != 1:
            attr.update({'use_batchNorm' : True})
            attr.update({'use_scales' : True})

    #elif LAYERTYPE.BATCHNORM == layer.type:
    #    attr.update({'flatten' : str('True')})

    elif LAYERTYPE.CONNECTED == layer.type:
        attr.update({'num_hidden' : str(layer.outputs)})
        attr.update({'activation' : (layer.activation)})
        if layer_num != 0:
            layer_prev = net.layers[layer_num - 1]
            if (layer_prev.out_h == layer.h and
                    layer_prev.out_w == layer.w and
                    layer_prev.out_c == layer.c):
                use_flatten = False
        attr.update({'use_flatten' : use_flatten})
        if layer.nbiases == 0:
            attr.update({'use_bias' : False})
        else:
            attr.update({'use_bias' : True})
        if layer.batch_normalize == 1 and layer.dontloadscales != 1:
            attr.update({'use_batchNorm' : True})
            attr.update({'use_scales' : True})

    elif LAYERTYPE.MAXPOOL == layer.type:
        attr.update({'pad' : str(layer.pad)})
        attr.update({'stride' : str(layer.stride)})
        attr.update({'kernel' : str(layer.size)})

    elif LAYERTYPE.AVGPOOL == layer.type:
        attr.update({'pad' : str(layer.pad)})
        if layer.stride == 0:
            attr.update({'stride' : str(1)})
        else:
            attr.update({'stride' : str(layer.stride)})
        if layer.size == 0 and layer.h == layer.w:
            attr.update({'kernel' : str(layer.h)})
        else:
            attr.update({'kernel' : str(layer.size)})

    elif LAYERTYPE.DROPOUT == layer.type:
        attr.update({'p' : str(layer.probability)})

    elif LAYERTYPE.SOFTMAX == layer.type:
        attr.update({'axis' : 1})
        attr.update({'use_flatten' : True})

    elif LAYERTYPE.SHORTCUT == layer.type:
        attr.update({'activation' : (layer.activation)})

    elif LAYERTYPE.ROUTE == layer.type:
        pass

    elif LAYERTYPE.COST == layer.type:
        pass

    elif LAYERTYPE.REORG == layer.type:
        attr.update({'stride' : layer.stride})

    elif LAYERTYPE.REGION == layer.type:
        attr.update({'n' : layer.n})
        attr.update({'classes' : layer.classes})
        attr.update({'coords' : layer.coords})
        attr.update({'background' : layer.background})
        attr.update({'softmax' : layer.softmax})
    else:
        err = "Darknet layer {} is not supported in nnvm.".format(op_name)
        raise NotImplementedError(err)

    return op_name, attr

def _get_tvm_params_name(opname, arg_name):
    """Makes the params name for the k,v pair."""
    return opname + '_'+ arg_name

def _get_darknet_params(layer, opname, tvmparams, dtype='float32'):
    """To parse and get the darknet params."""
    if LAYERTYPE.CONVOLUTIONAL == layer.type:
        _get_convolution_weights(layer, opname, tvmparams, dtype)

    #elif LAYERTYPE.BATCHNORM == layer.type:
    #   size = layer.outputs
    #   _get_batchnorm_weights(layer, opname, tvmparams, size, dtype)

    elif LAYERTYPE.CONNECTED == layer.type:
        _get_connected_weights(layer, opname, tvmparams, dtype)

def _preproc_layer(net, i, sym_array):
    """To preprocess each darknet layer, some layer doesnt need processing."""
    layer = net.layers[i]
    if i == 0:
        name = 'data'
        attribute = {}
        sym = [_sym.Variable(name, **attribute)]
    else:
        sym = sym_array[i - 1]
    skip_layer = False

    if LAYERTYPE.ROUTE == layer.type:
        sym = []
        for j in range(layer.n):
            sym.append(sym_array[layer.input_layers[j]])
        if layer.n == 1:
            skip_layer = True

    elif LAYERTYPE.COST == layer.type:
        skip_layer = True

    elif LAYERTYPE.SHORTCUT == layer.type:
        sym = [sym, sym_array[layer.index]]

    elif LAYERTYPE.BLANK == layer.type:
        skip_layer = True

    if skip_layer is True:
        sym_array[i] = sym

    return skip_layer, sym


def _from_darknet(net, dtype='float32'):
    """To convert the darknet symbol to nnvm symbols."""
    sym_array = {}
    tvmparams = {}
    for i in range(net.n):
        need_skip, sym = _preproc_layer(net, i, sym_array)
        if need_skip is True:
            continue
        op_name, attr = _get_darknet_attrs(net, i)
        layer_name, sym = _darknet_convert_symbol(op_name, _as_list(sym), attr)
        _get_darknet_params(net.layers[i], layer_name, tvmparams, dtype)
        sym_array[i] = sym

    return sym, tvmparams

def _resize_image(img, w_in, h_in):
    """Resize the image to the given height and width."""
    imc, imh, imw = img.shape
    h_in = int(h_in)
    w_in = int(w_in)
    part = np.zeros((imc, imh, w_in))
    resized = np.zeros((imc, h_in, w_in))
    w_scale = (imw - 1) / (w_in - 1)
    h_scale = (imh - 1) / (h_in - 1)
    for k in range(imc):
        for j in range(imh):
            for c in range(w_in):
                if c == w_in - 1 or imw == 1:
                    part[k][j][c] = img[k][j][imw - 1]
                else:
                    fdx, idx = math.modf(c * w_scale)
                    part[k][j][c] = (1 - fdx) * img[k][j][int(idx)] + \
                                            fdx * img[k][j][int(idx) + 1]
    for k in range(imc):
        for j in range(h_in):
            fdy, idy = math.modf(j * h_scale)
            for c in range(w_in):
                resized[k][j][c] = (1 - fdy)*part[k][int(idy)][c]
            if (j == h_in - 1) or (imh == 1):
                continue
            for c in range(w_in):
                resized[k][j][c] += fdy * part[k][int(idy) + 1][c]
    return resized

def _load_image_color(test_image):
    """To load the image using opencv api and do preprocessing."""
    imagex = cv2.imread(test_image)
    imagex = np.array(imagex)
    imagex = imagex.transpose((2, 0, 1))
    imagex = np.divide(imagex, 255)
    imagex = np.flip(imagex, 0)
    return imagex

def _letterbox_image(img, w_in, h_in):
    """To get the image in boxed format."""
    imc, imh, imw = img.shape
    if (w_in / imw) < (h_in / imh):
        new_w = w_in
        new_h = imh * w_in / imw
    else:
        new_h = h_in
        new_w = imw * h_in/imh
    resized = _resize_image(img, new_w, new_h)
    boxed = np.full((imc, h_in, w_in), 0.5, dtype=float)
    _, resizedh, resizedw = resized.shape
    boxed[:, int((h_in - new_h) / 2)
          :int((h_in - new_h) / 2) + resizedh, int((w_in - new_w) / 2)
          :int((w_in - new_w) / 2) + resizedw,] = resized
    return boxed

def load_image(image, resize_width, resize_height):
    """Load the image and convert to the darknet model format.
    The image processing of darknet is different from normal.
    Parameters
    ----------
    image : string
        The image file name with path

    resize_width : integer
        The width to which the image needs to be resized

    resize_height : integer
        The height to which the image needs to be resized

    Returns
    -------
    img : Float array
        Array of processed image
    """

    img = _load_image_color(image)
    return _letterbox_image(img, resize_width, resize_height)

def from_darknet(net, dtype='float32'):
    """Convert from darknet's model into compatible NNVM format.
    Reconstruct a nnvm symbol by traversing the darknet input.

    Parameters
    ----------
    net : ctype Pointer to network
        Darknet parsed symbols

    dtype : str
        Datatype of the input net structure, default is float32

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """

    return _from_darknet(net, dtype)
