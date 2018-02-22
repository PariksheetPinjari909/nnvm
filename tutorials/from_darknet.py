"""
Compile Darknet Models
=====================
**Author**: `Siju Samuel <https://github.com/siju-samuel/>`_

This article is an introductory tutorial to deploy darknet models with NNVM.

All the required models and libraries will be downloaded from the internet
by the script.
"""
from ctypes import *
import math
import random
import nnvm
import nnvm.frontend.darknet
from nnvm.frontend.darknet_c_interface import __darknetffi__
import numpy as np
import tvm
import os, sys, time, urllib, requests
if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse

def dlProgress(count, block_size, total_size):
    """Show the download progress."""
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
          (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download(url, path, overwrite=False, sizecompare=False):
    """Downloads the file from the internet.
    Set the input options correctly to overwrite or do the size comparison

    Parameters
    ----------
    url : str
        Operator name, such as Convolution, Connected, etc
    path : str
        List of input symbols.
    overwrite : dict
        Dict of operator attributes
    sizecompare : dict
        Dict of operator attributes

    Returns
    -------
    out_name : converted out name of operation
    sym : nnvm.Symbol
        Converted nnvm Symbol
    """
    if os.path.isfile(path) and not overwrite:
        if (sizecompare):
            fileSize = os.path.getsize(path)
            resHead = requests.head(url)
            resGet = requests.get(url,stream=True)
            if 'Content-Length' not in resHead.headers :
                resGet = urllib2.urlopen(url)
            urlFileSize = int(resGet.headers['Content-Length'])
            if urlFileSize != fileSize:
                print ("exist file got corrupted, downloading", path , " file freshly")
                download(url, path, True, False)
                return
        print('File {} exists, skip.'.format(path))
        return
    import urllib.request
    print('Downloading from url {} to {}'.format(url, path))
    try:
        urllib.request.urlretrieve(url, path, reporthook=dlProgress)
        print('')
    except:
        urllib.urlretrieve(url, path, reporthook=dlProgress)

######################################################################
# Prepare cfg and weights file
# Pretrained model available https://pjreddie.com/darknet/imagenet/
# --------------------------------------------------------------------
# Download cfg and weights file first time.
# Supported models alexnet, resnet50, resnet152, extraction, yolo
model_name = 'resnet50'

cfg_name = model_name + '.cfg'
weights_name = model_name + '.weights'
cfg_url = 'https://github.com/siju-samuel/darknet/blob/master/cfg/' + \
            cfg_name + '?raw=true'
weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'

download(cfg_url, cfg_name)
download(weights_url, weights_name)

######################################################################
# Download and Load darknet library
# ---------------------------------

darknet_lib = 'libdarknet.so'
darknetlib_url = 'https://github.com/siju-samuel/darknet/blob/master/lib/' + \
                        darknet_lib + '?raw=true'
download(darknetlib_url, darknet_lib)
darknet_lib = __darknetffi__.dlopen('./' + darknet_lib)
cfg = "./" + str(cfg_name)
weights = "./" + str(weights_name)
net = darknet_lib.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1
print("Converting darknet to nnvm symbols...")
sym, params = nnvm.frontend.darknet.from_darknet(net, dtype)

######################################################################
# Compile the model on NNVM
# --------------------------------------------------------------------
# compile the model
data = np.empty([batch_size, net.c ,net.h, net.w], dtype);
target = 'llvm'
shape = {'data': data.shape}
print("Compiling the model...")
with nnvm.compiler.build_config(opt_level=2):
    graph, lib, params = nnvm.compiler.build(sym, target, shape, dtype, params)

#####################################################################
# Save the json
# --------------------------------------------------------------------
def save_lib():
    '''Save the graph, params and .so to the current directory'''
    print("Saving the compiled output...")
    path_name = 'nnvm_darknet_' + model_name
    path_lib = path_name + '_deploy_lib.so'
    lib.export_library(path_lib)
    with open(path_name 
+ "deploy_graph.json", "w") as fo:
        fo.write(graph.json())
    with open(path_name 
+ "deploy_param.params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))
#save_lib()

######################################################################
# Load a test image
# --------------------------------------------------------------------
print("Loading the test image...")
test_image = 'dog.jpg'
img_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + \
            test_image   +'?raw=true'
download(img_url, test_image)

data = nnvm.frontend.darknet.load_image(test_image, net.w, net.h)
######################################################################
# Execute on TVM
# --------------------------------------------------------------------
# The process is no different from other examples.
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)

m = graph_runtime.create(graph, lib, ctx)

# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# execute
print("Predicing the test image...")

start_time = time.time()
m.run()
timediff = (time.time() - start_time)
# get outputs
out_shape = (net.outputs,)
tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
top1 = np.argmax(tvm_out)
print("TVM Run Time = %s seconds." % timediff)
print('TVM Prediction output id : ', top1)

#####################################################################
# Look up synset name
# --------------------------------------------------------------------
# Look up prdiction top 1 index in 1000 class imagenet.
out_name_file = {}
out_name_file['yolo'] = "coco.names"
out_name_file['yolo-voc'] = "voc.names"
out_name_file['alexnet'] = "imagenet.shortnames.list"
out_name_file['resnet50'] = "imagenet.shortnames.list"
out_name_file['resnet152'] = "imagenet.shortnames.list"
out_name_file['extraction'] = "imagenet.shortnames.list"

imagenet_name = out_name_file[model_name]
imagenet_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' \
                + imagenet_name +'?raw=true'

download(imagenet_url, imagenet_name)
with open(imagenet_name) as f:
    imagenet = f.readlines()

print("TVM Predicted result : ", imagenet[top1])

#####################################################################
# confirm correctness with darknet output
# --------------------------------------------------------------------
start_time = time.time()
darknet_lib.network_predict_image(net, darknet_lib.load_image_color(test_image.encode('utf-8'), 0, 0))
print("DARKNET Run Time = %s seconds." % (time.time() - start_time))
from cffi import FFI
ffi = FFI()
top1_darknet = ffi.new("int *")
darknet_lib.top_predictions(net, 1, top1_darknet)
print("DARKNET LIB Prediction output id : ", top1_darknet[0])
print("DARKNET predicted result = ", imagenet[top1_darknet[0]])


