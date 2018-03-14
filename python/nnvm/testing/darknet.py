# pylint: disable=invalid-name, unused-variable, unused-argument, old-style-class, no-init
"""
Compile DarkNet Models
====================
DarkNet helper functions for yolo and image loading.

"""
from __future__ import division
from enum import IntEnum
import math
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from cffi import FFI

class box:
    x = 0
    y = 0
    w = 0
    h = 0
def _entry_index(batch, w, h, outputs, classes, coords, location, entry):
    n = int(location/(w*h))
    loc = location%(w*h)
    return batch*outputs + n*w*h*(coords+classes+1) + entry*w*h + loc

def _get_region_box(x, biases, n, index, i, j, w, h, stride):
    b = box()
    b.x = (i + x[index + 0*stride]) / w
    b.y = (j + x[index + 1*stride]) / h
    b.w = np.exp(x[index + 2*stride]) * biases[2*n] / w
    b.h = np.exp(x[index + 3*stride]) * biases[2*n+1] / h
    return b

def _correct_region_boxes(boxes, n, w, h, netw, neth, relative):
    new_w, new_h = (netw, (h*netw)/w) if (netw/w < neth/h) else ((w*neth/h), neth)
    for i in range(n):
        b = boxes[i]
        b.x = (b.x - (netw - new_w)/2/netw) / (new_w/netw)
        b.y = (b.y - (neth - new_h)/2/neth) / (new_h/neth)
        b.w *= netw/new_w
        b.h *= neth/new_h
        if not relative:
            b.x *= w
            b.w *= w
            b.y *= h
            b.h *= h
        boxes[i] = b

def _overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = r1 if r1 < r2 else r2
    return right - left

def _box_intersection(a, b):
    w = _overlap(a.x, a.w, b.x, b.w)
    h = _overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0
    return w*h

def _box_union(a, b):
    i = _box_intersection(a, b)
    u = a.w*a.h + b.w*b.h - i
    return u

def _box_iou(a, b):
    return _box_intersection(a, b)/_box_union(a, b)

def get_region_boxes(layer_in, imw, imh, netw, neth, thresh, probs,
                     boxes, relative, tvm_out):
    "To get the boxes for the image based on the prediction"
    lw = layer_in.w
    lh = layer_in.h
    probs = [[0 for i in range(layer_in.classes + 1)] for y in range(lw*lh*layer_in.n)]
    boxes = [box() for i in range(lw*lh*layer_in.n)]
    for i in range(lw*lh):
        row = int(i / lw)
        col = int(i % lw)
        for n in range(layer_in.n):
            index = n*lw*lh + i
            obj_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                     layer_in.coords, n*lw*lh + i, layer_in.coords)
            box_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                     layer_in.coords, n*lw*lh + i, 0)
            mask_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                      layer_in.coords, n*lw*lh + i, 4)
            scale = 1 if layer_in.background  else tvm_out[obj_index]
            boxes[index] = _get_region_box(tvm_out, layer_in.biases, n, box_index, col,
                                           row, lw, lh, lw*lh)
            if not layer_in.softmax_tree:
                max_element = 0
                for j in range(layer_in.classes):
                    class_index = _entry_index(0, lw, lh, layer_in.outputs, layer_in.classes,
                                               layer_in.coords, n*lw*lh + i, layer_in.coords+1+j)
                    prob = scale*tvm_out[class_index]
                    probs[index][j] = prob if prob > thresh else 0
                    max_element = max(max_element, prob)
                probs[index][layer_in.classes] = max_element

    _correct_region_boxes(boxes, lw*lh*layer_in.n, imw, imh, netw, neth, relative)
    return boxes, probs


def do_nms_sort(boxes, probs, total, classes, thresh):
    "Does the sorting based on the threshold values"
    class sortable_bbox:
        index_var = 0
        class_var = 0
        probs = []

    s = [sortable_bbox() for i in range(total)]
    for i in range(total):
        s[i].index_var = i
        s[i].class_var = 0
        s[i].probs = probs

    for k in range(classes):
        for i in range(total):
            s[i].class_var = k
        s = sorted(s, key=lambda x: x.probs[x.index_var][x.class_var], reverse=True)
        for i in range(total):
            if probs[s[i].index_var][k] == 0:
                continue
            a = boxes[s[i].index_var]
            for j in range(i+1, total):
                b = boxes[s[j].index_var]
                if _box_iou(a, b) > thresh:
                    probs[s[j].index_var][k] = 0
    return boxes, probs

def draw_detections(im, num, thresh, boxes, probs, names, classes):
    "Draw the markings around the detected region"
    for i in range(num):
        labelstr = []
        category = -1
        for j in range(classes):
            if probs[i][j] > thresh:
                if category == -1:
                    category = j
                labelstr.append(names[j])
        if category > -1:
            imc, imh, imw = im.shape
            width = int(imh * 0.006)
            offset = category*123457 % classes
            red = _get_color(2, offset, classes)
            green = _get_color(1, offset, classes)
            blue = _get_color(0, offset, classes)
            rgb = [red, green, blue]
            b = boxes[i]
            left = int((b.x-b.w/2.)*imw)
            right = int((b.x+b.w/2.)*imw)
            top = int((b.y-b.h/2.)*imh)
            bot = int((b.y+b.h/2.)*imh)

            if left < 0:
                left = 0
            if right > imw-1:
                right = imw-1
            if top < 0:
                top = 0
            if bot > imh-1:
                bot = imh-1
            _draw_box_width(im, left, top, right, bot, width, red, green, blue)
            label = _get_label(''.join(labelstr), rgb)
            _draw_label(im, top + width, left, label, rgb)

def _get_pixel(im, x, y, c):
    return im[c][y][x]

def _set_pixel(im, x, y, c, val):
    if x < 0 or y < 0 or c < 0 or x >= im.shape[2] or y >= im.shape[1] or c >= im.shape[0]:
        return
    im[c][y][x] = val

def _draw_label(im, r, c, label, rgb):
    w = label.shape[2]
    h = label.shape[1]
    if (r - h) >= 0:
        r = r - h

    for j in range(h):
        if j < h and (j + r) < im.shape[1]:
            for i in range(w):
                if i < w and (i + c) < im.shape[2]:
                    for k in range(label.shape[0]):
                        val = _get_pixel(label, i, j, k)
                        _set_pixel(im, i+c, j+r, k, val)#rgb[k] * val)

def _get_label(labelstr, rgb):
    text = labelstr
    colorText = "black"
    testDraw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    font = ImageFont.truetype("arial.ttf", 25)
    width, height = testDraw.textsize(labelstr, font=font)
    img = Image.new('RGB', (width, height), color=(int(rgb[0]*255), int(rgb[1]*255),
                                                   int(rgb[2]*255)))
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, fill=colorText, font=font)
    opencvImage = np.divide(np.asarray(img), 255)
    return opencvImage.transpose(2, 0, 1)

def _get_color(c, x, max_value):
    c = int(c)
    colors = [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    ratio = (float(x)/float(max_value)) * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio -= i
    r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
    return r

def _draw_box(im, x1, y1, x2, y2, r, g, b):
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    ac, ah, aw = im.shape
    if x1 < 0:
        x1 = 0
    if x1 >= aw:
        y1 = 0
    if y1 >= ah:
        y1 = ah - 1
    if y2 < 0:
        y2 = 0
    if y2 >= ah:
        y2 = ah - 1

    for i in range(x1, x2):
        im[0][y1][i] = r
        im[0][y2][i] = r
        im[1][y1][i] = g
        im[1][y2][i] = g
        im[2][y1][i] = b
        im[2][y2][i] = b

    for i in range(y1, y2):
        im[0][i][x1] = r
        im[0][i][x2] = r
        im[1][i][x1] = g
        im[1][i][x2] = g
        im[2][i][x1] = b
        im[2][i][x2] = b

def _draw_box_width(im, x1, y1, x2, y2, w, r, g, b):
    for i in range(int(w)):
        _draw_box(im, x1+i, y1+i, x2-i, y2-i, r, g, b)


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

def load_image_color(test_image):
    """To load the image using opencv api and do preprocessing."""
    imagex = cv2.imread(test_image)
    imagex = np.array(imagex)
    imagex = imagex.transpose((2, 0, 1))
    imagex = np.divide(imagex, 255.0)
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
          :int((w_in - new_w) / 2) + resizedw] = resized
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

    img = load_image_color(image)
    return _letterbox_image(img, resize_width, resize_height)

class LAYERTYPE(IntEnum):
    """Darknet LAYERTYPE Class constant."""
    CONVOLUTIONAL = 0
    DECONVOLUTIONAL = 1
    CONNECTED = 2
    MAXPOOL = 3
    SOFTMAX = 4
    DETECTION = 5
    DROPOUT = 6
    CROP = 7
    ROUTE = 8
    COST = 9
    NORMALIZATION = 10
    AVGPOOL = 11
    LOCAL = 12
    SHORTCUT = 13
    ACTIVE = 14
    RNN = 15
    GRU = 16
    LSTM = 17
    CRNN = 18
    BATCHNORM = 19
    NETWORK = 20
    XNOR = 21
    REGION = 22
    REORG = 23
    BLANK = 24

class ACTIVATION(IntEnum):
    """Darknet ACTIVATION Class constant."""
    LOGISTIC = 0
    RELU = 1
    RELIE = 2
    LINEAR = 3
    RAMP = 4
    TANH = 5
    PLSE = 6
    LEAKY = 7
    ELU = 8
    LOGGY = 9
    STAIR = 10
    HARDTAN = 11
    LHTAN = 12

__darknetffi__ = FFI()

__darknetffi__.cdef("""
typedef struct network network;
typedef struct layer layer;

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;


typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    REORG,
    BLANK
} LAYERTYPE;

typedef enum{
    SSE, MASKED, LONE, SEG, SMOOTH
} COSTTYPE;


struct layer{
    LAYERTYPE type;
    ACTIVATION activation;
    COSTTYPE cost_type;
    void (*forward);
    void (*backward);
    void (*update);
    void (*forward_gpu);
    void (*backward_gpu);
    void (*update_gpu);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float thresh;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

    float * delta;
    float * output;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu;

    float * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;
};


typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} LEARNINGRATEPOLICY;

typedef struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    LEARNINGRATEPOLICY policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
} network;


typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

network *load_network(char *cfg, char *weights, int clear);
image letterbox_image(image im, int w, int h);
int resize_network(network *net, int w, int h);
void top_predictions(network *net, int n, int *index);
void free_image(image m);
image load_image_color(char *filename, int w, int h);
float *network_predict_image(network *net, image im);
network *make_network(int n);
layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);
layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
layer make_avgpool_layer(int batch, int w, int h, int c);
layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
layer make_batchnorm_layer(int batch, int w, int h, int c);
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
layer make_region_layer(int batch, int w, int h, int n, int classes, int coords);
void free_network(network *net);
"""
                   )
