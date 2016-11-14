#-*- coding: utf-8 -*-

from math import pow

import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H / 2 * W / 2, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        '''
        conv - relu - 2x2 max pool - affine - relu - affine - softmax
        '''
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        N, F, H_half, W_half = conv_out.shape
        conv_out_reshape = np.reshape(conv_out, [N, F * H_half * W_half])
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_out_reshape, W2, b2)
        scores, affine_cache = affine_forward(affine_relu_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.square(self.params['W3']).sum()
        loss += 0.5 * self.reg * np.square(self.params['W2']).sum()
        loss += 0.5 * self.reg * np.square(self.params['W1']).sum()

        daffine, grads['W3'], grads['b3'] = affine_backward(dout, affine_cache)
        daffine_relu, grads['W2'], grads['b2'] = affine_relu_backward(daffine, affine_relu_cache)
        # flat ---> tensor
        daffine_relu_reshape = np.reshape(daffine_relu, [N, F, H_half, W_half])
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(daffine_relu_reshape, conv_cache)

        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class VGGNet(object):
    """
    A convolutional network with the similar architecutre to VGGNet

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dim=(3, 32, 32),
                 fc_dim=4096,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        f_size = 3
        # num_filters = (64, 128, 256, 256, 512, 512, 512, 512)
        num_filters = (16, 32, 64, 64, 128, 128, 128, 128)
        num_pool_layers = 5

        print "initialization start..."
        self.params['CONV1-W'] = np.random.randn(num_filters[0], C, f_size, f_size) / np.sqrt(C * f_size * f_size / 2)
        self.params['CONV1-b'] = np.zeros(num_filters[0])
        self.params['CONV2-W'] = np.random.randn(num_filters[1], num_filters[0], f_size, f_size) / np.sqrt(num_filters[0] * f_size * f_size / 2)
        self.params['CONV2-b'] = np.zeros(num_filters[1])
        self.params['CONV3-W'] = np.random.randn(num_filters[2], num_filters[1], f_size, f_size) / np.sqrt(num_filters[1] * f_size * f_size / 2)
        self.params['CONV3-b'] = np.zeros(num_filters[2])
        self.params['CONV4-W'] = np.random.randn(num_filters[3], num_filters[2], f_size, f_size) / np.sqrt(num_filters[2] * f_size * f_size / 2)
        self.params['CONV4-b'] = np.zeros(num_filters[3])
        self.params['CONV5-W'] = np.random.randn(num_filters[4], num_filters[3], f_size, f_size) / np.sqrt(num_filters[3] * f_size * f_size / 2)
        self.params['CONV5-b'] = np.zeros(num_filters[4])
        self.params['CONV6-W'] = np.random.randn(num_filters[5], num_filters[4], f_size, f_size) / np.sqrt(num_filters[4] * f_size * f_size / 2)
        self.params['CONV6-b'] = np.zeros(num_filters[5])
        self.params['CONV7-W'] = np.random.randn(num_filters[6], num_filters[5], f_size, f_size) / np.sqrt(num_filters[5] * f_size * f_size / 2)
        self.params['CONV7-b'] = np.zeros(num_filters[6])
        self.params['CONV8-W'] = np.random.randn(num_filters[7], num_filters[6], f_size, f_size) / np.sqrt(num_filters[6] * f_size * f_size / 2)
        self.params['CONV8-b'] = np.zeros(num_filters[7])

        self.params['BN1-g'] = np.ones(num_filters[0])
        self.params['BN1-b'] = np.zeros(num_filters[0])
        self.params['BN2-g'] = np.ones(num_filters[1])
        self.params['BN2-b'] = np.zeros(num_filters[1])
        self.params['BN3-g'] = np.ones(num_filters[2])
        self.params['BN3-b'] = np.zeros(num_filters[2])
        self.params['BN4-g'] = np.ones(num_filters[3])
        self.params['BN4-b'] = np.zeros(num_filters[3])
        self.params['BN5-g'] = np.ones(num_filters[4])
        self.params['BN5-b'] = np.zeros(num_filters[4])
        self.params['BN6-g'] = np.ones(num_filters[5])
        self.params['BN6-b'] = np.zeros(num_filters[5])
        self.params['BN7-g'] = np.ones(num_filters[6])
        self.params['BN7-b'] = np.zeros(num_filters[6])
        self.params['BN8-g'] = np.ones(num_filters[7])
        self.params['BN8-b'] = np.zeros(num_filters[7])

        total_pooling = int(pow(2, num_pool_layers))
        last_num_filters = num_filters[-1]
        hidden_dims = (fc_dim, fc_dim)
        first_fc_input_dim = last_num_filters * H / total_pooling * W / total_pooling
        self.params['FC1-W'] = np.random.randn(first_fc_input_dim, hidden_dims[0]) / np.sqrt(first_fc_input_dim / 2)
        self.params['FC1-b'] = np.zeros(hidden_dims[0])
        self.params['FC2-W'] = np.random.randn(hidden_dims[0], hidden_dims[1]) / np.sqrt(hidden_dims[0] / 2)
        self.params['FC2-b'] = np.zeros(hidden_dims[1])
        self.params['FC3-W'] = np.random.randn(hidden_dims[1], num_classes) / np.sqrt(hidden_dims[1] / 2)
        self.params['FC3-b'] = np.zeros(num_classes)

        self.params['BN9-g'] = np.ones(hidden_dims[0])
        self.params['BN9-b'] = np.zeros(hidden_dims[0])
        self.params['BN10-g'] = np.ones(hidden_dims[1])
        self.params['BN10-b'] = np.zeros(hidden_dims[1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        self.bn_params = [{'mode': 'train'} for i in xrange(len(num_filters) + len(hidden_dims))]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        CONV1_W, CONV1_b = self.params['CONV1-W'], self.params['CONV1-b']
        CONV2_W, CONV2_b = self.params['CONV2-W'], self.params['CONV2-b']
        CONV3_W, CONV3_b = self.params['CONV3-W'], self.params['CONV3-b']
        CONV4_W, CONV4_b = self.params['CONV4-W'], self.params['CONV4-b']
        CONV5_W, CONV5_b = self.params['CONV5-W'], self.params['CONV5-b']
        CONV6_W, CONV6_b = self.params['CONV6-W'], self.params['CONV6-b']
        CONV7_W, CONV7_b = self.params['CONV7-W'], self.params['CONV7-b']
        CONV8_W, CONV8_b = self.params['CONV8-W'], self.params['CONV8-b']
        BN1_g, BN1_b = self.params['BN1-g'], self.params['BN1-b']
        BN2_g, BN2_b = self.params['BN2-g'], self.params['BN2-b']
        BN3_g, BN3_b = self.params['BN3-g'], self.params['BN3-b']
        BN4_g, BN4_b = self.params['BN4-g'], self.params['BN4-b']
        BN5_g, BN5_b = self.params['BN5-g'], self.params['BN5-b']
        BN6_g, BN6_b = self.params['BN6-g'], self.params['BN6-b']
        BN7_g, BN7_b = self.params['BN7-g'], self.params['BN7-b']
        BN8_g, BN8_b = self.params['BN8-g'], self.params['BN8-b']
        FC1_W, FC1_b = self.params['FC1-W'], self.params['FC1-b']
        FC2_W, FC2_b = self.params['FC2-W'], self.params['FC2-b']
        FC3_W, FC3_b = self.params['FC3-W'], self.params['FC3-b']
        BN9_g, BN9_b = self.params['BN9-g'], self.params['BN9-b']
        BN10_g, BN10_b = self.params['BN10-g'], self.params['BN10-b']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # subtract mean image per channel
        X_mean = np.mean(X, axis=(0, 2, 3))
        X[:, 0] -= X_mean[0]
        X[:, 1] -= X_mean[1]
        X[:, 2] -= X_mean[2]

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        '''
        conv1 - batch - relu - pool1
        conv2 - batch - relu - pool2
        conv3 - batch - relu - conv4 - batch - relu - pool3
        conv5 - batch - relu - conv6 - batch - relu - pool4
        conv7 - batch - relu - conv8 - batch - relu - pool5
        affine - relu - affine - relu - affine - softmax
        '''
        out1, cache1 = conv_batch_relu_pool_forward(X, CONV1_W, CONV1_b, BN1_g, BN1_b, conv_param, self.bn_params[0], pool_param)
        out2, cache2 = conv_batch_relu_pool_forward(out1, CONV2_W, CONV2_b, BN2_g, BN2_b, conv_param, self.bn_params[1], pool_param)

        out3, cache3 = conv_batch_relu_forward(out2, CONV3_W, CONV3_b, BN3_g, BN3_b, conv_param, self.bn_params[2])
        out4, cache4 = conv_batch_relu_pool_forward(out3, CONV4_W, CONV4_b, BN4_g, BN4_b, conv_param, self.bn_params[3], pool_param)
        out5, cache5 = conv_batch_relu_forward(out4, CONV5_W, CONV5_b, BN5_g, BN5_b, conv_param, self.bn_params[4])
        out6, cache6 = conv_batch_relu_pool_forward(out5, CONV6_W, CONV6_b, BN6_g, BN6_b, conv_param, self.bn_params[5], pool_param)
        out7, cache7 = conv_batch_relu_forward(out6, CONV7_W, CONV7_b, BN7_g, BN7_b, conv_param, self.bn_params[6])
        out8, cache8 = conv_batch_relu_pool_forward(out7, CONV8_W, CONV8_b, BN8_g, BN8_b, conv_param, self.bn_params[7], pool_param)

        # FC layer 에 진입할때 reshape 필요하다.
        n, f, h, w = out8.shape
        out8_reshape = np.reshape(out8, [n, f * h * w])

        fc_out1, fc_cache1 = affine_batchnorm_relu_forward(out8_reshape, FC1_W, FC1_b, BN9_g, BN9_b, self.bn_params[8])
        fc_out2, fc_cache2 = affine_batchnorm_relu_forward(fc_out1, FC2_W, FC2_b, BN10_g, BN10_b, self.bn_params[9])
        fc_out3, fc_cache3 = affine_forward(fc_out2, FC3_W, FC3_b)

        scores = fc_out3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.square(self.params['FC3-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['FC2-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['FC1-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV8-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV7-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV6-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV5-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV4-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV3-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV2-W']).sum()
        loss += 0.5 * self.reg * np.square(self.params['CONV1-W']).sum()

        dfc3, grads['FC3-W'], grads['FC3-b'] = affine_backward(dout, fc_cache3)
        dfc2, grads['FC2-W'], grads['FC2-b'], grads['BN10-g'], grads['BN10-b'] = affine_batchnorm_relu_backward(dfc3, fc_cache2)
        dfc1, grads['FC1-W'], grads['FC1-b'], grads['BN9-g'], grads['BN9-b'] = affine_batchnorm_relu_backward(dfc2, fc_cache1)

        # CONV layer 에 진입할 때 다시 reshape 해줘야 한다.
        dfc1_reshape = np.reshape(dfc1, [n, f, h, w])

        dconv8, grads['CONV8-W'], grads['CONV8-b'], grads['BN8-g'], grads['BN8-b'] = conv_batch_relu_pool_backward(dfc1_reshape, cache8)
        dconv7, grads['CONV7-W'], grads['CONV7-b'], grads['BN7-g'], grads['BN7-b'] = conv_batch_relu_backward(dconv8, cache7)
        dconv6, grads['CONV6-W'], grads['CONV6-b'], grads['BN6-g'], grads['BN6-b'] = conv_batch_relu_pool_backward(dconv7, cache6)
        dconv5, grads['CONV5-W'], grads['CONV5-b'], grads['BN5-g'], grads['BN5-b'] = conv_batch_relu_backward(dconv6, cache5)
        dconv4, grads['CONV4-W'], grads['CONV4-b'], grads['BN4-g'], grads['BN4-b'] = conv_batch_relu_pool_backward(dconv5, cache4)
        dconv3, grads['CONV3-W'], grads['CONV3-b'], grads['BN3-g'], grads['BN3-b'] = conv_batch_relu_backward(dconv4, cache3)

        dconv2, grads['CONV2-W'], grads['CONV2-b'], grads['BN2-g'], grads['BN2-b'] = conv_batch_relu_pool_backward(dconv3, cache2)
        dconv1, grads['CONV1-W'], grads['CONV1-b'], grads['BN1-g'], grads['BN1-b'] = conv_batch_relu_pool_backward(dconv2, cache1)

        grads['FC3-W'] += self.reg * FC3_W
        grads['FC2-W'] += self.reg * FC2_W
        grads['FC1-W'] += self.reg * FC1_W

        grads['CONV8-W'] += self.reg * CONV8_W
        grads['CONV7-W'] += self.reg * CONV7_W
        grads['CONV6-W'] += self.reg * CONV6_W
        grads['CONV5-W'] += self.reg * CONV5_W
        grads['CONV4-W'] += self.reg * CONV4_W
        grads['CONV3-W'] += self.reg * CONV3_W
        grads['CONV2-W'] += self.reg * CONV2_W
        grads['CONV1-W'] += self.reg * CONV1_W
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

pass
