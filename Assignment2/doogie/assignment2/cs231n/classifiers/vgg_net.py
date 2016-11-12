import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class VGGNet(object):
  """
  A VGG-network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  batch normalization as options. For a network with L layers,
  the architecture will be

  conv(64) - batch - relu - pool
  conv(128) - batch - relu - pool
  conv(512) - batch - relu - conv - batch - relu - pool
  conv(512) - batch - relu - conv - batch - relu - pool
  conv(512) - batch - relu - conv - batch - relu - pool
  FC(4096) - batch - relu
  FC(4096) - batch - relu
  FC(10) - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, input_dim=(3, 32, 32), num_classes=10, hidden_dim=4096,
               reg=0.0, dtype=np.float32):
    """
    Initialize a new VGGNet.

    Inputs:
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_layers = 11
    C, H, W = input_dim
    F = (64, 128, 256, 256, 512, 512, 512, 512)
    HH = 3
    WW = 3
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    ### Conv
    self.params['W1'] = np.random.randn(F[0], C, HH, WW) / np.sqrt(C / 2)
    self.params['b1'] = np.zeros(F[0])
    self.params['gamma1'] = np.ones(F[0])
    self.params['beta1'] = np.zeros(F[0])
    self.params['W2'] = np.random.randn(F[1], F[0], HH, WW) / np.sqrt(F[0] / 2)
    self.params['b2'] = np.zeros(F[1])
    self.params['gamma2'] = np.ones(F[1])
    self.params['beta2'] = np.zeros(F[1])
    self.params['W3'] = np.random.randn(F[2], F[1], HH, WW) / np.sqrt(F[1] / 2)
    self.params['b3'] = np.zeros(F[2])
    self.params['gamma3'] = np.ones(F[2])
    self.params['beta3'] = np.zeros(F[2])
    self.params['W4'] = np.random.randn(F[3], F[2], HH, WW) / np.sqrt(F[2] / 2)
    self.params['b4'] = np.zeros(F[3])
    self.params['gamma4'] = np.ones(F[3])
    self.params['beta4'] = np.zeros(F[3])
    self.params['W5'] = np.random.randn(F[4], F[3], HH, WW) / np.sqrt(F[3] / 2)
    self.params['b5'] = np.zeros(F[4])
    self.params['gamma5'] = np.ones(F[4])
    self.params['beta5'] = np.zeros(F[4])
    self.params['W6'] = np.random.randn(F[5], F[4], HH, WW) / np.sqrt(F[4] / 2)
    self.params['b6'] = np.zeros(F[5])
    self.params['gamma6'] = np.ones(F[5])
    self.params['beta6'] = np.zeros(F[5])
    self.params['W7'] = np.random.randn(F[6], F[5], HH, WW) / np.sqrt(F[5] / 2)
    self.params['b7'] = np.zeros(F[6])
    self.params['gamma7'] = np.ones(F[6])
    self.params['beta7'] = np.zeros(F[6])
    self.params['W8'] = np.random.randn(F[7], F[6], HH, WW) / np.sqrt(F[6] / 2)
    self.params['b8'] = np.zeros(F[7])
    self.params['gamma8'] = np.ones(F[7])
    self.params['beta8'] = np.zeros(F[7])
    ### FullyConnected
    self.params['W9'] = np.random.randn(F[7] * H / 32 * W / 32, hidden_dim) / np.sqrt((F[7] * H / 32 * W / 32) / 2)
    self.params['b9'] = np.zeros(hidden_dim)
    self.params['gamma9'] = np.ones(hidden_dim)
    self.params['beta9'] = np.zeros(hidden_dim)
    self.params['W10'] = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim / 2)
    self.params['b10'] = np.zeros(hidden_dim)
    self.params['gamma10'] = np.ones(hidden_dim)
    self.params['beta10'] = np.zeros(hidden_dim)
    self.params['W11'] = np.random.randn(hidden_dim, num_classes) / np.sqrt(hidden_dim / 2)
    self.params['b11'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    W1, b1, gamma1, beta1 = self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1']
    W2, b2, gamma2, beta2 = self.params['W2'], self.params['b2'], self.params['gamma2'], self.params['beta2']
    W3, b3, gamma3, beta3 = self.params['W3'], self.params['b3'], self.params['gamma3'], self.params['beta3']
    W4, b4, gamma4, beta4 = self.params['W4'], self.params['b4'], self.params['gamma4'], self.params['beta4']
    W5, b5, gamma5, beta5 = self.params['W5'], self.params['b5'], self.params['gamma5'], self.params['beta5']
    W6, b6, gamma6, beta6 = self.params['W6'], self.params['b6'], self.params['gamma6'], self.params['beta6']
    W7, b7, gamma7, beta7 = self.params['W7'], self.params['b7'], self.params['gamma7'], self.params['beta7']
    W8, b8, gamma8, beta8 = self.params['W8'], self.params['b8'], self.params['gamma8'], self.params['beta8']
    W9, b9, gamma9, beta9 = self.params['W9'], self.params['b9'], self.params['gamma9'], self.params['beta9']
    W10, b10, gamma10, beta10 = self.params['W10'], self.params['b10'], self.params['gamma10'], self.params['beta10']
    W11, b11 = self.params['W11'], self.params['b11']


    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    for bn_param in self.bn_params:
      bn_param[mode] = mode

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    F = (64, 128, 256, 256, 512, 512, 512, 512)
    (N, C, H, W) = X.shape

    # normalized X
    transposed_X = X.transpose(1, 0, 2, 3)
    norm_X = np.empty(transposed_X.shape)
    mean_X = transposed_X.mean(axis=(1, 2, 3))
    norm_X[0] = mean_X[0]
    norm_X[1] = mean_X[1]
    norm_X[2] = mean_X[2]

    norm_X = (transposed_X - norm_X).transpose(1, 0, 2, 3)

    # conv - batch - relu - pool
    out1, cache1 = conv_batch_relu_pool_forward(norm_X, W1, b1, conv_param, pool_param, gamma1, beta1, self.bn_params[0])
    out2, cache2 = conv_batch_relu_pool_forward(out1, W2, b2, conv_param, pool_param, gamma2, beta2, self.bn_params[1])

    # conv - batch - relu - conv - batch - relu - pool
    out3, cache3 = conv_batch_relu_forward(out2, W3, b3, conv_param, gamma3, beta3, self.bn_params[2])
    out4, cache4 = conv_batch_relu_pool_forward(out3, W4, b4, conv_param, pool_param, gamma4, beta4, self.bn_params[3])
    out5, cache5 = conv_batch_relu_forward(out4, W5, b5, conv_param, gamma5, beta5, self.bn_params[4])
    out6, cache6 = conv_batch_relu_pool_forward(out5, W6, b6, conv_param, pool_param, gamma6, beta6, self.bn_params[5])
    out7, cache7 = conv_batch_relu_forward(out6, W7, b7, conv_param, gamma7, beta7, self.bn_params[6])
    out8, cache8 = conv_batch_relu_pool_forward(out7, W8, b8, conv_param, pool_param, gamma8, beta8, self.bn_params[7])

    # affine - batch - relu
    out9, cache9 = affine_batch_relu_forward(out8.reshape(N, -1), W9, b9, gamma9, beta9, self.bn_params[8])
    out10, cache10 = affine_batch_relu_forward(out9, W10, b10, gamma10, beta10, self.bn_params[9])

    # affine
    out11, cache11 = affine_forward(out10, W11, b11)
    scores = out11
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dout11 = softmax_loss(out11, y)
    # add regularization factor
    for x in xrange(self.num_layers):
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W%d' % (x + 1)]))

    # affine backward
    dout10, grads['W11'], grads['b11'] = affine_backward(dout11, cache11)

    # affine - batch -relu backward
    dout9, grads['W10'], grads['b10'], grads['gamma10'], grads['beta10'] = affine_batch_relu_backward(dout10, cache10)
    dout8, grads['W9'], grads['b9'], grads['gamma9'], grads['beta9'] = affine_batch_relu_backward(dout9, cache9)

    # conv - batch - relu - conv - batch - relu - pool backward
    dout7, grads['W8'], grads['b8'], grads['gamma8'], grads['beta8'] = conv_batch_relu_pool_backward(dout8.reshape(N, F[7], H / 32, W / 32), cache8)
    dout6, grads['W7'], grads['b7'], grads['gamma7'], grads['beta7'] = conv_batch_relu_backward(dout7, cache7)
    dout5, grads['W6'], grads['b6'], grads['gamma6'], grads['beta6'] = conv_batch_relu_pool_backward(dout6, cache6)
    dout4, grads['W5'], grads['b5'], grads['gamma5'], grads['beta5'] = conv_batch_relu_backward(dout5, cache5)
    dout3, grads['W4'], grads['b4'], grads['gamma4'], grads['beta4'] = conv_batch_relu_pool_backward(dout4, cache4)
    dout2, grads['W3'], grads['b3'], grads['gamma3'], grads['beta3'] = conv_batch_relu_backward(dout3, cache3)

    # conv - batch - relu - pool
    dout1, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = conv_batch_relu_pool_backward(dout2, cache2)
    dX, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = conv_batch_relu_pool_backward(dout1, cache1)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
