import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    out2, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
    scores = out2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dout2 = softmax_loss(out2, y)
    loss += 0.5 * self.reg * (np.sum(np.square(self.params['W2'])) + 
                             np.sum(np.square(self.params['W1']))) 
    dout1, grads['W2'], grads['b2'] = affine_backward(dout2, cache2)
    dX1, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)
    grads['W2'] += self.reg * self.params['W2']
    grads['W1'] += self.reg * self.params['W1']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

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
    self.params['W1'] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
    self.params['b1'] = np.zeros(hidden_dims[0])
    if self.use_batchnorm:
        self.params['gamma1'] = np.ones(hidden_dims[0])
        self.params['beta1'] = np.zeros(hidden_dims[0])
    for x in xrange(self.num_layers - 2):
        self.params['W%d' % (x + 2)] = np.random.randn(hidden_dims[x],
                                               hidden_dims[x + 1]) * weight_scale
        self.params['b%d' % (x + 2)] = np.zeros(hidden_dims[x + 1])
        if self.use_batchnorm:
            self.params['gamma%d' % (x + 2)] = np.ones(hidden_dims[x + 1])
            self.params['beta%d' % (x + 2)] = np.zeros(hidden_dims[x + 1])
    self.params['W%d' % (self.num_layers)] = np.random.randn(
                    hidden_dims[self.num_layers - 2], num_classes) * weight_scale
    self.params['b%d' % (self.num_layers)] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
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

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

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
    hidden_state = {}
    if self.use_batchnorm:
        hidden_state['out1'], hidden_state['cache1'] = affine_batch_relu_forward(X,
                                              self.params['W1'], self.params['b1'],
                                              self.params['gamma1'], self.params['beta1'],
                                              self.bn_params[0])
    else:
        hidden_state['out1'], hidden_state['cache1'] = affine_relu_forward(X,
                                              self.params['W1'], self.params['b1'])
        
    if self.use_dropout:
        hidden_state['out1'], hidden_state['dropout_cache1'] = \
                    dropout_forward(hidden_state['out1'], self.dropout_param)
    for x in xrange(self.num_layers - 2):
        weight = self.params['W%d' % (x + 2)]
        bias = self.params['b%d' % (x + 2)]
        past_out = hidden_state['out%d' % (x + 1)]
        if self.use_batchnorm:
            gamma = self.params['gamma%d' % (x + 2)]
            beta = self.params['beta%d' % (x + 2)]
            out, cache = affine_batch_relu_forward(past_out, weight, bias, 
                                               gamma, beta, self.bn_params[x + 1])
        else:
            out, cache = affine_relu_forward(past_out, weight, bias)
        
        if self.use_dropout:
            out, hidden_state['dropout_cache%d' % (x + 2)] = dropout_forward(out,
                                                                 self.dropout_param)
            
        hidden_state['out%d' % (x + 2)] = out
        hidden_state['cache%d' % (x + 2)] = cache
        
        
    weight = self.params['W%d' % (self.num_layers)]
    bias = self.params['b%d' % (self.num_layers)]
    past_out = hidden_state['out%d' % (self.num_layers - 1)]
    out, cache = affine_forward(past_out, weight, bias)
    
    hidden_state['out%d' % (self.num_layers)] = out
    hidden_state['cache%d' % (self.num_layers)] = cache
    scores = out
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
    loss, dout_last = softmax_loss(out, y)
    for x in xrange(self.num_layers):
        loss += 0.5 * self.reg * np.sum(np.square(self.params['W%d' % (x + 1)]))

    
    dout_temp, w_grads, b_grads = affine_backward(dout_last, cache)
    dout = {}
    dout['dout%d' % (self.num_layers - 1)] = dout_temp
    grads['W%d' % (self.num_layers)] = w_grads + self.reg * \
                                    self.params['W%d' % (self.num_layers)]
    grads['b%d' % (self.num_layers)] = b_grads
    for x in xrange(self.num_layers - 1):
        dout_past = dout['dout%d' % (self.num_layers - x - 1)]
        cache_past = hidden_state['cache%d' % (self.num_layers - x - 1)]
        
        if self.use_dropout:
            dout_past = dropout_backward(dout_past,
                        hidden_state['dropout_cache%d' % (self.num_layers - x - 1)])
        
        if self.use_batchnorm:  
            dout_temp, w_grads, b_grads, gamma_grads, beta_grads = \
                            affine_batch_relu_backward(dout_past, cache_past)
            grads['gamma%d' % (self.num_layers - x - 1)] = gamma_grads
            grads['beta%d' % (self.num_layers - x - 1)] = beta_grads
        else:
            dout_temp, w_grads, b_grads = affine_relu_backward(dout_past, cache_past)
        
        loss += 0.5 * self.reg * np.square(self.params['W%d' % (self.num_layers - x - 1)]).sum()
        dout['dout%d' % (self.num_layers - x - 2)] = dout_temp
        grads['W%d' % (self.num_layers - x - 1)] = w_grads + self.reg * \
                                    self.params['W%d' % (self.num_layers - x - 1)]
        grads['b%d' % (self.num_layers - x - 1)] = b_grads
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads



def affine_batch_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by batch normalization followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta, bn_prarm: Parameter for the batch normalization

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  fc_out, fc_cache = affine_forward(x, w, b)
  batch_out, batch_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
  relu_out, relu_cache = relu_forward(batch_out)
  cache = (fc_cache, batch_cache, relu_cache)
  return relu_out, cache

    

def affine_batch_relu_backward(dout, cache):
  """
  Backward pass for the affine-batch_relu convenience layer
  """
  fc_cache, batch_cache, relu_cache = cache
  dbatch_out = relu_backward(dout, relu_cache)
  dfc_out, dgamma, dbeta = batchnorm_backward_alt(dbatch_out, batch_cache)
  dx, dw, db = affine_backward(dfc_out, fc_cache)
  return dx, dw, db, dgamma, dbeta
