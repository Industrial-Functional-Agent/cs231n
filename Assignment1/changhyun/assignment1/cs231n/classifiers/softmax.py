import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  for i in xrange(N):
      unnormalized_log_probability = np.dot(X[i, :], W)
      unnormalized_probability = np.exp(unnormalized_log_probability)
      probability = unnormalized_probability / np.sum(unnormalized_probability)
      for j in xrange(C):
          if j == y[i]:
              loss -= np.log(probability[j])
              dW[:, j] += (probability[j] - 1) * X[i, :]
          else:
              dW[:, j] += probability[j] * X[i, :]
  
  loss /= N
  dW /= N

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  # gradient: +probability
  unnormalized_log_probability = np.dot(X, W)
  unnormalized_probability = np.exp(unnormalized_log_probability)
  probability = unnormalized_probability / np.sum(unnormalized_probability, 1).reshape(N, 1)
  dW += np.dot(X.transpose(), probability)
  # gradient: -1
  label_count = np.zeros_like(probability)
  label_count[xrange(N), y] = 1
  dW -= np.dot(X.transpose(), label_count)
  # loss
  loss = -np.sum(np.log(probability[xrange(N), y]))

  loss /= N
  dW /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

